# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""OrcaRouter Cyber Inference provider and its credential seam.

Covers the API-key adapter and the OAuth 2.0 + PKCE adapter (Flow A, loopback) against a local
fake auth server, the provider adapter's routing, the model catalog and its verified fallback,
and one live check of the real endpoint that runs only when ORCAROUTER_API_KEY is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import urllib.parse
import urllib.request

import pytest
import requests

from humanbound_firewall import Firewall
from humanbound_firewall.llm import get_llm_pinger, get_llm_streamer
from humanbound_firewall.llm.models import Provider, ProviderIntegration, ProviderName
from humanbound_firewall.llm.orcarouter import (
    VERIFIED_FALLBACK_MODELS,
    LLMPinger,
    LLMStreamer,
    _is_chat_model,
    _normalise_model,
    chat_completions_url,
    credential_for,
    fetch_models,
    list_models,
)
from humanbound_firewall.llm.orcarouter_auth import (
    API_BASE_URL,
    AUTH_BASE_URL,
    OrcaAuthError,
    OrcaCredential,
    OrcaPkceConnector,
    api_base_url,
    api_key_credential,
    auth_base_url,
    authorize_url,
    new_state,
    pkce_pair,
)

FAKE_KEY = "sk-orca-" + "fake0" + "1234567890abcdef"


def _provider(name: ProviderName, model=None, **kw) -> Provider:
    return Provider(
        name=name,
        integration=ProviderIntegration(
            api_key=kw.get("api_key", FAKE_KEY),
            model=model or "orcarouter/auto",
            endpoint=kw.get("endpoint"),
            reasoning_effort=kw.get("reasoning_effort"),
        ),
    )


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "ORCA_BASE_URL",
        "ORCA_AUTH_BASE_URL",
        "ORCA_API_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "ORCAROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ────────────────────────────────────────────────────────────────
# Origins and PKCE primitives
# ────────────────────────────────────────────────────────────────


def test_default_origins_are_distinct_and_auth_is_not_under_v1(clean_env):
    assert auth_base_url() == "https://www.orcarouter.ai"
    assert api_base_url() == "https://api.orcarouter.ai/v1"
    assert AUTH_BASE_URL != API_BASE_URL
    url = authorize_url("chal", "st", "http://127.0.0.1:1234/cb")
    assert url.startswith("https://www.orcarouter.ai/auth?")
    assert "/v1/auth" not in url
    assert "code_challenge_method=S256" in url


def test_explicit_overrides_win_over_the_shared_base(clean_env):
    clean_env.setenv("ORCA_BASE_URL", "https://self.example.ai")
    assert auth_base_url() == "https://self.example.ai"
    assert api_base_url() == "https://self.example.ai"
    clean_env.setenv("ORCA_AUTH_BASE_URL", "https://auth.example.ai")
    clean_env.setenv("ORCA_API_BASE_URL", "https://relay.example.ai/v1")
    assert auth_base_url() == "https://auth.example.ai"
    assert api_base_url() == "https://relay.example.ai/v1"
    assert chat_completions_url() == "https://relay.example.ai/v1/chat/completions"


def test_remote_origins_must_be_https_and_http_is_loopback_only(clean_env):
    clean_env.setenv("ORCA_API_BASE_URL", "http://api.orcarouter.ai/v1")
    with pytest.raises(ValueError, match="HTTPS"):
        api_base_url()
    clean_env.setenv("ORCA_API_BASE_URL", "http://127.0.0.1:8080/v1")
    assert api_base_url() == "http://127.0.0.1:8080/v1"


def test_pkce_pair_is_s256_without_padding_and_fresh_each_time():
    v1, c1 = pkce_pair()
    v2, c2 = pkce_pair()
    assert v1 != v2 and c1 != c2
    assert "=" not in c1 and "=" not in v1
    expected = (
        __import__("base64")
        .urlsafe_b64encode(hashlib.sha256(v1.encode()).digest())
        .decode()
        .rstrip("=")
    )
    assert c1 == expected
    assert new_state() != new_state()


# ────────────────────────────────────────────────────────────────
# Fake OrcaRouter auth + relay server
# ────────────────────────────────────────────────────────────────


class FakeOrca:
    """A local stand-in for the consent exchange and the relay, on two different ports."""

    def __init__(self, handler_factory):
        self.exchange_requests: list[dict] = []
        self.inference_requests: list[dict] = []
        self.challenges: list[str] = []
        self.exchange_status = 200
        self.scope = "api"
        self.key = FAKE_KEY
        self.models_status = 200
        self._factory = handler_factory
        self.auth = self._serve(self._auth_handler)
        self.api = self._serve(self._api_handler)

    def _serve(self, handler):
        import http.server

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._factory(handler))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _auth_handler(self, path, query, body):
        if path == "/api/v1/auth/keys":
            self.exchange_requests.append(body)
            if self.exchange_status != 200:
                return self.exchange_status, {"error": "invalid_grant"}
            return 200, {"key": self.key, "user_id": "42", "scope": self.scope}
        return 404, {"error": "not_found"}

    def _api_handler(self, path, query, body):
        if path == "/v1/models":
            self.inference_requests.append({"path": path, "query": query})
            if self.models_status != 200:
                return self.models_status, {}
            return 200, {"data": CATALOG}
        if path == "/v1/chat/completions":
            self.inference_requests.append({"path": path, "body": body})
            return 200, {
                "choices": [{"message": {"role": "assistant", "content": "A allowed"}}],
            }
        return 404, {}

    @property
    def auth_url(self) -> str:
        return f"http://127.0.0.1:{self.auth.server_address[1]}"

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api.server_address[1]}/v1"

    def close(self):
        for s in (self.auth, self.api):
            s.shutdown()
            s.server_close()


def _make_handler(handler):
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                body = {}
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            status, payload = handler(url.path, query, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            status, payload = handler(url.path, query, {})
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            return

    return H


CATALOG = [
    {
        "id": "orca/orcacyber-zero-1.5",
        "supported_endpoint_types": ["openai"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "context_length": 1_000_000,
    },
    {
        "id": "orcarouter/auto",
        "supported_endpoint_types": ["openai", "anthropic", "gemini"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    },
    {
        "id": "vendor/vision-chat",
        "supported_endpoint_types": ["openai"],
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
    },
    {
        "id": "vendor/image-gen",
        "supported_endpoint_types": ["image-generation"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
    },
    {
        "id": "vendor/embed",
        "supported_endpoint_types": ["embeddings"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    },
    {
        "id": "vendor/reranker",
        "supported_endpoint_types": ["jina-rerank"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    },
    {
        "id": "vendor/video",
        "supported_endpoint_types": ["openai-video"],
        "architecture": {"input_modalities": ["text"], "output_modalities": ["video"]},
    },
    "not-a-model",
]


@pytest.fixture
def fake(monkeypatch):
    server = FakeOrca(_make_handler)
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", server.auth_url)
    monkeypatch.setenv("ORCA_API_BASE_URL", server.api_url)
    yield server
    server.close()


def _drive_redirect(
    connector: OrcaPkceConnector, fake: FakeOrca, *, deny=False, tamper_state=False
):
    """Play the browser: parse the consent URL the connector printed and hit its callback."""
    seen: dict = {}
    seen["ready"] = threading.Event()

    def on_url(url):
        seen["url"] = url
        seen["ready"].set()

    connector.on_authorize_url = on_url
    result = {}

    def run():
        try:
            result["cred"] = connector.connect(reuse=False)
        except OrcaAuthError as e:
            result["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    assert seen["ready"].wait(10), "the connector never printed the consent URL"
    connector.timeout = 20
    url = urllib.parse.urlparse(seen["url"])
    params = urllib.parse.parse_qs(url.query)
    assert params["callback_url"][0].startswith("http://127.0.0.1:")
    assert "code_verifier" not in seen["url"] and "verifier" not in seen["url"]
    callback = params["callback_url"][0]
    fake.challenges.append(params["code_challenge"][0])
    state = params["state"][0]
    query = {"state": ("other" if tamper_state else state)}
    if deny:
        query["error"] = "access_denied"
    else:
        query["code"] = "fake-code"
    with urllib.request.urlopen(f"{callback}?{urllib.parse.urlencode(query)}", timeout=5) as r:
        body = r.read().decode()
    thread.join(10)
    return result, body


def test_api_key_adapter_produces_the_same_credential_shape_as_pkce():
    pasted = api_key_credential(FAKE_KEY)
    assert pasted.source == "api_key"
    assert pasted.api_key == FAKE_KEY
    with pytest.raises(ValueError):
        api_key_credential("not-a-key")


def test_pkce_flow_a_authorize_callback_exchange_and_persist(tmp_path, fake):
    store = tmp_path / "orca.json"
    connector = OrcaPkceConnector(credentials_path=store, open_browser=False, timeout=20)
    result, body = _drive_redirect(connector, fake)
    cred = result["cred"]
    assert isinstance(cred, OrcaCredential) and cred.source == "pkce"
    assert cred.api_key == FAKE_KEY and cred.scope == "api" and cred.user_id == "42"
    assert "close this tab" in body
    # Exchange went to the auth origin, POSTed JSON, with S256 and the verifier.
    assert len(fake.exchange_requests) == 1
    data = fake.exchange_requests[0]
    assert data["code"] == "fake-code"
    assert data["code_challenge_method"] == "S256"
    assert "code_verifier" in data
    assert fake.inference_requests == []  # the exchange never hit the /v1 relay
    # Persisted with the user's own permissions, and reused instead of a second login.
    assert store.stat().st_mode & 0o777 == 0o600
    reuse = connector.connect()
    assert reuse.api_key == FAKE_KEY
    assert len(fake.exchange_requests) == 1
    # Logout clears only the store.
    connector.clear()
    assert connector.load() is None


def test_pkce_denial_ends_cleanly_without_calling_exchange(tmp_path, fake):
    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20)
    result, _ = _drive_redirect(connector, fake, deny=True)
    assert isinstance(result["error"], OrcaAuthError)
    assert "access_denied" in str(result["error"])
    assert fake.exchange_requests == []


def test_pkce_state_mismatch_is_rejected_before_the_code_is_used(tmp_path, fake):
    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20)
    result, _ = _drive_redirect(connector, fake, tamper_state=True)
    assert isinstance(result["error"], OrcaAuthError)
    assert "state" in str(result["error"])
    assert fake.exchange_requests == []


def test_pkce_exchange_403_is_a_terminal_actionable_error(tmp_path, fake):
    fake.exchange_status = 403
    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20)
    result, _ = _drive_redirect(connector, fake)
    assert isinstance(result["error"], OrcaAuthError)
    assert "403" in str(result["error"])
    assert not (tmp_path / "c.json").exists()


def test_pkce_scope_downgrade_is_reported_not_assumed(tmp_path, fake):
    fake.scope = "connector"
    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20)
    result, _ = _drive_redirect(connector, fake)
    assert isinstance(result["error"], OrcaAuthError)
    assert "scope" in str(result["error"])


def test_pkce_exchange_network_failure_never_echoes_the_verifier(tmp_path, fake):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20, http_post=boom)
    result, _ = _drive_redirect(connector, fake)
    assert isinstance(result["error"], OrcaAuthError)
    assert "ConnectionError" in str(result["error"])
    assert "code_verifier" not in str(result["error"])


def test_secrets_are_not_logged(tmp_path, fake, caplog):
    caplog.set_level(logging.DEBUG)
    connector = OrcaPkceConnector(credentials_path=tmp_path / "c.json", timeout=20)
    result, _ = _drive_redirect(connector, fake)
    verifier = fake.exchange_requests[0]["code_verifier"]
    text = caplog.text
    assert verifier not in text
    assert result["cred"].api_key not in text
    assert result["cred"].api_key not in repr(result["cred"])
    assert "sk-orca-***" in repr(result["cred"])


def test_corrupted_store_is_ignored(tmp_path):
    store = tmp_path / "c.json"
    store.write_text("{not json")
    assert OrcaPkceConnector(credentials_path=store).load() is None
    store.write_text(json.dumps({"api_key": "garbage"}))
    assert OrcaPkceConnector(credentials_path=store).load() is None


# ────────────────────────────────────────────────────────────────
# Provider routing: both adapters reach the same inference path
# ────────────────────────────────────────────────────────────────


def test_both_auth_choices_are_registered_and_route_to_the_same_adapter():
    for name in (ProviderName.ORCAROUTER, ProviderName.ORCAROUTER_OAUTH):
        assert isinstance(get_llm_streamer(_provider(name)), LLMStreamer)
        assert isinstance(get_llm_pinger(_provider(name)), LLMPinger)
        assert isinstance(get_llm_streamer({"name": name.value, "integration": {}}), LLMStreamer)


def test_oauth_choice_uses_the_persisted_pkce_key(tmp_path, monkeypatch):
    store = tmp_path / "c.json"
    OrcaPkceConnector(credentials_path=store).save(
        OrcaCredential(api_key=FAKE_KEY, source="pkce", scope="api", generation=1)
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "humanbound-firewall").mkdir(exist_ok=True)
    store.rename(tmp_path / "humanbound-firewall" / "orcarouter-credentials.json")
    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER_OAUTH, api_key=""))
    assert streamer._session.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    key_choice = get_llm_streamer(_provider(ProviderName.ORCAROUTER))
    assert key_choice._session.headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_provider_from_env_names_route_orcarouter(monkeypatch, clean_env):
    from humanbound_firewall.firewall import _provider_from_env

    monkeypatch.setenv("HUMANBOUND_FIREWALL_PROVIDER", "orcarouter")
    monkeypatch.setenv("HUMANBOUND_FIREWALL_API_KEY", FAKE_KEY)
    prov = _provider_from_env()
    assert prov.name is ProviderName.ORCAROUTER and prov.integration.model == "orcarouter/auto"
    monkeypatch.setenv("HUMANBOUND_FIREWALL_PROVIDER", "orcarouter-oauth")
    assert _provider_from_env().name is ProviderName.ORCAROUTER_OAUTH


def test_credential_for_reflects_the_choice(fake):
    assert credential_for(_provider(ProviderName.ORCAROUTER)).source == "api_key"
    assert credential_for(_provider(ProviderName.ORCAROUTER_OAUTH)).source == "pkce"


class _Resp:
    def __init__(self, status, payload, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def close(self):
        return None


def _sse(payloads):
    import io

    lines = []
    for p in payloads:
        lines.append("data: " + json.dumps(p) + "\n\n")
    lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


def test_streamer_sends_openai_compatible_body_to_the_orca_router_endpoint(fake, monkeypatch):
    captured = {}

    def post(url, json=None, stream=False, timeout=None, **kw):
        captured.update(url=url, json=json)
        return _Resp(200, {}, "ok")

    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER, model="orca/orcacyber-zero-1.5"))
    streamer._session.post = post  # type: ignore[method-assign]
    streamer.ping("sys", "user", max_tokens=16)
    assert captured["url"] == f"{fake.api_url}/chat/completions"
    assert captured["json"]["model"] == "orca/orcacyber-zero-1.5"
    assert captured["json"]["stream"] is True
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}


def test_streamer_yields_chunks_the_judge_can_read(fake):
    from humanbound_firewall.firewall import _extract_token

    chunk = {"choices": [{"delta": {"content": "A"}}]}

    class R:
        status_code = 200

        def iter_lines(self, decode_unicode=True):
            yield "event: message"
            yield "data: " + json.dumps(chunk)
            yield "data: [DONE]"

        def close(self):
            pass

        request = None

    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER))
    streamer._session.post = lambda *a, **k: R()  # type: ignore[method-assign]
    tokens = [_extract_token(c) for c in streamer.ping("s", "u")]
    assert tokens[0] == "A"


def test_streamer_surfaces_an_error_event_without_the_key(fake):
    class R:
        status_code = 200
        request = None

        def iter_lines(self, decode_unicode=True):
            yield "event: error"
            yield "data: " + json.dumps({"error": {"message": f"bad {FAKE_KEY}"}})

        def close(self):
            pass

    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER))
    streamer._session.post = lambda *a, **k: R()  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as excinfo:
        list(streamer.ping("s", "u"))
    assert FAKE_KEY not in str(excinfo.value)


def test_pinger_round_trip_and_401_is_terminal(fake):
    pinger = get_llm_pinger(_provider(ProviderName.ORCAROUTER))
    assert pinger.ping("s", "u") == "A allowed"
    pinger = get_llm_pinger(_provider(ProviderName.ORCAROUTER))

    class Unauthorized(_Resp):
        pass

    import humanbound_firewall.llm.orcarouter as mod

    monkeypatch_post = mod.requests.post
    mod.requests.post = lambda *a, **k: _Resp(401, {"error": "revoked"}, text="revoked")
    try:
        with pytest.raises(mod.OrcaReauthRequired, match="connect"):
            pinger.ping("s", "u")
    finally:
        mod.requests.post = monkeypatch_post


def test_pinger_redacts_the_key_from_error_bodies(fake):
    import humanbound_firewall.llm.orcarouter as mod

    monkeypatch_post = mod.requests.post
    mod.requests.post = lambda *a, **k: _Resp(500, {}, text=f"bad key {FAKE_KEY}")
    try:
        with pytest.raises(Exception) as excinfo:
            get_llm_pinger(_provider(ProviderName.ORCAROUTER)).ping("s", "u")
    finally:
        mod.requests.post = monkeypatch_post
    assert FAKE_KEY not in str(excinfo.value)
    assert "sk-orca-***" in str(excinfo.value)


def test_streamer_error_body_never_carries_the_key(fake):
    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER))
    streamer._session.post = lambda *a, **k: _Resp(500, {}, text=f"upstream says {FAKE_KEY}")  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as excinfo:
        streamer.ping("s", "u")
    assert FAKE_KEY not in str(excinfo.value)


POLICY = """
name: ShopAssist
scope:
  business: Shopping assistant.
intents:
  restricted:
    - disclose contract prices
settings:
  mode: block
  timeout: 30
"""


def test_the_firewall_judge_streams_verdicts_from_orcarouter(tmp_path, fake, monkeypatch):
    """End to end through the real LLMStreamer: a Firewall whose Tier 3 judge is OrcaRouter."""
    replies = [
        {"choices": [{"delta": {"content": "C"}}]},
        {"choices": [{"delta": {"content": " restricted intent disclosed"}}]},
    ]

    class R:
        status_code = 200
        request = None

        def iter_lines(self, decode_unicode=True):
            for r in replies:
                yield "data: " + json.dumps(r)
            yield "data: [DONE]"

        def close(self):
            pass

    streamer = get_llm_streamer(_provider(ProviderName.ORCAROUTER))
    streamer._session.post = lambda *a, **k: R()  # type: ignore[method-assign]
    policy = tmp_path / "agent.yaml"
    policy.write_text(POLICY)
    fw = Firewall.from_config(policy, provider=_provider(ProviderName.ORCAROUTER), mode="block")
    fw._streamer = streamer
    result = fw.evaluate("What is the supplier's contract price?")
    assert result.tier in (3,), f"expected the LLM judge tier, got {result.tier}"
    assert result.raw_letter == "C"


def test_the_judge_is_selectable_from_the_environment(tmp_path, fake, monkeypatch):
    from humanbound_firewall.firewall import _provider_from_env

    fake.models_status = 500
    monkeypatch.setenv("HUMANBOUND_FIREWALL_PROVIDER", "orcarouter")
    monkeypatch.setenv("HUMANBOUND_FIREWALL_API_KEY", FAKE_KEY)
    monkeypatch.delenv("HUMANBOUND_FIREWALL_MODEL", raising=False)
    monkeypatch.delenv("HB_FIREWALL_MODEL", raising=False)
    provider = _provider_from_env()
    policy = tmp_path / "agent.yaml"
    policy.write_text(POLICY)
    fw = Firewall.from_config(policy, provider=provider, mode="block")
    assert provider.integration.model == "orcarouter/auto"  # the default works with any key
    assert fw._streamer.model == "orcarouter/auto"


# ────────────────────────────────────────────────────────────────
# Model catalog
# ────────────────────────────────────────────────────────────────


def test_live_catalog_is_authoritative_and_chat_filtered(fake):
    cred = credential_for(_provider(ProviderName.ORCAROUTER))
    models = fetch_models(cred)
    ids = [m["id"] for m in models]
    # Recommended order first (auto, then the cyber tier), then the rest of the catalog.
    assert ids == ["orcarouter/auto", "orca/orcacyber-zero-1.5", "vendor/vision-chat"]
    assert fake.inference_requests[0]["path"] == "/v1/models"
    assert fake.inference_requests[0]["query"] == {"capability": ["chat"]}
    assert all(m["context_length"] is not None for m in models if m["id"].startswith("orca/"))


def test_chat_filter_excludes_non_text_endpoints_fail_closed():
    for item in CATALOG:
        if not isinstance(item, dict):
            continue
        keep = _is_chat_model(_normalise_model(item))
        assert keep is (
            item["id"] in {"orca/orcacyber-zero-1.5", "orcarouter/auto", "vendor/vision-chat"}
        )
    # A record that does not declare its shape is treated as text-only chat and never multimodal.
    assert _normalise_model({"id": "x/y"})["input_modalities"] == ["text"]


def test_input_modalities_are_declared_not_guessed():
    vision = _normalise_model(CATALOG[2])
    assert "image" in vision["input_modalities"]
    assert "image" not in _normalise_model(CATALOG[1])["input_modalities"]


def test_catalog_failure_falls_back_to_the_verified_seed(fake):
    fake.models_status = 500
    cred = credential_for(_provider(ProviderName.ORCAROUTER))
    models, source = list_models(cred)
    assert source == "fallback"
    assert [m["id"] for m in models] == [m["id"] for m in VERIFIED_FALLBACK_MODELS]
    gpt = next(m for m in models if m["id"] == "openai/gpt-5.5")
    assert gpt["reasoning_efforts"] == ["low", "medium", "high", "xhigh"]


def test_catalog_network_error_falls_back(monkeypatch, clean_env):
    monkeypatch.setenv("ORCA_API_BASE_URL", "http://127.0.0.1:1/v1")
    models, source = list_models(api_key_credential(FAKE_KEY))
    assert source == "fallback" and len(models) == len(VERIFIED_FALLBACK_MODELS)


def test_live_catalog_does_not_mix_the_seed_in(fake):
    cred = credential_for(_provider(ProviderName.ORCAROUTER))
    models, source = list_models(cred)
    assert source == "live"
    assert all(m["id"] != "openai/gpt-5.5" for m in models)


# ────────────────────────────────────────────────────────────────
# Live: the real OrcaRouter endpoint through this adapter
# ────────────────────────────────────────────────────────────────


def _live_key():
    import os

    key = os.environ.get("ORCAROUTER_API_KEY", "")
    if not key.startswith("sk-orca-"):
        pytest.skip("ORCAROUTER_API_KEY not set")


def test_live_model_catalog_through_the_adapter():
    _live_key()
    import os

    cred = api_key_credential(os.environ["ORCAROUTER_API_KEY"])
    models, source = list_models(cred)
    assert source == "live", "live catalog request failed; check the network"
    ids = {m["id"] for m in models}
    assert ids, "live chat catalog is empty"
    # Every item was filtered through the chat capability check.
    assert all(_is_chat_model(m) for m in models)
    for m in models:
        assert "input_modalities" in m


def test_live_judge_call_through_the_provider():
    _live_key()
    import os

    cred = api_key_credential(os.environ["ORCAROUTER_API_KEY"])
    models, _ = list_models(cred)
    ids = {m["id"] for m in models}
    candidates = [
        "orca/orcacyber-ultra-1.0",
        "orca/orcacyber-zero-1.5",
        "deepseek/deepseek-v4.1-flash",
    ]
    model = next((m for m in candidates if m in ids), None)
    if model is None:
        pytest.skip("no known live model available to this key")
    provider = _provider(ProviderName.ORCAROUTER, model=model)
    pinger = get_llm_pinger(provider)
    try:
        reply = pinger.ping(
            "You are a strict gate. Reply with one word: ALLOW or BLOCK.",
            "The user asks: what is the store's return policy?",
            max_tokens=64,
        )
    except Exception as e:  # a key without the cyber tier gets 403 on the cyber models
        if model.startswith("orca/") and candidates[2] in ids:
            pytest.skip(f"cyber tier not available to this key ({type(e).__name__})")
        raise
    assert isinstance(reply, str) and reply.strip()
