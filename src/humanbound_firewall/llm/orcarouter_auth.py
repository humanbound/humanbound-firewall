# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""OrcaRouter Cyber Inference credential seam.

Two ways to obtain the same thing — an OrcaRouter API key (``sk-orca-…``) belonging to the user:

- **API key adapter** (:func:`api_key_credential`): the user pastes an existing key, or supplies it
  through the deployment's environment/config (``HUMANBOUND_FIREWALL_API_KEY``).
- **PKCE adapter** (:func:`connect_with_pkce` / :class:`OrcaPkceConnector`): the user authorizes in
  a browser with OAuth 2.0 + PKCE (S256, loopback redirect), and the exchange returns a normal
  API key. There is no client secret and nothing to pre-register.

Both adapters produce an :class:`OrcaCredential`, which is the only thing the provider adapter and
the model catalog consume — neither cares which adapter produced it.

A PKCE-issued key is durable, not a refresh token: it is reused until OrcaRouter revokes it, a
``401`` from the relay means the user must re-authorize, and nothing here ever "refreshes" it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

AUTH_BASE_URL = "https://www.orcarouter.ai"
API_BASE_URL = "https://api.orcarouter.ai/v1"
AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"
APP_NAME = "humanbound-firewall"
DEFAULT_SCOPE = "api"
GRANTED_SCOPES = ("api", "connector")
# The auth code is single-use with a 10 minute TTL; the listener gives up a little sooner.
AUTHORIZE_TIMEOUT = 600
EXCHANGE_TIMEOUT = 30
KEY_PREFIX = "sk-orca-"


AUTH_HINT = (
    "OrcaRouter Cyber Inference — for security teams running authorized, scoped work. "
    "Two ways to connect: use an existing sk-orca-… key "
    "(humanbound llm orcarouter api --api-key sk-orca-… / HUMANBOUND_FIREWALL_API_KEY), or "
    "connect an account with OAuth 2.0 + PKCE "
    "(humanbound llm orcarouter connect). https://www.orcarouter.ai/cyber · "
    "cyber model catalog: https://www.orcarouter.ai/models?tier=cyber"
)


class OrcaAuthError(Exception):
    """A PKCE authorization failed. The message is safe to show: it never holds a secret."""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def pkce_pair() -> tuple[str, str]:
    """A fresh (verifier, S256 challenge) from the OS CSPRNG. New for every attempt."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def new_state() -> str:
    return _b64url(secrets.token_bytes(16))


def _origin(value: str | None, default: str, name: str) -> str:
    """A configured origin: HTTPS for remote hosts, HTTP only for loopback. No path guessing."""
    origin = (value or default).strip().rstrip("/")
    lowered = origin.lower()
    if lowered.startswith("http://"):
        host = lowered[len("http://") :].split("/", 1)[0].rsplit(":", 1)[0]
        if host not in ("localhost", "127.0.0.1", "[::1]", "::1"):
            raise ValueError(f"{name} must use HTTPS for a remote host: {origin!r}")
    elif not lowered.startswith("https://"):
        raise ValueError(f"{name} must be an absolute http(s) URL: {origin!r}")
    return origin


def auth_base_url() -> str:
    """Consent + exchange origin. Explicit auth override, then the shared self-hosted base."""
    return _origin(
        os.environ.get("ORCA_AUTH_BASE_URL") or os.environ.get("ORCA_BASE_URL"),
        AUTH_BASE_URL,
        "ORCA_AUTH_BASE_URL",
    )


def api_base_url() -> str:
    """Inference + model-catalog origin (includes ``/v1``). Never derived from the auth origin."""
    return _origin(
        os.environ.get("ORCA_API_BASE_URL") or os.environ.get("ORCA_BASE_URL"),
        API_BASE_URL,
        "ORCA_API_BASE_URL",
    )


def authorize_url(challenge: str, state: str, callback_url: str, scope: str = DEFAULT_SCOPE) -> str:
    """The consent URL. Only the challenge travels here — never the verifier."""
    params = {
        "callback_url": callback_url,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "app_name": APP_NAME,
        "scope": scope,
    }
    query = "&".join(f"{k}={requests.utils.quote(v, safe='')}" for k, v in params.items())
    return f"{auth_base_url()}{AUTHORIZE_PATH}?{query}"


def default_credentials_path() -> Path:
    """Where a PKCE-issued key is persisted: the platform config dir the user already owns."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "humanbound-firewall" / "orcarouter-credentials.json"


@dataclass(frozen=True)
class OrcaCredential:
    """The one credential shape every downstream consumer sees, whatever produced it."""

    api_key: str
    source: str  # "api_key" | "pkce"
    scope: str | None = None
    user_id: str | None = None
    generation: int = 0  # bumped on every replacement; stale failures must match it

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.startswith(KEY_PREFIX):
            raise ValueError("An OrcaRouter API key starts with 'sk-orca-'.")

    def __repr__(self) -> str:  # never print the key in logs or tracebacks
        return (
            f"OrcaCredential(source={self.source!r}, scope={self.scope!r}, "
            f"user_id={self.user_id!r}, generation={self.generation}, api_key='sk-orca-***')"
        )


# ────────────────────────────────────────────────────────────────
# Adapter 1: pasted API key
# ────────────────────────────────────────────────────────────────


def api_key_credential(api_key: str | None) -> OrcaCredential:
    """Wrap an existing user API key. Never sends it anywhere; it is only stored in the Provider."""
    return OrcaCredential(api_key=(api_key or "").strip(), source="api_key")


# ────────────────────────────────────────────────────────────────
# Adapter 2: OAuth 2.0 + PKCE (Flow A, loopback redirect)
# ────────────────────────────────────────────────────────────────

_EXCHANGE_ERRORS = {
    400: "OrcaRouter rejected the exchange (400): the code challenge method did not match.",
    403: "OrcaRouter rejected the authorization (403): the code is unknown, expired, already used, "
    "or the verifier did not match. Start the connection again.",
    429: "OrcaRouter rate-limited the authorization (429). Try again later.",
}


class OrcaPkceConnector:
    """Flow A: listen on 127.0.0.1, open the consent page, receive the code, exchange it.

    The client is a local Python process that can bind a loopback port, so the redirect flow is
    the shortest path for its user. The listener, browser opener and HTTP poster are injectable
    so tests can drive the whole flow against a local fake auth server.
    """

    def __init__(
        self,
        *,
        scope: str = DEFAULT_SCOPE,
        timeout: int = AUTHORIZE_TIMEOUT,
        open_browser: bool = True,
        credentials_path: Path | str | None = None,
        on_authorize_url: Callable[[str], None] | None = None,
        browser_open: Callable[[str], bool] | None = None,
        http_post: Callable[..., requests.Response] | None = None,
    ):
        self.scope = scope
        self.timeout = timeout
        self.open_browser = open_browser
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self.on_authorize_url = on_authorize_url
        self.browser_open = browser_open or webbrowser.open
        self.http_post = http_post or requests.post

    # -- authorize --------------------------------------------------------------

    def _callback(self, challenge: str, state: str) -> str:
        """Open a loopback listener, print/open the consent URL, return the code."""
        import http.server
        import threading

        captured: dict[str, str] = {}
        done = threading.Event()
        mismatch = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — stdlib name
                from urllib.parse import parse_qs, urlparse

                url = urlparse(self.path)
                if url.path != "/cb":
                    self.send_response(404)
                    self.end_headers()
                    return
                query = parse_qs(url.query)
                ok = secrets.compare_digest(query.get("state", [""])[0], state)
                body = (
                    b"<p>Connected. You can close this tab.</p>"
                    if ok
                    else b"<p>Authorization could not be verified. You can close this tab.</p>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if not ok:
                    mismatch.set()
                    done.set()
                    return
                if "error" in query:
                    captured["error"] = query["error"][0]
                elif "code" in query:
                    captured["code"] = query["code"][0]
                done.set()

            def log_message(self, *args):  # the code never goes to stderr through this path
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = authorize_url(challenge, state, f"http://127.0.0.1:{port}/cb", self.scope)
            if self.on_authorize_url:
                self.on_authorize_url(url)
            if self.open_browser:
                self.browser_open(url)
            if not done.wait(self.timeout):
                raise OrcaAuthError("Authorization timed out. Start the connection again.")
        finally:
            server.shutdown()
            server.server_close()
        if mismatch.is_set():
            raise OrcaAuthError("The redirect state did not match; the authorization was ignored.")
        if "error" in captured:
            raise OrcaAuthError(f"Authorization was not granted ({captured['error']}).")
        if "code" not in captured:
            raise OrcaAuthError("The redirect carried no authorization code.")
        return captured["code"]

    # -- exchange ---------------------------------------------------------------

    def _exchange(self, code: str, verifier: str) -> dict:
        # The exchange lives on the auth origin; /v1/auth/keys on the inference origin is a 404.
        url = f"{auth_base_url()}{EXCHANGE_PATH}"
        try:
            resp = self.http_post(
                url,
                json={
                    "code": code,
                    "code_verifier": verifier,
                    "code_challenge_method": "S256",
                },
                timeout=EXCHANGE_TIMEOUT,
            )
        except requests.RequestException as e:
            raise OrcaAuthError(
                f"Could not reach OrcaRouter to exchange the code ({type(e).__name__})."
            ) from None
        if resp.status_code != 200:
            raise OrcaAuthError(_EXCHANGE_ERRORS.get(resp.status_code, "Code exchange failed."))
        try:
            data = resp.json()
        except ValueError:
            raise OrcaAuthError("OrcaRouter returned an unreadable exchange response.") from None
        if not isinstance(data, dict) or not isinstance(data.get("key"), str):
            raise OrcaAuthError("OrcaRouter's exchange response carried no key.")
        return data

    def connect(self, *, reuse: bool = True) -> OrcaCredential:
        """Reuse the persisted key if there is one, else run the PKCE flow and persist the key."""
        if reuse:
            stored = self.load()
            if stored is not None:
                return stored
        verifier, challenge = pkce_pair()
        state = new_state()
        code = self._callback(challenge, state)
        data = self._exchange(code, verifier)
        granted = data.get("scope")
        if granted not in GRANTED_SCOPES or (self.scope == "api" and granted != "api"):
            # Read what was granted, not what was requested.
            raise OrcaAuthError(f"Granted scope {granted!r} does not permit inference.")
        credential = OrcaCredential(
            api_key=data["key"],
            source="pkce",
            scope=granted,
            user_id=str(data["user_id"]) if data.get("user_id") else None,
            generation=1,
        )
        self.save(credential)
        return credential

    def load(self) -> OrcaCredential | None:
        """The persisted key, or None. A corrupted store is ignored, never fatal."""
        path = self.credentials_path or default_credentials_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OrcaCredential(
                api_key=data["api_key"],
                source="pkce",
                scope=data.get("scope"),
                user_id=data.get("user_id"),
                generation=int(data.get("generation", 1)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, credential: OrcaCredential) -> None:
        path = self.credentials_path or default_credentials_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "api_key": credential.api_key,
                    "scope": credential.scope,
                    "user_id": credential.user_id,
                    "generation": credential.generation,
                },
                f,
            )
        os.replace(tmp, path)

    def clear(self) -> None:
        """Remove the persisted key (logout). Never touches a pasted API key."""
        path = self.credentials_path or default_credentials_path()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def connect_with_pkce(**kwargs) -> OrcaCredential:
    """Convenience: run the PKCE adapter with the given options."""
    return OrcaPkceConnector(**kwargs).connect()


__all__ = [
    "API_BASE_URL",
    "AUTH_HINT",
    "AUTH_BASE_URL",
    "OrcaAuthError",
    "OrcaCredential",
    "OrcaPkceConnector",
    "api_base_url",
    "api_key_credential",
    "auth_base_url",
    "authorize_url",
    "connect_with_pkce",
    "default_credentials_path",
    "new_state",
    "pkce_pair",
]
