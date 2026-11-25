# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""OrcaRouter Cyber Inference provider.

A named provider for security teams running authorized, scoped work: the Tier 3 judge, and the
ping used to generate training data, are sent to OrcaRouter's OpenAI-compatible endpoint at
``https://api.orcarouter.ai/v1`` (or the configured ``ORCA_API_BASE_URL``), which can route them to
cyber-tuned judges (``orcacyber-ultra-1.0``, ``orcacyber-zero-1.5``) or to ``orcarouter/auto``.

Two provider names exist so the two ways of obtaining the key stay distinct:

- ``orcarouter`` — the user pastes an existing ``sk-orca-…`` key
  (``HUMANBOUND_FIREWALL_API_KEY``).
- ``orcarouter-oauth`` — the key comes from ``humanbound_firewall llm orcarouter connect``
  (OAuth 2.0 + PKCE, see :mod:`.orcarouter_auth`) and is reused from the credential store.

Both end in the same :class:`~.orcarouter_auth.OrcaCredential` and the same adapter below.
"""

from __future__ import annotations

import json
import time
from os import getenv

import requests

from .openai import _note_rejected, _sampling_params
from .orcarouter_auth import OrcaCredential, OrcaPkceConnector, api_base_url

DEFAULT_MODEL = "orcarouter/auto"
ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0
MODELS_TIMEOUT = 15
# A catalog response cannot consume unbounded memory: at most this many items are read.
MAX_CATALOG_ITEMS = 2000

# When live discovery is unavailable a fresh install still starts. Every entry here is verified
# and carries the metadata the firewall relies on (text input; reasoning effort ladder where the
# model has one). Live discovery, when it succeeds, replaces this list entirely.
# What a security team selects first, in this order, when choosing a judge model in any UI that
# lists this provider's options: the default, then the cyber tier (business-verified accounts), then
# the general fallback. Discovered live, these are ranked to the top of the chat catalog.
RECOMMENDED_MODELS = (
    "orcarouter/auto",  # default: automatic routing, works with any key
    "orca/orcacyber-ultra-1.0",  # security-code scanning and multi-step reasoning over large repos
    "orca/orcacyber-zero-1.0",  # OrcaCyber harness
    "orcarouter/fusion",  # general fallback
)
VERIFIED_FALLBACK_MODELS = (
    {
        "id": "openai/gpt-5.5",
        "context_length": 400_000,
        "input_modalities": ["text", "image"],
        "reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "id": "anthropic/claude-opus-4.8",
        "context_length": 200_000,
        "input_modalities": ["text", "image"],
        "reasoning_efforts": None,
    },
    {
        "id": "google/gemini-3.5-flash",
        "context_length": 1_000_000,
        "input_modalities": ["text", "image"],
        "reasoning_efforts": None,
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "context_length": 128_000,
        "input_modalities": ["text"],
        "reasoning_efforts": None,
    },
    {
        "id": "orcarouter/auto",
        "context_length": None,
        "input_modalities": ["text"],
        "reasoning_efforts": None,
    },
)

# Endpoint types this adapter can speak (the OpenAI chat-completions wire format).
CHAT_ENDPOINT_TYPES = ("openai", "openai-response")


def chat_completions_url() -> str:
    return f"{api_base_url()}/chat/completions"


def _resolve(provider) -> dict:
    """The provider as a dict, with the credential and model filled from the environment."""
    if provider is None:
        integ: dict = {
            "api_key": getenv("LLM_API_KEY", getenv("ORCAROUTER_API_KEY", "")),
            "model": getenv("LLM_MODEL", DEFAULT_MODEL),
        }
        source = "api_key"
    else:
        if hasattr(provider, "model_dump"):
            source = provider.name.value
            provider = provider.model_dump()
        else:
            source = provider.get("name", "orcarouter")
            if hasattr(source, "value"):
                source = source.value
        integ = dict(provider.get("integration") or {})
        integ.setdefault("model", DEFAULT_MODEL)
    # "orcarouter-oauth": no key in the Provider, the PKCE adapter's stored key is used instead.
    if not integ.get("api_key") and source == "orcarouter-oauth":
        credential = OrcaPkceConnector().load()
        integ["api_key"] = credential.api_key if credential else ""
    return {"integration": integ, "source": source}


def credential_for(provider) -> OrcaCredential:
    """The one credential shape downstream uses, from whichever adapter the provider name names."""
    resolved = _resolve(provider)
    integ, source = resolved["integration"], resolved["source"]
    return OrcaCredential(
        api_key=integ.get("api_key", ""),
        source="pkce" if source == "orcarouter-oauth" else "api_key",
    )


def _is_chat_model(item: dict) -> bool:
    """Text chat/agent models only: an OpenAI-format chat endpoint and no non-text output."""
    if item.get("endpoint_types"):
        if not set(item["endpoint_types"]) & set(CHAT_ENDPOINT_TYPES):
            return False
    if (item.get("output_modalities") or ["text"]) != ["text"]:
        return False
    return "text" in (item.get("input_modalities") or ["text"])


def fetch_models(credential: OrcaCredential, *, capability: str = "chat") -> list[dict | None]:
    """Fetch the live catalog from ``GET {api_base}/models`` and return compatible models.

    Raises on transport/auth/shape errors so the caller can fall back to the verified seed.
    """
    params = {"capability": capability} if capability else None
    resp = requests.get(
        f"{api_base_url()}/models",
        headers={"Authorization": f"Bearer {credential.api_key}"},
        params=params,
        timeout=MODELS_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OrcaRouter model catalog returned {resp.status_code}")
    payload = resp.json()
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError("OrcaRouter model catalog is not a list")
    models = []
    for item in items[:MAX_CATALOG_ITEMS]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        normalised = _normalise_model(item)
        if capability == "chat" and not _is_chat_model(normalised):
            continue
        models.append(normalised)
    # A model ID from here on without a verified shape is never offered: fail closed.
    return rank_models(models)


def rank_models(models: list[dict]) -> list[dict]:
    """Recommended models first (in RECOMMENDED_MODELS order), then the rest as the catalog gave."""
    order = {m: i for i, m in enumerate(RECOMMENDED_MODELS)}
    return sorted(models, key=lambda m: (order.get(m["id"], len(order)),))


def _normalise_model(item: dict) -> dict:
    """The subset of a catalog item this adapter uses. Missing modality data means text only."""
    arch = item.get("architecture") or {}
    in_mods = arch.get("input_modalities")
    out_mods = arch.get("output_modalities")
    endpoint_types = item.get("supported_endpoint_types") or []
    if not isinstance(endpoint_types, list):
        endpoint_types = []
    reasoning = item.get("reasoning")
    efforts = reasoning.get("efforts") if isinstance(reasoning, dict) else None
    return {
        "id": item["id"],
        "context_length": item.get("context_length")
        if isinstance(item.get("context_length"), int)
        else None,
        "input_modalities": in_mods if isinstance(in_mods, list) else ["text"],
        "output_modalities": out_mods if isinstance(out_mods, list) else ["text"],
        "endpoint_types": endpoint_types,
        "reasoning_efforts": efforts if isinstance(efforts, list) else None,
    }


def list_models(
    credential: OrcaCredential, *, capability: str = "chat", live: bool = True
) -> tuple[list[dict], str]:
    """(models, source): live catalog when it succeeds, else the verified fallback list."""
    if live:
        try:
            return fetch_models(credential, capability=capability), "live"
        except (requests.RequestException, ValueError, RuntimeError, AttributeError):
            pass
    return [dict(m) for m in VERIFIED_FALLBACK_MODELS], "fallback"


class LLMStreamer:
    def __init__(self, provider=None):
        resolved = _resolve(provider)
        integ = resolved["integration"]
        self.model = integ["model"]
        self.reasoning_effort = integ.get("reasoning_effort")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {integ.get('api_key', '')}",
                "Content-Type": "application/json",
            }
        )

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        params = _sampling_params(self.model, max_tokens, temperature, self.reasoning_effort)
        body = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_p},
                {"role": "user", "content": user_p},
            ],
        }
        while True:
            resp = self._session.post(
                chat_completions_url(),
                json={**body, **params},
                stream=True,
                timeout=LLM_PING_TIMEOUT,
            )
            if resp.status_code == 200:
                return _SseChunks(resp)
            error_text = _redact(resp.text, self._session.headers["Authorization"])
            drop = (
                _note_rejected(self.model, error_text, sent=params)
                if resp.status_code == 400
                else set()
            )
            resp.close()
            if not drop:
                raise RuntimeError(
                    f"502/Error pinging OrcaRouter LLM - {resp.status_code}/{error_text}"
                )
            for name in drop:
                params.pop(name)


class LLMPinger:
    def __init__(self, provider=None):
        self._provider = _resolve(provider)

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        integ = self._provider["integration"]
        do_retry_counter = 0
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        while do_retry_counter <= MAX_RETRY_COUNTER:
            params = _sampling_params(
                integ["model"], max_tokens, temperature, integ.get("reasoning_effort")
            )
            authz = f"Bearer {integ.get('api_key', '')}"
            resp = requests.post(
                chat_completions_url(),
                headers={"Content-Type": "application/json", "Authorization": authz},
                json={
                    "model": integ["model"],
                    "messages": [
                        {"role": "system", "content": system_p},
                        {"role": "user", "content": user_p},
                    ],
                    **params,
                },
                timeout=LLM_PING_TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                if "choices" not in result or not result["choices"]:
                    raise Exception("502/Invalid LLM response format.")
                content = result["choices"][0]["message"].get("content")
                if content is None:
                    refusal = result["choices"][0]["message"].get("refusal", "")
                    return refusal or "[No content in LLM response]"
                return content
            elif resp.status_code == 429:
                do_retry_counter += 1
                if do_retry_counter <= MAX_RETRY_COUNTER:
                    time.sleep(do_retry_counter)
                    continue
                raise Exception("502/Rate limit error.")
            elif resp.status_code == 400:
                if _note_rejected(integ["model"], resp.text, sent=params):
                    continue
                raise Exception(f"502/Inappropriate content ({resp.text}).")
            elif resp.status_code == 401:
                # A durable key was revoked: terminal, never retried, never "refreshed".
                raise OrcaReauthRequired(
                    "OrcaRouter rejected the stored key (401). Re-run "
                    "`humanbound_firewall llm orcarouter connect` to authorize again."
                )
            else:
                raise Exception(
                    f"502/Error pinging OrcaRouter LLM - {resp.status_code}/{_redact(resp.text, authz)}"
                )


class OrcaReauthRequired(RuntimeError):
    """The OrcaRouter key was revoked or is invalid: the user must authorize again."""


def _redact(text: str, authorization: str) -> str:
    """Error bodies never echo the key back."""
    key = authorization.replace("Bearer ", "")
    return text.replace(key, "sk-orca-***") if key else text


class _SseChunks:
    """Server-sent events. Yields typed dict attributes the judge's `_extract_token` can read."""

    def __init__(self, resp, key=""):
        self._resp = resp
        self.event = ""
        self._key = key or ""
        try:
            self._key = resp.request.headers["Authorization"].replace("Bearer ", "")
        except Exception:
            pass

    def __iter__(self):
        try:
            for line in self._resp.iter_lines(decode_unicode=True):
                if line.startswith("event: "):
                    self.event = line[len("event: ") :].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    return
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                if self.event == "error":
                    raise RuntimeError("502/OrcaRouter judge stream error")
                if isinstance(payload, dict) and payload.get("error"):
                    message = str((payload.get("error") or {}).get("message", "")).replace(
                        self._key, "sk-orca-***"
                    )
                    raise RuntimeError(f"502/OrcaRouter judge error: {message[:200]}")
                yield _Attr(payload)
        finally:
            self._resp.close()


def _Attr(obj):
    """An attribute-view over parsed JSON, so `chunk.choices[0].delta.content` works."""
    if isinstance(obj, dict):
        return _AttrDict(obj)
    if isinstance(obj, list):
        return [_Attr(v) for v in obj]
    return obj


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return _Attr(self[name])
        except KeyError:
            return None
