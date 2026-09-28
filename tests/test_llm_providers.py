# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Smoke tests for the LLM provider adapters.

These don't exercise real APIs — they mock the third-party SDK entry
points (openai.OpenAI, anthropic.Anthropic, google.generativeai,
AzureOpenAI) and verify:

- The adapter's LLMStreamer / LLMPinger constructs cleanly from a Provider
- ``ping()`` forwards arguments to the right SDK method
- Missing optional deps produce the actionable ImportError we promise

The rest of each provider's behaviour (retries, rate-limit handling,
streaming iteration) is covered where it affects routing in
``tests/test_firewall.py``.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from humanbound_firewall.llm.models import (
    Provider,
    ProviderIntegration,
    ProviderName,
)


def _provider(name: ProviderName, **kw) -> Provider:
    return Provider(
        name=name,
        integration=ProviderIntegration(
            api_key="sk-test",
            model=kw.get("model", "test-model"),
            endpoint=kw.get("endpoint"),
            api_version=kw.get("api_version"),
        ),
    )


# ────────────────────────────────────────────────────────────────
# OpenAI
# ────────────────────────────────────────────────────────────────


ANY_MODEL = ["gpt-4o-mini", "gpt-4.1", "gpt-5-nano", "o3-mini", "gpt-6", "gpt-7-turbo"]


class _TemperatureRejected(Exception):
    """What the OpenAI SDK raises when a reasoning model is sent a non-default temperature."""

    status_code = 400

    def __init__(self):
        super().__init__(
            "Error code: 400 - {'error': {'message': \"Unsupported value: 'temperature' does not "
            'support 0.0 with this model. Only the default (1) value is supported.", '
            "'param': 'temperature', 'code': 'unsupported_value'}}"
        )


@pytest.fixture(autouse=True)
def _forget_temperature_rejections():
    import humanbound_firewall.llm.openai as mod

    mod._models_without_temperature.clear()
    yield
    mod._models_without_temperature.clear()


def _streamer(monkeypatch, model, reject_temperature=False):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if reject_temperature and "temperature" in kwargs:
            raise _TemperatureRejected()
        return iter([])

    fake_client.chat.completions.create = create
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.openai import LLMStreamer

    return LLMStreamer(_provider(ProviderName.OPENAI, model=model)), calls


@pytest.mark.parametrize("model", ["test-model", *ANY_MODEL])
def test_openai_streamer_forwards_args_with_max_completion_tokens_for_every_model(
    monkeypatch, model
):
    """No model list: `max_completion_tokens` is accepted by every chat model (the deprecated
    `max_tokens` is what reasoning models reject), so a gpt-6 works the day it appears."""
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from humanbound_firewall.llm.openai import REASONING_TOKEN_HEADROOM, LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.OPENAI, model=model))
    assert streamer.model == model
    fake_openai.OpenAI.assert_called_once_with(api_key="sk-test")

    streamer.ping("system text", "user text", max_tokens=100, temperature=0.2)

    assert fake_client.chat.completions.create.call_count == 1
    call = fake_client.chat.completions.create.call_args
    assert call.kwargs["model"] == model
    assert call.kwargs["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert call.kwargs["max_completion_tokens"] == 100 + REASONING_TOKEN_HEADROOM
    assert "max_tokens" not in call.kwargs
    assert call.kwargs["temperature"] == 0.2  # sent first: classic models honour it
    assert call.kwargs["stream"] is True


def test_openai_streamer_retries_without_temperature_and_remembers_the_rejection(monkeypatch):
    streamer, calls = _streamer(monkeypatch, "gpt-6", reject_temperature=True)
    streamer.ping("s", "u", max_tokens=100, temperature=0.0)
    assert len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    streamer.ping("s", "u")
    assert len(calls) == 3  # one wasted request in total, not one per call
    assert "temperature" not in calls[2]


def test_openai_streamer_does_not_swallow_other_400s(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()

    class Other400(Exception):
        status_code = 400

    fake_client.chat.completions.create = MagicMock(side_effect=Other400("context length exceeded"))
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.openai import LLMStreamer

    with pytest.raises(Other400):
        LLMStreamer(_provider(ProviderName.OPENAI, model="gpt-6")).ping("s", "u")
    assert fake_client.chat.completions.create.call_count == 1


def _pinger_calls(monkeypatch, model, reject_temperature=False):
    import humanbound_firewall.llm.openai as mod

    bodies = []

    def post(url, headers, json, timeout):
        bodies.append(json)
        if reject_temperature and "temperature" in json:
            return MagicMock(status_code=400, text=str(_TemperatureRejected()))
        return MagicMock(
            status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr(mod.requests, "post", post)
    return mod.LLMPinger(_provider(ProviderName.OPENAI, model=model)), bodies


@pytest.mark.parametrize("model", ANY_MODEL)
def test_openai_pinger_uses_max_completion_tokens_and_retries_without_a_rejected_temperature(
    monkeypatch, model
):
    from humanbound_firewall.llm.openai import REASONING_TOKEN_HEADROOM

    pinger, bodies = _pinger_calls(monkeypatch, model)
    pinger.ping("s", "u", max_tokens=100, temperature=0.0)
    (body,) = bodies
    assert (
        body["max_completion_tokens"] == 100 + REASONING_TOKEN_HEADROOM and "max_tokens" not in body
    )

    pinger, bodies = _pinger_calls(monkeypatch, model, reject_temperature=True)
    assert pinger.ping("s", "u") == "ok"
    assert len(bodies) == 2 and "temperature" not in bodies[1]


def test_openai_streamer_missing_dep_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    from humanbound_firewall.llm.openai import LLMStreamer

    with pytest.raises(ImportError, match=r"humanbound-firewall\[openai\]"):
        LLMStreamer(_provider(ProviderName.OPENAI))


# ────────────────────────────────────────────────────────────────
# Anthropic Claude
# ────────────────────────────────────────────────────────────────


def test_claude_streamer_constructs_and_ping_forwards_args(monkeypatch):
    fake = types.ModuleType("anthropic")
    fake_client = MagicMock()
    fake.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from humanbound_firewall.llm.claude import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.CLAUDE, model="claude-x"))
    assert streamer.model == "claude-x"
    fake.Anthropic.assert_called_once_with(api_key="sk-test")

    streamer.ping("s", "u", max_tokens=50, temperature=0.0)

    call = fake_client.messages.create.call_args
    assert call.kwargs["model"] == "claude-x"
    assert call.kwargs["system"] == "s"
    assert call.kwargs["messages"] == [{"role": "user", "content": "u"}]
    assert call.kwargs["max_tokens"] == 50
    assert call.kwargs["stream"] is True


def test_claude_missing_dep_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from humanbound_firewall.llm.claude import LLMStreamer

    with pytest.raises(ImportError, match=r"humanbound-firewall\[anthropic\]"):
        LLMStreamer(_provider(ProviderName.CLAUDE))


# ────────────────────────────────────────────────────────────────
# Google Gemini
# ────────────────────────────────────────────────────────────────


def test_gemini_streamer_happy_path(monkeypatch):
    fake = types.ModuleType("google.generativeai")
    fake.configure = MagicMock()
    fake.GenerativeModel = MagicMock()
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)

    from humanbound_firewall.llm.gemini import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.GEMINI, model="gemini-pro"))
    fake.configure.assert_called_once_with(api_key="sk-test")
    fake.GenerativeModel.assert_called_once_with("gemini-pro")


def test_gemini_missing_dep_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.generativeai", None)
    from humanbound_firewall.llm.gemini import LLMStreamer

    with pytest.raises(ImportError, match=r"humanbound-firewall\[gemini\]"):
        LLMStreamer(_provider(ProviderName.GEMINI))


# ────────────────────────────────────────────────────────────────
# Azure OpenAI
# ────────────────────────────────────────────────────────────────


def test_azure_openai_streamer_happy_path(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from humanbound_firewall.llm.azureopenai import LLMStreamer

    streamer = LLMStreamer(
        _provider(
            ProviderName.AZURE_OPENAI,
            model="gpt-4o-mini",
            endpoint="https://example.openai.azure.com",
            api_version="2024-06-01",
        )
    )
    assert streamer.model == "gpt-4o-mini"
    call = fake_openai.AzureOpenAI.call_args
    assert call.kwargs["api_key"] == "sk-test"
    assert call.kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert call.kwargs["api_version"] == "2024-06-01"


def test_azure_openai_streamer_defaults_the_api_version_when_the_provider_leaves_it_out(
    monkeypatch,
):
    """A Provider without api_version dumps it as None; the default must still apply."""
    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from humanbound_firewall.llm.azureopenai import LLMStreamer

    LLMStreamer(_provider(ProviderName.AZURE_OPENAI, endpoint="https://example.openai.azure.com"))
    assert fake_openai.AzureOpenAI.call_args.kwargs["api_version"] == "2024-06-01"


def _azure_streamer(monkeypatch, model, reject_temperature=False):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if reject_temperature and "temperature" in kwargs:
            raise _TemperatureRejected()
        return iter([])

    fake_client.chat.completions.create = create
    fake_openai.AzureOpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.azureopenai import LLMStreamer

    return LLMStreamer(
        _provider(ProviderName.AZURE_OPENAI, model=model, endpoint="https://x.openai.azure.com")
    ), calls


@pytest.mark.parametrize("model", ANY_MODEL)
def test_azure_streamer_uses_max_completion_tokens_for_every_deployment(monkeypatch, model):
    """Azure deployment names are arbitrary, so a model-name list could never work here anyway."""
    from humanbound_firewall.llm.openai import REASONING_TOKEN_HEADROOM

    streamer, calls = _azure_streamer(monkeypatch, model)
    streamer.ping("s", "u", max_tokens=100, temperature=0.0)
    (kwargs,) = calls
    assert kwargs["max_completion_tokens"] == 100 + REASONING_TOKEN_HEADROOM
    assert "max_tokens" not in kwargs and kwargs["temperature"] == 0.0


def test_azure_streamer_retries_without_temperature_when_the_deployment_rejects_it(monkeypatch):
    streamer, calls = _azure_streamer(monkeypatch, "my-o3-deployment", reject_temperature=True)
    streamer.ping("s", "u")
    streamer.ping("s", "u")
    assert len(calls) == 3  # rejected once, remembered
    assert (
        "temperature" in calls[0]
        and "temperature" not in calls[1]
        and "temperature" not in calls[2]
    )


def test_azure_streamer_does_not_swallow_other_400s(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()

    class Other400(Exception):
        status_code = 400

    fake_client.chat.completions.create = MagicMock(side_effect=Other400("content filter"))
    fake_openai.AzureOpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.azureopenai import LLMStreamer

    with pytest.raises(Other400):
        LLMStreamer(_provider(ProviderName.AZURE_OPENAI, model="d", endpoint="https://x")).ping(
            "s", "u"
        )


def _azure_pinger_calls(monkeypatch, model, reject_temperature=False):
    import humanbound_firewall.llm.azureopenai as mod

    bodies = []

    def post(url, headers, json, timeout):
        bodies.append(json)
        if reject_temperature and "temperature" in json:
            return MagicMock(status_code=400, text=str(_TemperatureRejected()))
        return MagicMock(
            status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr(mod.requests, "post", post)
    return mod.LLMPinger(
        _provider(ProviderName.AZURE_OPENAI, model=model, endpoint="https://x")
    ), bodies


def test_azure_pinger_uses_max_completion_tokens_and_retries_without_temperature(monkeypatch):
    from humanbound_firewall.llm.openai import REASONING_TOKEN_HEADROOM

    pinger, bodies = _azure_pinger_calls(monkeypatch, "my-o3-deployment", reject_temperature=True)
    assert pinger.ping("s", "u", max_tokens=100) == "ok"
    assert len(bodies) == 2
    assert (
        bodies[0]["max_completion_tokens"] == 100 + REASONING_TOKEN_HEADROOM
        and "max_tokens" not in bodies[0]
    )
    assert "temperature" in bodies[0] and "temperature" not in bodies[1]


def test_azure_openai_missing_dep_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    from humanbound_firewall.llm.azureopenai import LLMStreamer

    with pytest.raises(ImportError, match=r"humanbound-firewall\[openai\]"):
        LLMStreamer(_provider(ProviderName.AZURE_OPENAI))


# ────────────────────────────────────────────────────────────────
# Factory routing
# ────────────────────────────────────────────────────────────────


def test_get_llm_streamer_routes_by_provider_name(monkeypatch):
    # Stub all four SDKs so factory can instantiate any provider
    for sdk_name in ("openai", "anthropic"):
        mod = types.ModuleType(sdk_name)
        mod.OpenAI = MagicMock()
        mod.AzureOpenAI = MagicMock()
        mod.Anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, sdk_name, mod)
    gem = types.ModuleType("google.generativeai")
    gem.configure = MagicMock()
    gem.GenerativeModel = MagicMock()
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = gem
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", gem)

    from humanbound_firewall.llm import get_llm_streamer
    from humanbound_firewall.llm.azureopenai import LLMStreamer as AzureStreamer
    from humanbound_firewall.llm.claude import LLMStreamer as ClaudeStreamer
    from humanbound_firewall.llm.gemini import LLMStreamer as GeminiStreamer
    from humanbound_firewall.llm.openai import LLMStreamer as OpenAIStreamer

    assert isinstance(get_llm_streamer(_provider(ProviderName.OPENAI)), OpenAIStreamer)
    assert isinstance(get_llm_streamer(_provider(ProviderName.CLAUDE)), ClaudeStreamer)
    assert isinstance(get_llm_streamer(_provider(ProviderName.GEMINI)), GeminiStreamer)
    assert isinstance(get_llm_streamer(_provider(ProviderName.AZURE_OPENAI)), AzureStreamer)


def test_get_llm_streamer_rejects_unknown_provider():
    from humanbound_firewall.llm import get_llm_streamer

    with pytest.raises(ValueError, match="Unsupported"):
        get_llm_streamer({"name": "unsupported", "integration": {}})
