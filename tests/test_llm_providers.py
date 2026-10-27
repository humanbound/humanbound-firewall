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
            reasoning_effort=kw.get("reasoning_effort"),
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


class _ReasoningEffortRejected(Exception):
    """What the OpenAI SDK raises when a model is sent a `reasoning_effort` it does not support."""

    status_code = 400

    def __init__(self):
        super().__init__(
            "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'reasoning_effort' "
            'is not supported with this model.", '
            "'param': 'reasoning_effort', 'code': 'unsupported_parameter'}}"
        )


class _InvalidReasoningEffort(Exception):
    """What the current openai SDK raises for an invalid value: the message does not name the
    param, which is only in the structured `param` attribute."""

    status_code = 400
    param = "reasoning_effort"

    def __init__(self):
        super().__init__(
            "Error code: 400 - Invalid value: 'hihg'. "
            "Supported values are: 'minimal', 'low', 'medium', and 'high'."
        )


@pytest.fixture(autouse=True)
def _forget_rejections():
    import humanbound_firewall.llm.openai as mod

    mod._models_without_temperature.clear()
    mod._models_without_reasoning_effort.clear()
    yield
    mod._models_without_temperature.clear()
    mod._models_without_reasoning_effort.clear()


def _streamer(monkeypatch, model, reject_temperature=False, reject_reasoning_effort=False, **kw):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if reject_temperature and "temperature" in kwargs:
            raise _TemperatureRejected()
        if reject_reasoning_effort and "reasoning_effort" in kwargs:
            error = reject_reasoning_effort
            raise (error if isinstance(error, type) else _ReasoningEffortRejected)()
        return iter([])

    fake_client.chat.completions.create = create
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.openai import LLMStreamer

    return LLMStreamer(_provider(ProviderName.OPENAI, model=model, **kw)), calls


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


def test_openai_streamer_sends_reasoning_effort_only_when_configured(monkeypatch):
    with_effort, calls_with = _streamer(monkeypatch, "gpt-5-mini", reasoning_effort="minimal")
    with_effort.ping("s", "u")
    assert calls_with[0]["reasoning_effort"] == "minimal"

    without_effort, calls_without = _streamer(monkeypatch, "gpt-5-mini")
    without_effort.ping("s", "u")
    assert "reasoning_effort" not in calls_without[0]  # None means: let the model's default stand


@pytest.mark.parametrize(
    "error", [_ReasoningEffortRejected, _InvalidReasoningEffort], ids=["unsupported", "invalid"]
)
def test_openai_streamer_drops_a_rejected_reasoning_effort_and_remembers_it(
    monkeypatch, caplog, error
):
    streamer, calls = _streamer(
        monkeypatch, "gpt-4o", reject_reasoning_effort=error, reasoning_effort="minimal"
    )
    streamer.ping("s", "u")
    assert len(calls) == 2
    assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[1]
    # A typo'd effort draws the same 400 as an unsupported one, so the drop must not be silent.
    assert "gpt-4o" in caplog.text and "reasoning_effort" in caplog.text
    caplog.clear()
    streamer.ping("s", "u")
    assert len(calls) == 3  # one wasted request in total, not one per call
    assert "reasoning_effort" not in calls[2]
    assert not caplog.text  # warned once per model, not once per call


def _pinger_calls(
    monkeypatch, model, reject_temperature=False, reject_reasoning_effort=False, **kw
):
    import humanbound_firewall.llm.openai as mod

    bodies = []

    def post(url, headers, json, timeout):
        bodies.append(json)
        if reject_temperature and "temperature" in json:
            return MagicMock(status_code=400, text=str(_TemperatureRejected()))
        if reject_reasoning_effort and "reasoning_effort" in json:
            return MagicMock(status_code=400, text=str(_ReasoningEffortRejected()))
        return MagicMock(
            status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr(mod.requests, "post", post)
    return mod.LLMPinger(_provider(ProviderName.OPENAI, model=model, **kw)), bodies


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


def test_openai_pinger_sends_and_retries_away_a_rejected_reasoning_effort(monkeypatch):
    pinger, bodies = _pinger_calls(monkeypatch, "gpt-5-mini", reasoning_effort="low")
    pinger.ping("s", "u")
    assert bodies[0]["reasoning_effort"] == "low"

    pinger, bodies = _pinger_calls(
        monkeypatch, "gpt-4o", reject_reasoning_effort=True, reasoning_effort="low"
    )
    assert pinger.ping("s", "u") == "ok"
    assert len(bodies) == 2 and "reasoning_effort" not in bodies[1]


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

    from humanbound_firewall.llm.claude import THINKING_TOKEN_HEADROOM

    streamer.ping("s", "u", max_tokens=50, temperature=0.0)

    call = fake_client.messages.create.call_args
    assert call.kwargs["model"] == "claude-x"
    assert call.kwargs["system"] == "s"
    assert call.kwargs["messages"] == [{"role": "user", "content": "u"}]
    # the visible budget, plus headroom for a model that thinks before it answers
    assert call.kwargs["max_tokens"] == 50 + THINKING_TOKEN_HEADROOM
    assert call.kwargs["temperature"] == 0.0  # sent first: models that accept it honour it
    assert call.kwargs["stream"] is True


@pytest.fixture
def _forget_claude_temperature_rejections():
    import humanbound_firewall.llm.claude as mod

    mod._models_without_temperature.clear()
    yield
    mod._models_without_temperature.clear()


class _ClaudeTemperatureRejected(Exception):
    """What the Anthropic SDK raises when a model is sent a non-default temperature."""

    status_code = 400

    def __init__(self):
        super().__init__(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'temperature: only the default value is supported for this model'}}"
        )


def _claude(monkeypatch, create):
    fake = types.ModuleType("anthropic")
    client = MagicMock()
    client.messages.create = create
    fake.Anthropic = MagicMock(return_value=client)
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


@pytest.mark.usefixtures("_forget_claude_temperature_rejections")
def test_claude_streamer_retries_without_temperature_and_remembers_the_rejection(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise _ClaudeTemperatureRejected()
        return iter([])

    _claude(monkeypatch, create)
    from humanbound_firewall.llm.claude import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.CLAUDE, model="claude-sonnet-5-5"))
    streamer.ping("s", "u", max_tokens=100, temperature=0.0)
    assert len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    streamer.ping("s", "u")
    assert len(calls) == 3 and "temperature" not in calls[2]  # one wasted request per model


@pytest.mark.usefixtures("_forget_claude_temperature_rejections")
def test_claude_streamer_does_not_swallow_other_400s(monkeypatch):
    class Other400(Exception):
        status_code = 400

    create = MagicMock(side_effect=Other400("prompt is too long"))
    _claude(monkeypatch, create)
    from humanbound_firewall.llm.claude import LLMStreamer

    with pytest.raises(Other400):
        LLMStreamer(_provider(ProviderName.CLAUDE, model="claude-x")).ping("s", "u")
    assert create.call_count == 1


@pytest.mark.usefixtures("_forget_claude_temperature_rejections")
def test_claude_pinger_reads_the_text_after_a_thinking_block(monkeypatch):
    reply = MagicMock(
        content=[
            types.SimpleNamespace(type="thinking", thinking="let me think", signature="s"),
            types.SimpleNamespace(type="text", text="P The request is in scope."),
        ]
    )
    _claude(monkeypatch, MagicMock(return_value=reply))
    from humanbound_firewall.llm.claude import LLMPinger

    assert LLMPinger(_provider(ProviderName.CLAUDE)).ping("s", "u") == "P The request is in scope."


def test_the_default_claude_model_ids_are_undated_aliases():
    """A dated snapshot ID must be one Anthropic published; the default uses the plain alias."""
    from humanbound_firewall.llm.claude import DEFAULT_MODEL

    assert DEFAULT_MODEL == "claude-sonnet-4-6"


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


def _azure_streamer(
    monkeypatch, model, reject_temperature=False, reject_reasoning_effort=None, **kw
):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if reject_temperature and "temperature" in kwargs:
            raise _TemperatureRejected()
        if reject_reasoning_effort and "reasoning_effort" in kwargs:
            raise reject_reasoning_effort()
        return iter([])

    fake_client.chat.completions.create = create
    fake_openai.AzureOpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    from humanbound_firewall.llm.azureopenai import LLMStreamer

    return LLMStreamer(
        _provider(
            ProviderName.AZURE_OPENAI, model=model, endpoint="https://x.openai.azure.com", **kw
        )
    ), calls


def test_azure_streamer_drops_an_invalid_reasoning_effort_named_only_by_param(monkeypatch):
    streamer, calls = _azure_streamer(
        monkeypatch, "dep", reject_reasoning_effort=_InvalidReasoningEffort, reasoning_effort="hihg"
    )
    streamer.ping("s", "u")
    assert len(calls) == 2
    assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[1]


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


@pytest.mark.usefixtures("_forget_claude_temperature_rejections")
def test_claude_never_sends_temperature_to_an_sdk_that_removed_it(monkeypatch):
    """The 1.x Anthropic SDK dropped `temperature` from messages.create: passing it is a
    TypeError before any request, for every model."""
    calls = []

    def create(*, model, max_tokens, messages, system=None, stream=False, timeout=None):
        calls.append({"model": model, "max_tokens": max_tokens, "stream": stream})
        return iter([])

    _claude(monkeypatch, create)
    from humanbound_firewall.llm.claude import LLMPinger, LLMStreamer

    LLMStreamer(_provider(ProviderName.CLAUDE, model="claude-sonnet-5-5")).ping("s", "u")
    assert len(calls) == 1 and calls[0]["stream"] is True  # one call, no TypeError

    def create_sync(*, model, max_tokens, messages, system=None, timeout=None):
        return MagicMock(content=[types.SimpleNamespace(type="text", text="P ok")])

    _claude(monkeypatch, create_sync)
    assert LLMPinger(_provider(ProviderName.CLAUDE)).ping("s", "u") == "P ok"


@pytest.mark.usefixtures("_forget_claude_temperature_rejections")
def test_claude_pinger_sends_an_explicit_timeout(monkeypatch):
    """With the thinking headroom, the pinger's non-streaming max_tokens exceeds the SDK's
    non-streaming cap for some models (8192 for Opus 4 / 4.1); the SDK refuses such a request
    unless it carries an explicit timeout. The pinger sends the same one as the streamer."""
    create = MagicMock(
        return_value=MagicMock(content=[types.SimpleNamespace(type="text", text="P ok")])
    )
    _claude(monkeypatch, create)
    from humanbound_firewall.llm.claude import LLM_PING_TIMEOUT, LLMPinger

    assert (
        LLMPinger(_provider(ProviderName.CLAUDE, model="claude-opus-4-1")).ping("s", "u") == "P ok"
    )
    assert create.call_args.kwargs["timeout"] == LLM_PING_TIMEOUT
    assert create.call_args.kwargs["max_tokens"] > 8192  # the case the SDK would refuse untimed
