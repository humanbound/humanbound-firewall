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


def test_openai_streamer_happy_path(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from humanbound_firewall.llm.openai import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.OPENAI, model="gpt-4o-mini"))
    assert streamer.model == "gpt-4o-mini"
    fake_openai.OpenAI.assert_called_once_with(api_key="sk-test")


def test_openai_streamer_ping_forwards_args(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_client = MagicMock()
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from humanbound_firewall.llm.openai import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.OPENAI))
    streamer.ping("system text", "user text", max_tokens=100, temperature=0.2)

    call = fake_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert call.kwargs["max_tokens"] == 100
    assert call.kwargs["temperature"] == 0.2
    assert call.kwargs["stream"] is True


def test_openai_streamer_missing_dep_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    from humanbound_firewall.llm.openai import LLMStreamer

    with pytest.raises(ImportError, match=r"humanbound-firewall\[openai\]"):
        LLMStreamer(_provider(ProviderName.OPENAI))


# ────────────────────────────────────────────────────────────────
# Anthropic Claude
# ────────────────────────────────────────────────────────────────


def test_claude_streamer_happy_path(monkeypatch):
    fake = types.ModuleType("anthropic")
    fake.Anthropic = MagicMock()
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from humanbound_firewall.llm.claude import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.CLAUDE, model="claude-x"))
    assert streamer.model == "claude-x"
    fake.Anthropic.assert_called_once_with(api_key="sk-test")


def test_claude_streamer_ping_forwards_args(monkeypatch):
    fake = types.ModuleType("anthropic")
    fake_client = MagicMock()
    fake.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from humanbound_firewall.llm.claude import LLMStreamer

    streamer = LLMStreamer(_provider(ProviderName.CLAUDE))
    streamer.ping("s", "u", max_tokens=50, temperature=0.0)

    call = fake_client.messages.create.call_args
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
