# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""LLM provider abstraction — supports OpenAI, Azure OpenAI, Anthropic Claude, and Gemini.

Two interfaces:
- LLMPinger: non-streaming completion with retry (for training data generation)
- LLMStreamer: streaming completion (for Tier 2 judge evaluation)
"""

from .models import Provider, ProviderIntegration, ProviderName


def get_llm_pinger(provider):
    """Return an LLMPinger for the given provider.

    Args:
        provider: Provider instance or dict with "name" and "integration" keys.
    """
    name = provider.name if isinstance(provider, Provider) else provider["name"]

    if name == ProviderName.AZURE_OPENAI or name == "azureopenai":
        from humanbound_firewall.llm.azureopenai import LLMPinger
    elif name == ProviderName.OPENAI or name == "openai":
        from humanbound_firewall.llm.openai import LLMPinger
    elif name == ProviderName.CLAUDE or name == "claude":
        from humanbound_firewall.llm.claude import LLMPinger
    elif name == ProviderName.GEMINI or name == "gemini":
        from humanbound_firewall.llm.gemini import LLMPinger
    else:
        raise ValueError(f"Unsupported LLM provider: {name}")

    return LLMPinger(provider)


def get_llm_streamer(provider):
    """Return an LLMStreamer for the given provider."""
    name = provider.name if isinstance(provider, Provider) else provider["name"]

    if name == ProviderName.AZURE_OPENAI or name == "azureopenai":
        from humanbound_firewall.llm.azureopenai import LLMStreamer
    elif name == ProviderName.OPENAI or name == "openai":
        from humanbound_firewall.llm.openai import LLMStreamer
    elif name == ProviderName.CLAUDE or name == "claude":
        from humanbound_firewall.llm.claude import LLMStreamer
    elif name == ProviderName.GEMINI or name == "gemini":
        from humanbound_firewall.llm.gemini import LLMStreamer
    else:
        raise ValueError(f"Unsupported LLM provider: {name}")

    return LLMStreamer(provider)


__all__ = [
    "Provider",
    "ProviderIntegration",
    "ProviderName",
    "get_llm_pinger",
    "get_llm_streamer",
]
