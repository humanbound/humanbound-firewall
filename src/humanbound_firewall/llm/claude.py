# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Anthropic Claude provider."""

import time
from os import getenv

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0


class LLMStreamer:
    def __init__(self, provider=None):
        provider = _resolve(provider)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the [anthropic] extra. "
                "Install with: pip install humanbound-firewall[anthropic]"
            ) from e
        self.client = anthropic.Anthropic(api_key=provider["integration"]["api_key"])
        self.model = provider["integration"]["model"]

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        return self.client.messages.create(
            model=self.model,
            system=system_p,
            messages=[{"role": "user", "content": user_p}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=LLM_PING_TIMEOUT,
            stream=True,
        )


class LLMPinger:
    def __init__(self, provider=None):
        self._provider = _resolve(provider)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the [anthropic] extra. "
                "Install with: pip install humanbound-firewall[anthropic]"
            ) from e
        self.client = anthropic.Anthropic(api_key=self._provider["integration"]["api_key"])

    def __do_completion_api_call(self, system_p, user_p, max_tokens, temperature):
        return self.client.messages.create(
            model=self._provider["integration"]["model"],
            system=system_p,
            messages=[{"role": "user", "content": user_p}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the [anthropic] extra. "
                "Install with: pip install humanbound-firewall[anthropic]"
            ) from e
        retry_counter = 0
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        while retry_counter <= MAX_RETRY_COUNTER:
            try:
                response = self.__do_completion_api_call(system_p, user_p, max_tokens, temperature)
                return response.content[0].text
            except anthropic.RateLimitError:
                retry_counter += 1
                if retry_counter <= MAX_RETRY_COUNTER:
                    time.sleep(retry_counter)
                    continue
                raise Exception("502/Rate limit error.")
            except anthropic.AuthenticationError:
                raise Exception("502/Authentication error. Check your API key.")
            except Exception as e:
                raise Exception(f"502/Error pinging LLM - {str(e)}")


def _resolve(provider):
    if provider is None:
        return {
            "integration": {
                "api_key": getenv("LLM_API_KEY", ""),
                "model": getenv("LLM_MODEL", "claude-sonnet-4-6-20250514"),
            }
        }
    if hasattr(provider, "model_dump"):
        return provider.model_dump()
    return provider
