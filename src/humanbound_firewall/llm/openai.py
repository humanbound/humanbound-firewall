# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""OpenAI provider."""

import time
from os import getenv

import requests

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0

OPENAI_CHAT_COMPLETION_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class LLMStreamer:
    def __init__(self, provider=None):
        provider = _resolve(provider)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the [openai] extra. "
                "Install with: pip install humanbound-firewall[openai]"
            ) from e
        self.__client = OpenAI(api_key=provider["integration"]["api_key"])
        self.model = provider["integration"]["model"]

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        return self.__client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_p},
                {"role": "user", "content": user_p},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=LLM_PING_TIMEOUT,
            stream=True,
        )


class LLMPinger:
    def __init__(self, provider=None):
        self._provider = _resolve(provider)

    def __do_completion_api_call(self, system_p, user_p, max_tokens, temperature):
        return requests.post(
            OPENAI_CHAT_COMPLETION_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._provider['integration']['api_key']}",
            },
            json={
                "model": self._provider["integration"]["model"],
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=LLM_PING_TIMEOUT,
        )

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        do_retry_counter = 0
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        while do_retry_counter <= MAX_RETRY_COUNTER:
            resp = self.__do_completion_api_call(system_p, user_p, max_tokens, temperature)
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
                raise Exception(f"502/Inappropriate content ({resp.text}).")
            else:
                raise Exception(f"502/Error pinging LLM - {resp.status_code}/{resp.text}")


def _resolve(provider):
    """Resolve provider to dict format."""
    if provider is None:
        return {
            "integration": {
                "api_key": getenv("LLM_API_KEY", ""),
                "model": getenv("LLM_MODEL", "gpt-4o-mini"),
            }
        }
    if hasattr(provider, "model_dump"):
        return provider.model_dump()
    return provider
