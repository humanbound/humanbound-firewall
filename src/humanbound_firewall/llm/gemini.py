# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Google Gemini provider."""

import time
from os import getenv

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 1024
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0.0


def _import_genai():
    try:
        import google.generativeai as genai

        return genai
    except ImportError as e:
        raise ImportError(
            "Gemini provider requires the [gemini] extra. "
            "Install with: pip install humanbound-firewall[gemini]"
        ) from e


class LLMStreamer:
    def __init__(self, provider=None):
        provider = _resolve(provider)
        genai = _import_genai()
        genai.configure(api_key=provider["integration"]["api_key"])
        self.model = genai.GenerativeModel(provider["integration"]["model"])

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        genai = _import_genai()
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        prompt = f"{system_p}\n{user_p}"
        return self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
            stream=True,
        )


class LLMPinger:
    def __init__(self, provider=None):
        self._provider = _resolve(provider)
        genai = _import_genai()
        genai.configure(api_key=self._provider["integration"]["api_key"])
        self.model = genai.GenerativeModel(self._provider["integration"]["model"])

    def __do_completion_api_call(self, prompt, max_tokens, temperature):
        genai = _import_genai()
        return self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        retry_counter = 0
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        prompt = f"{system_p}\n{user_p}"
        while retry_counter <= MAX_RETRY_COUNTER:
            try:
                response = self.__do_completion_api_call(prompt, max_tokens, temperature)
                return response.text
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    retry_counter += 1
                    if retry_counter <= MAX_RETRY_COUNTER:
                        time.sleep(retry_counter)
                        continue
                    raise Exception("502/Rate limit error.")
                raise Exception(f"502/Error pinging LLM - {str(e)}")


def _resolve(provider):
    if provider is None:
        return {
            "integration": {
                "api_key": getenv("LLM_API_KEY", ""),
                "model": getenv("LLM_MODEL", "gemini-pro"),
            }
        }
    if hasattr(provider, "model_dump"):
        return provider.model_dump()
    return provider
