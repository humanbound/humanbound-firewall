# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Azure OpenAI provider."""

import time
from os import getenv

import requests

# Same parameter handling as the OpenAI provider (see there for the reasoning): the modern
# `max_completion_tokens` for every deployment, and `temperature` retried away once on the 400
# that names it. Azure deployment names are arbitrary, so a model-name list could never work here.
from .openai import _models_without_temperature, _rejects_temperature, _sampling_params

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0


class LLMStreamer:
    def __init__(self, provider=None):
        provider = _resolve(provider)
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError(
                "Azure OpenAI provider requires the [openai] extra. "
                "Install with: pip install humanbound-firewall[openai]"
            ) from e
        integ = provider["integration"]
        self.__client = AzureOpenAI(
            api_key=integ["api_key"],
            # A Provider dumps an unset api_version as None, so .get()'s default would not apply.
            api_version=integ.get("api_version") or "2024-06-01",
            azure_endpoint=integ.get("endpoint", ""),
        )
        self.model = integ["model"]

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        messages: list = [  # the SDK's typed message params; a plain list is what it accepts at runtime
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ]
        params = _sampling_params(self.model, max_tokens, temperature)
        try:
            return self.__client.chat.completions.create(
                model=self.model, messages=messages, timeout=LLM_PING_TIMEOUT, stream=True, **params
            )
        except Exception as e:
            if "temperature" not in params or not (
                getattr(e, "status_code", None) == 400 and _rejects_temperature(str(e))
            ):
                raise
            _models_without_temperature.add(self.model)
            params.pop("temperature")
            return self.__client.chat.completions.create(
                model=self.model, messages=messages, timeout=LLM_PING_TIMEOUT, stream=True, **params
            )


class LLMPinger:
    def __init__(self, provider=None):
        self._provider = _resolve(provider)

    def __do_completion_api_call(self, system_p, user_p, max_tokens, temperature):
        integ = self._provider["integration"]
        return requests.post(
            integ.get("endpoint", ""),
            headers={
                "Content-Type": "application/json",
                "Api-Key": integ["api_key"],
            },
            json={
                "model": integ["model"],
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                ],
                **_sampling_params(integ["model"], max_tokens, temperature),
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
                model = self._provider["integration"]["model"]
                if model not in _models_without_temperature and _rejects_temperature(resp.text):
                    _models_without_temperature.add(model)
                    continue  # retry once, now without temperature
                raise Exception(f"502/Inappropriate content ({resp.text}).")
            else:
                raise Exception(f"502/Error pinging LLM - {resp.status_code}/{resp.text}")


def _resolve(provider):
    if provider is None:
        return {
            "integration": {
                "api_key": getenv("LLM_API_KEY", ""),
                "model": getenv("LLM_MODEL", "gpt-4o-mini"),
                "endpoint": getenv("LLM_ENDPOINT", ""),
                "api_version": getenv("LLM_API_VERSION", "2024-06-01"),
            }
        }
    if hasattr(provider, "model_dump"):
        return provider.model_dump()
    return provider
