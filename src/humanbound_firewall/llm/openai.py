# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""OpenAI provider."""

import logging
import re
import time
from os import getenv

import requests

logger = logging.getLogger(__name__)

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0

OPENAI_CHAT_COMPLETION_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Every chat model accepts `max_completion_tokens`; it is the deprecated `max_tokens` that
# the reasoning families (gpt-5*, o-series, and whatever comes next) reject. So there is
# no list of model names to keep up to date: the modern parameter is sent to every model.
#
# `max_completion_tokens` counts a reasoning model's hidden reasoning tokens as well as
# its visible answer. With only the caller's visible budget, such a model can spend all
# of it thinking and return nothing — for the judge, no verdict at all. So the reasoning
# gets its own headroom on top of the visible budget; on a classic model the extra cap is
# simply never reached.
REASONING_TOKEN_HEADROOM = 8192

# Reasoning models also reject a non-default `temperature`, and the classic models reject
# `reasoning_effort`. Rather than guess which model does which, send what we have and drop the
# offending parameter on the specific 400 that names it — then remember, so the wasted request
# happens once per model, not once per call. `reasoning_effort` is only ever sent when the caller
# asked for one (ProviderIntegration.reasoning_effort); `temperature` is always sent first because
# classic models honour it.
_TEMPERATURE_REJECTED = re.compile(r"temperature", re.I)
_REASONING_EFFORT_REJECTED = re.compile(
    r"reasoning_effort|reasoning\.effort|reasoning effort", re.I
)
_models_without_temperature: set = set()
_models_without_reasoning_effort: set = set()


def _sampling_params(
    model: str, max_tokens: int, temperature: float, reasoning_effort: str | None = None
) -> dict:
    """The sampling params to send. Works for any model, present or future: a model that turns
    out to reject `temperature` or `reasoning_effort` has it recorded and dropped on the next call."""
    params: dict = {"max_completion_tokens": max_tokens + REASONING_TOKEN_HEADROOM}
    if model not in _models_without_temperature:
        params["temperature"] = temperature
    if reasoning_effort and model not in _models_without_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    return params


def _note_rejected(model: str, error_text: str, param: str | None = None) -> set:
    """Record, per model, which sampling params a 400 says this model rejects, and return the
    names recorded *for the first time* on this call. An empty set means the 400 was about
    something else (or a param already dropped) and must not be retried away — which also stops
    a mis-scoped 400 from looping forever.

    `param` is the SDK's structured `error.param`, read first: the openai SDK's message for an
    invalid value ("Invalid value: 'hihg'. Supported values are: ...") does not name the param."""
    rejected = set()
    if (
        param == "temperature" or _TEMPERATURE_REJECTED.search(error_text or "")
    ) and model not in _models_without_temperature:
        _models_without_temperature.add(model)
        rejected.add("temperature")
    if (
        param == "reasoning_effort" or _REASONING_EFFORT_REJECTED.search(error_text or "")
    ) and model not in _models_without_reasoning_effort:
        _models_without_reasoning_effort.add(model)
        rejected.add("reasoning_effort")
        # The same 400 covers a model without the param and an invalid value (a typo like
        # "hihg"); either way the configured effort is no longer applied, so say so.
        logger.warning(
            "Model %s rejected the configured reasoning_effort; continuing without it: %s",
            model,
            error_text,
        )
    return rejected


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
        self.reasoning_effort = provider["integration"].get("reasoning_effort")

    def ping(
        self, system_p, user_p, max_tokens=DEFAULT_MAX_OUT_TOKENS, temperature=DEFAULT_TEMPERATURE
    ):
        max_tokens = min(max_tokens, ALLOWED_MAX_OUT_TOKENS)
        messages: list = [  # the SDK's typed message params; a plain list is what it accepts at runtime
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ]
        params = _sampling_params(self.model, max_tokens, temperature, self.reasoning_effort)
        while True:
            try:
                return self.__client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=LLM_PING_TIMEOUT,
                    stream=True,
                    **params,
                )
            except Exception as e:
                drop = (
                    _note_rejected(self.model, str(e), getattr(e, "param", None)) & params.keys()
                    if getattr(e, "status_code", None) == 400
                    else set()
                )
                if not drop:  # a 400 about anything else is a real error, not a param to retry away
                    raise
                for name in drop:
                    params.pop(name)


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
                **_sampling_params(
                    self._provider["integration"]["model"],
                    max_tokens,
                    temperature,
                    self._provider["integration"].get("reasoning_effort"),
                ),
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
                if _note_rejected(self._provider["integration"]["model"], resp.text):
                    continue  # retry; the next request omits the param(s) the model rejected
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
