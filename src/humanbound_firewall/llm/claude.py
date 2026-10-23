# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Anthropic Claude provider."""

import inspect
import re
import time
from os import getenv

ALLOWED_MAX_OUT_TOKENS = 4096
DEFAULT_MAX_OUT_TOKENS = 2048
MAX_RETRY_COUNTER = 3
LLM_PING_TIMEOUT = 90
DEFAULT_TEMPERATURE = 0
DEFAULT_MODEL = "claude-sonnet-4-6"

# Newer Claude models think before they answer, and `max_tokens` counts the thinking as well as
# the visible reply. With only the caller's visible budget such a model can spend all of it
# thinking and return no text: for the judge, no verdict at all. So the thinking gets headroom on
# top of the visible budget; a model that does not think never reaches the extra cap.
THINKING_TOKEN_HEADROOM = 8192

# Newer Claude models also reject a non-default `temperature`. Rather than keep a list of model
# names, send it and retry without it once on the 400 that names it, then remember the model, so
# the wasted request happens once per model, not once per call (as the OpenAI provider does).
_TEMPERATURE_REJECTED = re.compile(r"temperature", re.I)
_models_without_temperature: set = set()


def _params(model: str, max_tokens: int, temperature: float) -> dict:
    params: dict = {"max_tokens": min(max_tokens, ALLOWED_MAX_OUT_TOKENS) + THINKING_TOKEN_HEADROOM}
    if model not in _models_without_temperature:
        params["temperature"] = temperature
    return params


def _rejects_temperature(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 400 and bool(
        _TEMPERATURE_REJECTED.search(str(error))
    )


def _sdk_takes_temperature(client) -> bool:
    """Whether the installed SDK's messages.create still has a `temperature` parameter. The 1.x
    SDK removed it: passing it raises a TypeError before any request is made, for every model."""
    try:
        parameters = inspect.signature(client.messages.create).parameters.values()
    except (TypeError, ValueError):  # not introspectable: assume the older signature
        return True
    return any(p.name == "temperature" or p.kind is p.VAR_KEYWORD for p in parameters)


def _create(client, model: str, params: dict, **request):
    """messages.create, without temperature when the SDK no longer takes it, and retried once
    without it if the model rejects it."""
    if "temperature" in params and not _sdk_takes_temperature(client):
        params = {k: v for k, v in params.items() if k != "temperature"}
    try:
        return client.messages.create(model=model, **params, **request)
    except Exception as e:
        if "temperature" not in params or not _rejects_temperature(e):
            raise
        _models_without_temperature.add(model)
        params = {k: v for k, v in params.items() if k != "temperature"}
        return client.messages.create(model=model, **params, **request)


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
        return _create(
            self.client,
            self.model,
            _params(self.model, max_tokens, temperature),
            system=system_p,
            messages=[{"role": "user", "content": user_p}],
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
        model = self._provider["integration"]["model"]
        # An explicit timeout, as the streamer sends: without one the SDK refuses a non-streaming
        # request whose max_tokens (with the thinking headroom) exceeds a model's non-streaming
        # cap (8192 for Opus 4 / 4.1) before sending it, with "Streaming is required...".
        return _create(
            self.client,
            model,
            _params(model, max_tokens, temperature),
            system=system_p,
            messages=[{"role": "user", "content": user_p}],
            timeout=LLM_PING_TIMEOUT,
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
        while retry_counter <= MAX_RETRY_COUNTER:
            try:
                response = self.__do_completion_api_call(system_p, user_p, max_tokens, temperature)
                # A thinking model's reply starts with a thinking block: read the first text block.
                return next(
                    (
                        block.text
                        for block in response.content
                        if getattr(block, "type", "") == "text"
                    ),
                    "",
                )
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
                "model": getenv("LLM_MODEL", DEFAULT_MODEL),
            }
        }
    if hasattr(provider, "model_dump"):
        return provider.model_dump()
    return provider
