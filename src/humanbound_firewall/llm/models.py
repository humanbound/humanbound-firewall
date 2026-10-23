# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""LLM provider configuration models."""

from enum import Enum

from pydantic import BaseModel


class ProviderName(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azureopenai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ProviderIntegration(BaseModel):
    api_key: str
    model: str
    endpoint: str | None = None
    api_version: str | None = None
    # Reasoning effort for the judge, when the model is a reasoning model that accepts it
    # (e.g. OpenAI "minimal" | "low" | "medium" | "high"). Left None, no effort is sent and
    # the model's own default applies. A model that rejects the parameter has it dropped on
    # the first 400 and remembered, the same way an unsupported temperature is.
    reasoning_effort: str | None = None


class Provider(BaseModel):
    name: ProviderName
    integration: ProviderIntegration
