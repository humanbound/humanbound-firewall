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


class Provider(BaseModel):
    name: ProviderName
    integration: ProviderIntegration
