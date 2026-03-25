"""LLM provider configuration models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ProviderName(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azureopenai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ProviderIntegration(BaseModel):
    api_key: str
    model: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None


class Provider(BaseModel):
    name: ProviderName
    integration: ProviderIntegration
