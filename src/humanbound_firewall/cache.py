# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Prompt cache — avoids rebuilding the system prompt on every evaluation."""

import hashlib

from .judge import build_system_prompt
from .models import AgentConfig


class PromptCache:
    """Caches the built system prompt to avoid recomputation.

    The system prompt (agent scope + intents + few-shots) is constant across
    evaluations. Only the user message and session context change. Caching
    the base prompt saves ~1-2ms per call and enables provider-level caching
    (Anthropic prompt caching, OpenAI prefix caching).
    """

    def __init__(self):
        self._base_prompts: dict[str, str] = {}
        self._config_hash: str | None = None

    def get_or_build(self, config: AgentConfig, cls: str = "request") -> str:
        """Return the cached base prompt for a trust class, building it if needed."""
        current_hash = self._hash_config(config)
        if self._config_hash != current_hash:
            self._base_prompts = {}
            self._config_hash = current_hash

        if cls not in self._base_prompts:
            # Build fresh prompt (without session context — that's added per-call)
            self._base_prompts[cls] = build_system_prompt(config, session_turns=None, cls=cls)
        return self._base_prompts[cls]

    def invalidate(self):
        """Force rebuild on next call."""
        self._base_prompts = {}
        self._config_hash = None

    @staticmethod
    def _hash_config(config: AgentConfig) -> str:
        """Hash config fields that affect the system prompt."""
        key = (
            f"{config.business_scope}|{config.more_info}|"
            f"{','.join(config.permitted_intents)}|"
            f"{','.join(config.restricted_intents)}|"
            f"{len(config.few_shots)}"
        )
        return hashlib.md5(key.encode()).hexdigest()
