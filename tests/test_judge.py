# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for judge prompt builder."""

import pytest

from humanbound_firewall.judge import (
    _format_few_shots,
    _format_session_context,
    build_system_prompt,
)
from humanbound_firewall.models import CLASSES, AgentConfig, Turn


class TestBuildSystemPrompt:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_every_judge_carries_the_policy_from_config(self, cls):
        config = AgentConfig(
            business_scope="Online banking",
            more_info="Only for retail customers",
            permitted_intents=["Check balance", "View history"],
            restricted_intents=["Transfer funds", "Close account"],
        )
        prompt = build_system_prompt(config, cls=cls)
        for text in (
            "Online banking",
            "Only for retail customers",
            "Check balance",
            "View history",
            "Transfer funds",
            "Close account",
        ):
            assert text in prompt


class TestFewShots:
    def test_truncates_long_prompts(self):
        shots = [{"prompt": "x" * 300, "verdict": "block"}]
        result = _format_few_shots(shots)
        assert "..." in result
        assert len(result) < 400


class TestSessionContext:
    def test_formats_turns_and_is_empty_without_any(self):
        assert _format_session_context(None) == ""
        assert _format_session_context([]) == ""
        turns = [
            Turn(user="Hello", assistant="Hi, how can I help?"),
            Turn(user="Check my balance"),
        ]
        result = _format_session_context(turns)
        assert "CONVERSATION CONTEXT" in result
        assert "Hello" in result
        assert "how can I help" in result
        assert "Check my balance" in result
