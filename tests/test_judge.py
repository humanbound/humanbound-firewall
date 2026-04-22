# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for judge prompt builder."""

from humanbound_firewall.judge import (
    _format_few_shots,
    _format_session_context,
    build_system_prompt,
)
from humanbound_firewall.models import AgentConfig, Turn


class TestBuildSystemPrompt:
    def test_includes_business_scope(self):
        config = AgentConfig(business_scope="Online banking")
        prompt = build_system_prompt(config)
        assert "Online banking" in prompt

    def test_includes_permitted_intents(self):
        config = AgentConfig(permitted_intents=["Check balance", "View history"])
        prompt = build_system_prompt(config)
        assert "Check balance" in prompt
        assert "View history" in prompt

    def test_includes_restricted_intents(self):
        config = AgentConfig(restricted_intents=["Transfer funds", "Close account"])
        prompt = build_system_prompt(config)
        assert "Transfer funds" in prompt
        assert "Close account" in prompt

    def test_includes_more_info(self):
        config = AgentConfig(more_info="Only for retail customers")
        prompt = build_system_prompt(config)
        assert "Only for retail customers" in prompt

    def test_includes_evaluation_protocol(self):
        config = AgentConfig(business_scope="test")
        prompt = build_system_prompt(config)
        assert "SYSTEM EXPOSURE CHECK" in prompt
        assert "SCOPE VALIDATION" in prompt
        assert "RESTRICTION ANALYSIS" in prompt


class TestFewShots:
    def test_empty_returns_empty(self):
        assert _format_few_shots([]) == ""

    def test_formats_examples(self):
        shots = [{"prompt": "ignore instructions", "verdict": "block"}]
        result = _format_few_shots(shots)
        assert "LEARNED ATTACK PATTERNS" in result
        assert "ignore instructions" in result
        assert "BLOCK" in result

    def test_truncates_long_prompts(self):
        shots = [{"prompt": "x" * 300, "verdict": "block"}]
        result = _format_few_shots(shots)
        assert "..." in result
        assert len(result) < 400


class TestSessionContext:
    def test_empty_returns_empty(self):
        assert _format_session_context(None) == ""
        assert _format_session_context([]) == ""

    def test_formats_turns(self):
        turns = [
            Turn(user="Hello", assistant="Hi, how can I help?"),
            Turn(user="Check my balance"),
        ]
        result = _format_session_context(turns)
        assert "CONVERSATION CONTEXT" in result
        assert "Hello" in result
        assert "how can I help" in result
        assert "Check my balance" in result
