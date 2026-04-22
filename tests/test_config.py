# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for configuration loading."""

from pathlib import Path

import pytest

from humanbound_firewall.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


class TestConfigLoading:
    def test_load_valid_config(self):
        config = load_config(FIXTURES / "agent.yaml")
        assert config.name == "Test Banking Agent"
        assert config.business_scope == "Online banking customer support for retail customers"
        assert len(config.permitted_intents) == 5
        assert len(config.restricted_intents) == 5
        assert config.timeout == 5
        assert config.mode == "block"

    def test_few_shots_loaded(self):
        config = load_config(FIXTURES / "agent.yaml")
        assert len(config.few_shots) == 3
        assert config.few_shots[0]["verdict"] == "block"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_defaults_applied(self, tmp_path):
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("name: test\nscope:\n  business: testing\n")
        config = load_config(minimal)
        assert config.name == "test"
        assert config.timeout == 5
        assert config.mode == "block"
        assert config.session_window == 5
        assert config.permitted_intents == []
        assert config.restricted_intents == []
