# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""YAML configuration loader."""

from pathlib import Path

import yaml

from .models import AgentConfig


def load_config(path: str | Path) -> AgentConfig:
    """Load agent configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}")

    scope = data.get("scope", {})
    intents = data.get("intents", {})
    settings = data.get("settings", {})

    return AgentConfig(
        name=data.get("name", ""),
        version=str(data.get("version", "1.0")),
        business_scope=scope.get("business", ""),
        more_info=scope.get("more_info", ""),
        permitted_intents=intents.get("permitted", []),
        restricted_intents=intents.get("restricted", []),
        timeout=settings.get("timeout", 5),
        mode=settings.get("mode", "block"),
        session_window=settings.get("session_window", 5),
        tier2_min_turns=settings.get("tier2_min_turns", 3),
        risk_tolerance=settings.get("risk_tolerance", "medium"),
        temperature=settings.get("temperature", 0.0),
        few_shots=data.get("few_shots", []),
    )
