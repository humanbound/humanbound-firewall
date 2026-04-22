# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Data models for humanbound-firewall."""

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """Firewall verdict."""

    PASS = "pass"
    BLOCK = "block"
    REVIEW = "review"


class Category(str, Enum):
    """Block category — why the verdict was block."""

    NONE = ""
    OFF_TOPIC = "off_topic"
    VIOLATION = "violation"
    RESTRICTION = "restriction"
    UNCERTAIN = "uncertain"


# Map single-letter judge output to verdict + category
VERDICT_MAP = {
    "P": (Verdict.PASS, Category.NONE),
    "A": (Verdict.BLOCK, Category.OFF_TOPIC),
    "B": (Verdict.BLOCK, Category.VIOLATION),
    "C": (Verdict.BLOCK, Category.RESTRICTION),
    "D": (Verdict.REVIEW, Category.UNCERTAIN),
}


@dataclass
class EvalResult:
    """Result of a firewall evaluation."""

    verdict: Verdict = Verdict.PASS
    category: Category = Category.NONE
    explanation: str = ""
    latency_ms: int = 0
    session_id: str = ""
    prompt: str = ""
    raw_letter: str = ""
    tier: int = 0  # 0=sanitization, 1=classifier, 2=LLM judge
    attack_probability: float = 0.0  # Tier 1 confidence score

    @property
    def blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


@dataclass
class Turn:
    """A single conversation turn."""

    user: str = ""
    assistant: str = ""


@dataclass
class AgentConfig:
    """Agent configuration loaded from YAML."""

    name: str = ""
    version: str = "1.0"

    # Scope
    business_scope: str = ""
    more_info: str = ""

    # Intents
    permitted_intents: list[str] = field(default_factory=list)
    restricted_intents: list[str] = field(default_factory=list)

    # Settings
    timeout: int = 5
    mode: str = "block"  # block | log | passthrough
    session_window: int = 5  # number of turns for context
    tier2_min_turns: int = 3  # minimum turns before Tier 2 activates
    risk_tolerance: str = "medium"  # high | medium | low
    temperature: float = 0.0

    # Few-shot examples
    few_shots: list[dict] = field(default_factory=list)
