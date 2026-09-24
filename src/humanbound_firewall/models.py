# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Data models for humanbound-firewall."""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Firewall verdict."""

    PASS = "pass"
    BLOCK = "block"
    REVIEW = "review"


# Trust classes: who authored the payload decides how it is judged. "request" is a
# principal's turn (may direct the agent, within policy); "ingest" is outside content
# (tool output, pages, documents) that carries no authority; "recall" is the agent's own
# records coming back, which must never instruct.
CLASSES = ("request", "ingest", "recall")

# The kinds of boundary an agent has, in the platform's vocabulary (the project's capabilities
# flag): where content can enter the model's context besides the principal's turn.
CAPABILITIES = ("tools", "memory", "inter_agent", "reasoning_model")


class Category(str, Enum):
    """Block category — why the verdict was block."""

    NONE = ""
    OFF_TOPIC = "off_topic"
    VIOLATION = "violation"
    RESTRICTION = "restriction"
    INTEGRITY = "integrity"
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
    session: Any = None  # the Session after this verdict was folded in (token out)

    # Set by the Tier 3 judge while the explanation is still streaming in.
    _explanation_ready: threading.Event | None = field(default=None, repr=False, compare=False)

    def wait_explanation(self, timeout: float | None = None) -> str:
        """Return the explanation, waiting for it to finish streaming if needed.

        The Tier 3 judge returns as soon as the verdict letter arrives and streams
        the explanation in afterwards, so `explanation` can be empty right after
        `evaluate()`. Call this when you need the text (logs, reports). Returns
        whatever has arrived if `timeout` elapses first.
        """
        if self._explanation_ready is not None:
            self._explanation_ready.wait(timeout=timeout)
        return self.explanation

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
    tool: str = ""  # tool output / retrieved content that entered the agent in this turn


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
    session_window: int = 5  # unused; removed in 0.4
    tier2_min_turns: int = 3  # minimum turns before Tier 2 activates
    temperature: float = 0.0  # unused (the judge runs at 0); removed in 0.4

    # Few-shot examples
    few_shots: list[dict] = field(default_factory=list)

    # Boundaries: what kinds exist (capabilities) and, optionally, a whitebox `tools:` block that
    # relaxes the default (every tool is ingest) for tools returning our own records (recall).
    capabilities: list[str] = field(default_factory=lambda: ["tools"])
    tool_classes: dict[str, str] = field(default_factory=dict)  # tool name -> ingest | recall
    tool_expects: dict[str, str] = field(default_factory=dict)  # tool name -> what it returns

    def class_of(self, tool_name: str) -> str:
        """The trust class of a tool's output: as declared under `tools:`, else ingest."""
        return self.tool_classes.get(tool_name, "ingest")
