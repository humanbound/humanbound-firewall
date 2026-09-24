# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Guard — turns a verdict into an action, by mode, fail mode and trust class.

`Firewall.evaluate()` is the filter: it reports a verdict and the caller decides. `Firewall.inspect()`
is the gateway: the Guard decides, and adapters enforce what it says. Everything that shapes a
decision is set once, in code, at `from_config()` time:

- ``classes``        the trust classes enabled in this deployment (others are "off": not evaluated);
- ``mode``           block | log | passthrough, overridable per class with ``mode_by_class``;
- ``fail``           open | closed — what an uncertain verdict or an engine failure means in block mode;
- ``on_decision``    an observe-only callback, fired for every Decision; it cannot change one;
- ``withheld_template``  the text that replaces a withheld payload (``{category}``, ``{explanation}``,
  ``{cls}``, ``{boundary}``).

Actions: a blocked ingest or recall payload is *withheld* (replaced by the template, the run goes on
without it); a blocked request is *rejected* (the caller returns the replacement to the principal);
everything else *passes*. In log mode every decision passes but keeps its verdict; in passthrough
mode nothing is evaluated at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import CLASSES, Category, EvalResult, Verdict
from .session import Session

logger = logging.getLogger(__name__)

MODES = ("block", "log", "passthrough")
FAIL_MODES = ("open", "closed")
ACTIONS = ("pass", "withhold", "reject")

DEFAULT_WITHHELD_TEMPLATE = (
    "[Content withheld by Humanbound Firewall: {category}. Continue the task without it.]"
)
_TEMPLATE_FIELDS = ("category", "explanation", "cls", "boundary")


@dataclass(frozen=True)
class Decision:
    """What the gateway decided about one payload, and the session after it."""

    verdict: Verdict
    category: Category
    tier: int
    probabilities: dict[str, float]
    explanation: str
    action: str  # "pass" | "withhold" | "reject"
    replacement: str  # the text standing in for the payload when it is withheld or rejected
    session: Session
    cls: str
    boundary: dict[str, Any] | None
    mode_applied: str  # "block" | "log" | "passthrough" | "off"
    elapsed_ms: int
    result: EvalResult | None = field(default=None, compare=False, repr=False)

    def wait_explanation(self, timeout: float | None = None) -> str:
        """The explanation, waiting for the Tier 3 judge to finish streaming it if needed."""
        if self.result is not None:
            return self.result.wait_explanation(timeout=timeout)
        return self.explanation


class Guard:
    def __init__(
        self,
        *,
        classes: tuple[str, ...] | list[str] = CLASSES,
        mode: str = "block",
        mode_by_class: dict[str, str] | None = None,
        fail: str = "open",
        on_decision: Callable[[Decision], Any] | None = None,
        withheld_template: str | None = None,
    ):
        self.classes = tuple(classes)
        self.mode = mode
        self.mode_by_class = dict(mode_by_class or {})
        self.fail = fail
        self.on_decision = on_decision
        self.withheld_template = withheld_template or DEFAULT_WITHHELD_TEMPLATE
        self._validate()

    def _validate(self) -> None:
        unknown = [c for c in self.classes if c not in CLASSES]
        if unknown:
            raise ValueError(f"Unknown trust class(es) {unknown}. Expected a subset of {CLASSES}.")
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected one of {MODES}.")
        for cls, mode in self.mode_by_class.items():
            if cls not in CLASSES:
                raise ValueError(f"mode_by_class names unknown class '{cls}'. Expected {CLASSES}.")
            if mode not in MODES:
                raise ValueError(f"mode_by_class['{cls}'] = '{mode}'. Expected one of {MODES}.")
        if self.fail not in FAIL_MODES:
            raise ValueError(f"Unknown fail mode '{self.fail}'. Expected one of {FAIL_MODES}.")
        try:
            self.withheld_template.format(**dict.fromkeys(_TEMPLATE_FIELDS, ""))
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"withheld_template may only use {_TEMPLATE_FIELDS}: {e!r}") from e

    # ------------------------------------------------------------------ modes
    def mode_for(self, cls: str) -> str:
        """The mode applied to a class: its own, else the default; "off" if the class is disabled."""
        if cls not in self.classes:
            return "off"
        return self.mode_by_class.get(cls, self.mode)

    # ------------------------------------------------------------------ decisions
    def decide(
        self,
        result: EvalResult,
        *,
        cls: str,
        boundary: dict[str, Any] | None,
        session: Session,
        elapsed_ms: int,
    ) -> Decision:
        """Apply the class's mode and the fail mode to an evaluation."""
        mode = self.mode_for(cls)
        action = self._action(result.verdict, cls) if mode == "block" else "pass"
        replacement = self._replacement(result, cls, boundary) if action != "pass" else ""
        probabilities = {"attack": result.attack_probability}
        return self._emit(
            Decision(
                verdict=result.verdict,
                category=result.category,
                tier=result.tier,
                probabilities=probabilities,
                explanation=result.explanation,
                action=action,
                replacement=replacement,
                session=session.record(result, cls=cls, boundary=_name(boundary)),
                cls=cls,
                boundary=boundary,
                mode_applied=mode,
                elapsed_ms=elapsed_ms,
                result=result,
            )
        )

    def failure(
        self,
        error: BaseException,
        *,
        cls: str,
        boundary: dict[str, Any] | None,
        session: Session,
        elapsed_ms: int,
    ) -> Decision:
        """The engine failed: the agent's failure, resolved by the fail mode (in block mode)."""
        mode = self.mode_for(cls)
        result = EvalResult(
            verdict=Verdict.REVIEW,
            category=Category.UNCERTAIN,
            explanation=f"Engine failure: {error}",
            latency_ms=elapsed_ms,
        )
        action = self._action(result.verdict, cls) if mode == "block" else "pass"
        replacement = self._replacement(result, cls, boundary) if action != "pass" else ""
        return self._emit(
            Decision(
                verdict=result.verdict,
                category=result.category,
                tier=0,
                probabilities={},
                explanation=result.explanation,
                action=action,
                replacement=replacement,
                session=session.record(result, cls=cls, boundary=_name(boundary)),
                cls=cls,
                boundary=boundary,
                mode_applied=mode,
                elapsed_ms=elapsed_ms,
            )
        )

    def passthrough(
        self, *, cls: str, boundary: dict[str, Any] | None, session: Session
    ) -> Decision:
        """A decision without an evaluation: the class is in passthrough mode or disabled."""
        mode = self.mode_for(cls)
        return self._emit(
            Decision(
                verdict=Verdict.PASS,
                category=Category.NONE,
                tier=0,
                probabilities={},
                explanation="Not evaluated.",
                action="pass",
                replacement="",
                session=session,
                cls=cls,
                boundary=boundary,
                mode_applied=mode if mode in ("passthrough", "off") else "passthrough",
                elapsed_ms=0,
            )
        )

    # ------------------------------------------------------------------ helpers
    def _action(self, verdict: Verdict, cls: str) -> str:
        stop = "reject" if cls == "request" else "withhold"
        if verdict == Verdict.BLOCK:
            return stop
        if verdict == Verdict.REVIEW and self.fail == "closed":
            return stop
        return "pass"

    def _replacement(self, result: EvalResult, cls: str, boundary) -> str:
        return self.withheld_template.format(
            category=result.category.value or "blocked",
            explanation=result.explanation,
            cls=cls,
            boundary=_name(boundary) or cls,
        )

    def _emit(self, decision: Decision) -> Decision:
        if self.on_decision is not None:
            try:
                self.on_decision(decision)
            except Exception:  # observe-only: an observer can never break the agent
                logger.warning("on_decision callback failed", exc_info=True)
        return decision


def _name(boundary) -> str:
    return str((boundary or {}).get("name", "")) if isinstance(boundary, dict) else ""
