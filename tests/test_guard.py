# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The Guard turns a verdict into an action, by mode, fail mode and trust class.

evaluate() is the filter (the caller decides); inspect() is the gateway (the Guard decides). These
tests cover the Guard alone, with hand-made EvalResults, so the rules are pinned independently of
any engine.
"""

import pytest

from humanbound_firewall.guard import Decision, Guard
from humanbound_firewall.models import Category, EvalResult, Verdict
from humanbound_firewall.session import Session

BLOCKED = EvalResult(
    verdict=Verdict.BLOCK,
    category=Category.RESTRICTION,
    explanation="names a restriction",
    tier=2,
    attack_probability=0.91,
)
PASSED = EvalResult(verdict=Verdict.PASS, category=Category.NONE, tier=2)
UNCERTAIN = EvalResult(verdict=Verdict.REVIEW, category=Category.UNCERTAIN, tier=3)
BOUNDARY = {"name": "fetch_url", "kind": "tools"}


def _decide(result, *, cls="ingest", session=None, **guard):
    return Guard(**guard).decide(
        result, cls=cls, boundary=BOUNDARY, session=session or Session.new(), elapsed_ms=7
    )


# --- block mode: the verdict becomes an action --------------------------------------------------


@pytest.mark.parametrize("cls", ["ingest", "recall"])
def test_a_blocked_ingest_or_recall_payload_is_withheld_with_a_replacement_text(cls):
    d = _decide(BLOCKED, cls=cls)
    assert isinstance(d, Decision)
    assert d.action == "withhold"
    assert "restriction" in d.replacement and "Humanbound" in d.replacement
    assert d.verdict == Verdict.BLOCK and d.category == Category.RESTRICTION and d.tier == 2
    assert d.cls == cls and d.boundary == BOUNDARY
    assert d.mode_applied == "block" and d.elapsed_ms == 7
    assert d.result is BLOCKED
    # the decision carries the probabilities and the explanation
    assert d.probabilities == {"attack": 0.91}
    assert d.explanation == "names a restriction"


def test_a_blocked_request_is_rejected_not_withheld():
    d = _decide(BLOCKED, cls="request")
    assert d.action == "reject"
    assert d.replacement  # the text the caller may return to the principal


def test_a_passed_payload_passes_with_no_replacement():
    d = _decide(PASSED)
    assert d.action == "pass" and d.replacement == ""


@pytest.mark.parametrize(
    "fail, action, replaced", [("open", "pass", False), ("closed", "withhold", True)]
)
def test_an_uncertain_verdict_passes_when_failing_open_and_is_withheld_when_failing_closed(
    fail, action, replaced
):
    d = _decide(UNCERTAIN, fail=fail)
    assert d.action == action
    assert ("uncertain" in d.replacement) is replaced


# --- log mode observes; passthrough does not even look ------------------------------------------


def test_log_mode_keeps_the_verdict_but_always_passes():
    d = _decide(BLOCKED, mode="log")
    assert d.verdict == Verdict.BLOCK and d.action == "pass" and d.replacement == ""
    assert d.mode_applied == "log"


def test_log_mode_ignores_the_fail_mode():
    assert _decide(UNCERTAIN, mode="log", fail="closed").action == "pass"


def test_passthrough_is_a_decision_without_an_evaluation_and_leaves_the_session_untouched():
    s = Session.new().record(BLOCKED, cls="ingest")
    d = Guard(mode="passthrough").passthrough(cls="ingest", boundary=BOUNDARY, session=s)
    assert d.action == "pass" and d.verdict == Verdict.PASS and d.tier == 0
    assert d.mode_applied == "passthrough" and d.result is None
    assert d.session is s


def test_mode_by_class_overrides_the_default_mode_per_class():
    g = Guard(mode="block", mode_by_class={"request": "log"})
    assert g.mode_for("request") == "log"
    assert g.mode_for("ingest") == "block" and g.mode_for("recall") == "block"


def test_a_class_that_is_not_enabled_is_off():
    g = Guard(classes=("ingest", "recall"))
    assert g.mode_for("request") == "off"
    assert g.mode_for("ingest") == "block"


def test_unknown_modes_fail_modes_and_classes_are_rejected_at_construction():
    with pytest.raises(ValueError):
        Guard(mode="shadow")
    with pytest.raises(ValueError):
        Guard(mode_by_class={"ingest": "shadow"})
    with pytest.raises(ValueError):
        Guard(fail="sometimes")
    with pytest.raises(ValueError):
        Guard(classes=("ingest", "carrier-pigeon"))
    with pytest.raises(ValueError):
        Guard(mode_by_class={"carrier-pigeon": "log"})
    with pytest.raises(ValueError):
        Guard(withheld_template="[{reason}]")  # a placeholder the template cannot fill


# --- the withheld text is a template --------------------------------------------------------------


def test_withheld_template_sees_category_explanation_class_and_boundary():
    d = _decide(BLOCKED, withheld_template="[{category}|{explanation}|{cls}|{boundary}]")
    assert d.replacement == "[restriction|names a restriction|ingest|fetch_url]"


def test_withheld_template_without_a_boundary_names_the_class():
    d = Guard(withheld_template="{boundary}").decide(
        BLOCKED, cls="recall", boundary=None, session=Session.new(), elapsed_ms=0
    )
    assert d.replacement == "recall"


# --- the engine failed: the agent's failure, by fail mode ---------------------------------------


@pytest.mark.parametrize("fail, action", [("open", "pass"), ("closed", "withhold")])
def test_an_engine_failure_passes_when_failing_open_withholds_when_failing_closed_and_says_so(
    fail, action
):
    d = Guard(fail=fail).failure(
        RuntimeError("engine timed out"),
        cls="ingest",
        boundary=BOUNDARY,
        session=Session.new(),
        elapsed_ms=5000,
    )
    assert d.action == action
    assert d.verdict == Verdict.REVIEW and d.category == Category.UNCERTAIN
    assert "engine timed out" in d.explanation and d.result is None


def test_an_engine_failure_in_log_mode_only_passes():
    d = Guard(mode="log", fail="closed").failure(
        RuntimeError("down"), cls="ingest", boundary=BOUNDARY, session=Session.new(), elapsed_ms=1
    )
    assert d.action == "pass" and d.mode_applied == "log"


# --- the session is folded in ---------------------------------------------------------------------


def test_an_engine_failure_counts_as_an_escalation_in_the_session():
    d = Guard().failure(
        RuntimeError("x"), cls="ingest", boundary=None, session=Session.new(), elapsed_ms=1
    )
    assert d.session.counts["escalations"] == 1


# --- on_decision observes and cannot interfere ----------------------------------------------------


def test_on_decision_is_called_with_every_decision():
    seen = []
    g = Guard(on_decision=seen.append)
    d1 = g.decide(BLOCKED, cls="ingest", boundary=None, session=Session.new(), elapsed_ms=1)
    d2 = g.passthrough(cls="request", boundary=None, session=Session.new())
    d3 = g.failure(
        RuntimeError("x"), cls="recall", boundary=None, session=Session.new(), elapsed_ms=1
    )
    assert seen == [d1, d2, d3]


def test_a_failing_or_returning_callback_cannot_change_the_decision(caplog):
    def bad(decision):
        raise RuntimeError("observer crashed")

    d = Guard(on_decision=bad).decide(
        BLOCKED, cls="ingest", boundary=None, session=Session.new(), elapsed_ms=1
    )
    assert d.action == "withhold"
    assert "observer crashed" in caplog.text

    d = Guard(on_decision=lambda decision: "pass").decide(
        BLOCKED, cls="ingest", boundary=None, session=Session.new(), elapsed_ms=1
    )
    assert d.action == "withhold"


def test_decision_wait_explanation_delegates_to_the_result():
    d = _decide(BLOCKED)
    assert d.wait_explanation(timeout=0) == "names a restriction"
    p = Guard().passthrough(cls="ingest", boundary=None, session=Session.new())
    assert p.wait_explanation(timeout=0) == p.explanation
