# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Firewall.inspect(): the gateway call — evaluate, then let the Guard decide; the session travels."""

from pathlib import Path

import pytest

from humanbound_firewall.firewall import Firewall
from humanbound_firewall.guard import Decision
from humanbound_firewall.models import AgentConfig, Category, EvalResult, Verdict
from humanbound_firewall.session import Session

FIXTURES = Path(__file__).parent / "fixtures"
BOUNDARY = {"name": "fetch_url", "kind": "tools", "description": "fetch a web page"}


class FakeClassifier:
    supports_class = True

    def __init__(self, decision="BLOCK", *, tightened=None):
        self._decision = decision
        self.calls = []
        self._tightened = tightened

    def classify(self, conversation, cls="request"):
        self.calls.append((conversation, cls))
        return {"decision": self._decision, "category": "restriction", "attack_probability": 0.9}

    def tightened(self):
        return self._tightened if self._tightened is not None else self


class BrokenClassifier(FakeClassifier):
    def classify(self, conversation, cls="request"):
        raise RuntimeError("engine down")


def _firewall(classifier=None, **options):
    return Firewall(AgentConfig(business_scope="x"), scope_classifier=classifier, **options)


# --- the gateway call -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine, verdict, category, action, probabilities, replacement",
    [
        ("BLOCK", Verdict.BLOCK, Category.RESTRICTION, "withhold", {"attack": 0.9}, "restriction"),
        ("ALLOW", Verdict.PASS, Category.NONE, "pass", {"attack": 0.0}, ""),
    ],
)
def test_inspect_returns_a_decision_built_on_the_evaluation(
    engine, verdict, category, action, probabilities, replacement
):
    clf = FakeClassifier(engine)
    d = _firewall(clf).inspect("<scraped page>", cls="ingest", boundary=BOUNDARY)
    assert isinstance(d, Decision)
    assert d.verdict == verdict and d.category == category and d.tier == 2
    assert d.action == action
    assert replacement in d.replacement if replacement else d.replacement == ""
    assert d.result.prompt == "<scraped page>" and d.boundary == BOUNDARY
    assert d.probabilities == probabilities
    assert isinstance(d.elapsed_ms, int)
    assert clf.calls[0][1] == "ingest"


def test_inspect_honours_the_mode_per_class():
    # tier2_min_turns=0 lets a single-shot request reach Tier 2, so both payloads are BLOCKed and
    # only the class's mode (not the fail mode) decides the action
    fw = Firewall(
        AgentConfig(business_scope="x", tier2_min_turns=0),
        scope_classifier=FakeClassifier(),
        mode="block",
        mode_by_class={"request": "log"},
    )
    ingest = fw.inspect("<page>", cls="ingest")
    assert ingest.verdict == Verdict.BLOCK
    assert ingest.action == "withhold" and ingest.mode_applied == "block"
    request = fw.inspect("do the thing", cls="request")
    assert request.verdict == Verdict.BLOCK
    assert request.action == "pass" and request.mode_applied == "log"


def test_passthrough_mode_does_not_consult_any_engine():
    clf = FakeClassifier()
    d = _firewall(clf, mode="passthrough").inspect("<page>", cls="ingest")
    assert d.action == "pass" and d.mode_applied == "passthrough" and clf.calls == []


def test_a_disabled_class_is_off_and_not_evaluated():
    clf = FakeClassifier()
    d = _firewall(clf, classes=("ingest",)).inspect("hello", cls="request")
    assert d.action == "pass" and d.mode_applied == "off" and clf.calls == []


def test_an_unknown_class_is_rejected():
    with pytest.raises(ValueError):
        _firewall(FakeClassifier()).inspect("x", cls="carrier-pigeon")


def test_the_config_mode_is_the_default_guard_mode():
    assert _firewall(FakeClassifier()).guard.mode_for("ingest") == "block"
    fw = Firewall(AgentConfig(business_scope="x", mode="log"), scope_classifier=FakeClassifier())
    assert fw.guard.mode_for("ingest") == "log"
    fw = Firewall(
        AgentConfig(business_scope="x", mode="log"), scope_classifier=FakeClassifier(), mode="block"
    )
    assert fw.guard.mode_for("ingest") == "block"


# --- the engine failed ----------------------------------------------------------------------------


def test_an_engine_failure_is_the_agents_failure_by_fail_mode():
    d = _firewall(BrokenClassifier(), fail="open").inspect("<page>", cls="ingest")
    assert d.action == "pass" and d.verdict == Verdict.REVIEW and "engine down" in d.explanation
    d = _firewall(BrokenClassifier(), fail="closed").inspect("<page>", cls="ingest")
    assert d.action == "withhold"


def test_evaluate_still_raises_on_an_engine_failure():
    """The filter reports; the caller is the gateway and decides what a failure means."""
    with pytest.raises(RuntimeError):
        _firewall(BrokenClassifier()).evaluate("<page>", cls="ingest")


# --- the session travels: token in, token out --------------------------------------------------------


def test_inspect_folds_the_verdict_into_the_session_it_is_given():
    fw = _firewall(FakeClassifier())
    s0 = Session.new()
    d1 = fw.inspect("<page 1>", cls="ingest", session=s0, boundary=BOUNDARY)
    d2 = fw.inspect("<page 2>", cls="ingest", session=d1.session, boundary=BOUNDARY)
    assert s0.counts["evaluations"] == 0
    assert d1.session.counts == {
        "evaluations": 1,
        "blocks": 1,
        "escalations": 0,
        "by_class": {"ingest": 1},
        "by_category": {"restriction": 1},
    }
    assert d1.session.recent[-1]["b"] == "fetch_url" and d1.session.posture == "elevated"
    assert d1.session.counts["evaluations"] == 1 and d2.session.counts["evaluations"] == 2
    assert d2.session.recent[-1]["b"] == "fetch_url" and d2.session.recent[-1]["cls"] == "ingest"
    assert d2.session.posture == "elevated"


@pytest.mark.parametrize("given, evaluations", [("nothing", 1), ("a token", 2)])
def test_inspect_starts_a_fresh_session_or_continues_a_token(given, evaluations):
    fw = _firewall(FakeClassifier())
    token = None if given == "nothing" else fw.inspect("<page>", cls="ingest").session.to_json()
    d = fw.inspect("<page>", cls="ingest", session=token)
    assert isinstance(d.session, Session) and d.session.counts["evaluations"] == evaluations


def test_evaluate_also_returns_the_updated_session():
    fw = _firewall(FakeClassifier())
    r = fw.evaluate("<page>", cls="ingest", session=Session.new(), boundary=BOUNDARY)
    assert r.session.counts["blocks"] == 1 and r.session.recent[-1]["b"] == "fetch_url"
    assert fw.evaluate("<page>", cls="ingest").session.counts["evaluations"] == 1


@pytest.mark.parametrize(
    "posture, verdict, tightened",
    [("elevated", Verdict.BLOCK, True), ("normal", Verdict.PASS, False)],
)
def test_an_elevated_posture_uses_the_tightened_classifier_and_a_normal_one_does_not(
    posture, verdict, tightened
):
    strict = FakeClassifier("BLOCK")
    lenient = FakeClassifier("ALLOW", tightened=strict)
    session = Session.new()
    if posture == "elevated":
        earlier_block = EvalResult(verdict=Verdict.BLOCK, category=Category.RESTRICTION)
        session = session.record(earlier_block, cls="ingest")
    assert session.posture == posture
    d = _firewall(lenient).inspect("<page 2>", cls="ingest", session=session)
    assert d.verdict == verdict and bool(strict.calls) is tightened


# --- the window is the recent transcript ---------------------------------------------------------


def test_the_window_becomes_the_history_handed_to_tier_2():
    clf = FakeClassifier()
    window = [
        {"role": "user", "content": "Check the price"},
        {"role": "tool", "content": "PAGE 1"},
        {"role": "assistant", "content": "Following the link."},
    ]
    _firewall(clf).inspect("PAGE 2", cls="ingest", window=window)
    conversation, cls = clf.calls[0]
    assert conversation[0] == {"u": "Check the price", "a": "Following the link.", "t": "PAGE 1"}
    assert conversation[-1]["u"] == "PAGE 2" and cls == "ingest"


# --- from_config takes the deployment choices ---------------------------------------------------------


def test_from_config_wires_the_guard():
    seen = []
    fw = Firewall.from_config(
        FIXTURES / "agent.yaml",
        classes=("ingest", "recall"),
        mode="log",
        mode_by_class={"ingest": "block"},
        fail="closed",
        on_decision=seen.append,
        withheld_template="[gone: {category}]",
    )
    assert fw.guard.mode_for("ingest") == "block"
    assert fw.guard.mode_for("recall") == "log"
    assert fw.guard.mode_for("request") == "off"
    assert fw.guard.fail == "closed"
    assert fw.guard.withheld_template == "[gone: {category}]"
    d = fw.guard.passthrough(cls="ingest", boundary=None, session=Session.new())
    assert seen == [d]
