# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Trust classes: every payload crossing into the LLM's context is judged by the class of its author.

"request" is a principal's turn (may direct the agent, within policy); "ingest" is outside content
(no authority to instruct); "recall" is our own records coming back (must never instruct). These tests use plain fake classifiers, so they cover the firewall's
class handling independently of any particular Tier 2 engine.
"""

import pytest

from humanbound_firewall.firewall import Firewall
from humanbound_firewall.models import AgentConfig, Category, Verdict

_TOOL_CONV = [
    {"role": "user", "content": "Look up the details for SKU-1001"},
    {"role": "assistant", "content": "Fetching the supplier page."},
    {"role": "tool", "content": "<scraped page>"},
]


class ClassAwareClassifier:
    """A Tier 2 classifier that declares class support and records its input."""

    supports_class = True

    def __init__(self, result=None):
        self._result = result or {"decision": "BLOCK", "attack_probability": 0.9}
        self.conversation = None
        self.cls = None

    def classify(self, conversation, cls="request"):
        self.conversation, self.cls = conversation, cls
        return self._result


class LegacyClassifier:
    """A Tier 2 classifier written before trust classes existed (like the .hbfw one)."""

    def __init__(self):
        self.calls = 0

    def classify(self, conversation):
        self.calls += 1
        return {"decision": "BLOCK", "attack_probability": 0.9}


def _firewall(classifier, **config):
    return Firewall(AgentConfig(business_scope="x", **config), scope_classifier=classifier)


def test_a_trailing_tool_message_is_ingest():
    clf = ClassAwareClassifier()
    result = _firewall(clf).evaluate(_TOOL_CONV)
    assert clf.cls == "ingest"
    assert clf.conversation[0]["u"] == "Look up the details for SKU-1001"  # the request is history
    assert clf.conversation[-1]["u"] == "<scraped page>"
    assert result.prompt == "<scraped page>"
    assert result.verdict == Verdict.BLOCK
    assert result.tier == 2


def test_earlier_tool_outputs_are_kept_in_the_history_handed_to_tier_2():
    clf = ClassAwareClassifier()
    _firewall(clf).evaluate(
        [
            {"role": "user", "content": "Check the price"},
            {"role": "tool", "content": "PAGE 1"},
            {"role": "assistant", "content": "Following the link."},
            {"role": "tool", "content": "PAGE 2"},
        ]
    )
    assert clf.conversation[0] == {
        "u": "Check the price",
        "a": "Following the link.",
        "t": "PAGE 1",
    }
    assert clf.conversation[-1]["u"] == "PAGE 2"


@pytest.mark.parametrize(
    ("cls", "payload"),
    [("ingest", "<scraped page>"), ("recall", '{"sku": "SKU-1001", "cost": 42.5}')],
)
def test_ingest_and_recall_skip_the_chat_minimum_turns_gate(cls, payload):
    """An explicit class on a plain payload: a scraped page is ingest, our own record is recall."""
    clf = ClassAwareClassifier()
    _firewall(clf, tier2_min_turns=3).evaluate(payload, cls=cls)
    assert clf.cls == cls  # consulted despite a single-shot payload
    assert clf.conversation[-1]["u"] == payload


def test_request_still_waits_for_tier2_min_turns():
    clf = ClassAwareClassifier()
    result = _firewall(clf, tier2_min_turns=3).evaluate("hello")
    assert clf.cls is None  # Tier 2 not consulted on a single user turn
    assert result.verdict == Verdict.REVIEW


@pytest.mark.parametrize("method", ["evaluate", "inspect"])
def test_unknown_class_is_rejected(method):
    with pytest.raises(ValueError):
        getattr(_firewall(ClassAwareClassifier()), method)("x", cls="carrier-pigeon")


def test_legacy_classifier_is_never_handed_a_class():
    clf = LegacyClassifier()
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    conv.append({"role": "user", "content": "do the thing"})
    result = _firewall(clf, tier2_min_turns=1).evaluate(conv)
    assert clf.calls == 1
    assert result.category == Category.VIOLATION  # legacy default is unchanged


def test_legacy_classifier_is_not_consulted_for_single_shot_ingest_payloads():
    clf = LegacyClassifier()
    _firewall(clf, tier2_min_turns=3).evaluate("<scraped page>", cls="ingest")
    assert clf.calls == 0


def test_classifier_may_name_the_category_and_explanation():
    clf = ClassAwareClassifier(
        {"decision": "BLOCK", "category": "restriction", "explanation": "names a restriction"}
    )
    result = _firewall(clf).evaluate("<scraped page>", cls="ingest")
    assert result.category == Category.RESTRICTION
    assert result.explanation == "names a restriction"
