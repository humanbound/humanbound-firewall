# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The Tier 3 verdict returns on the first token; the explanation streams in after.

Callers that need the explanation (logs, reports) can wait for it explicitly.
"""

import time
from unittest.mock import MagicMock

import pytest

from humanbound_firewall.firewall import Firewall
from humanbound_firewall.models import AgentConfig, EvalResult, Verdict


def _chunk(text):
    return MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])


class SlowExplainer:
    """Yields the verdict letter at once, then the explanation after a delay."""

    def __init__(self, delay=0.3):
        self._delay = delay

    def ping(self, system_p, user_p, max_tokens=1024, temperature=0.0):
        yield _chunk("B")
        yield _chunk(" The payload directs")  # the word boundary: the verdict is known here
        time.sleep(self._delay)  # ... long before the explanation has finished streaming
        yield _chunk(" a non-permitted action.")


def test_verdict_returns_before_the_explanation_has_streamed():
    fw = Firewall(AgentConfig(business_scope="x"), SlowExplainer(delay=0.5))
    t0 = time.time()
    result = fw.evaluate("payload")
    assert result.verdict == Verdict.BLOCK
    assert time.time() - t0 < 0.4  # did not wait for the explanation


def test_wait_explanation_returns_the_full_text():
    fw = Firewall(AgentConfig(business_scope="x"), SlowExplainer(delay=0.2))
    result = fw.evaluate("payload")
    assert result.wait_explanation(timeout=5) == "The payload directs a non-permitted action."
    assert result.explanation == "The payload directs a non-permitted action."


def test_wait_explanation_gives_up_after_the_timeout():
    fw = Firewall(AgentConfig(business_scope="x"), SlowExplainer(delay=2))
    result = fw.evaluate("payload")
    t0 = time.time()
    assert result.wait_explanation(timeout=0.1) == ""
    assert time.time() - t0 < 1


def test_wait_explanation_is_immediate_for_results_without_a_stream():
    result = EvalResult(explanation="Tier 1: attack detected")
    assert result.wait_explanation(timeout=5) == "Tier 1: attack detected"


class NoVerdictStreamer:
    """A judge that finishes without ever producing a verdict letter."""

    def __init__(self, text=""):
        self._text = text

    def ping(self, system_p, user_p, max_tokens=1024, temperature=0.0):
        if self._text:
            yield _chunk(self._text)


@pytest.mark.parametrize(
    "text, says",
    [
        ("", "no verdict"),  # the stream ends empty
        ("???", None),  # text, but no verdict letter (even after the retry)
    ],
)
def test_stream_that_ends_without_a_verdict_returns_review_at_once(text, says):
    fw = Firewall(AgentConfig(business_scope="x", timeout=30), NoVerdictStreamer(text))
    t0 = time.time()
    result = fw.evaluate("payload")
    assert time.time() - t0 < 2  # did not wait out the 30 s timeout
    assert result.verdict == Verdict.REVIEW
    if says:
        assert says in result.explanation.lower()
