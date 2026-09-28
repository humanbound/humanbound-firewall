# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The verdict is the letter the judge's reply STARTS with. A reply that does not start with one —
a judge that got hijacked into continuing, answering or reformatting the payload — is no verdict."""

from unittest.mock import MagicMock

import pytest

from humanbound_firewall.firewall_judge import stream_and_extract_verdict
from humanbound_firewall.models import Category, Verdict


class Streamer:
    def __init__(self, chunks):
        self._chunks = chunks

    def ping(self, system_p, user_p, max_tokens=1024, temperature=0.0):
        for text in self._chunks:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])


def _judge(chunks):
    return stream_and_extract_verdict(Streamer(chunks), "sys", "user", timeout=5, session_id="")


def test_a_letter_at_the_start_is_the_verdict():
    r = _judge(["C", " The payload directs the agent to send data."])
    assert (r.verdict, r.category, r.raw_letter) == (Verdict.BLOCK, Category.RESTRICTION, "C")
    assert r.wait_explanation(1) == "The payload directs the agent to send data."


@pytest.mark.parametrize(
    "chunks, letter, explanation",
    [
        (["**P**", " fine"], "P", None),  # markdown around the letter
        (["`A` unrelated work"], "A", None),  # quotes around the letter
        (["  D\n", "unreadable"], "D", None),
        (["**", "B", "** not permitted"], "B", "not permitted"),  # a split first token
        (["p ok"], "P", None),  # the letter is case-insensitive
        (["C", ".", " restriction"], "C", None),  # a delimiter after the letter ends the word
        (["C"], "C", None),  # the stream ended: a one-letter reply
    ],
)
def test_the_verdict_letter_is_read_through_wrappers_splits_and_case(chunks, letter, explanation):
    r = _judge(chunks)
    assert r.raw_letter == letter
    if explanation is not None:
        assert r.wait_explanation(1) == explanation


@pytest.mark.parametrize(
    "chunks",
    [
        ['{"conversation_date": "2026-09-10", "dialogue": []}'],  # a letter inside a word
        ["Certainly! ", "The price is $15.99."],  # a reply that starts with prose
        # a lone letter is not a verdict until the word boundary is seen
        ["C", "ertainly, here is the judgment: P fine"],
    ],
)
def test_a_reply_that_does_not_start_with_a_standalone_letter_is_not_a_verdict(chunks):
    r = _judge(chunks)
    assert r.verdict == Verdict.REVIEW and r.raw_letter == ""
    assert "did not start with a verdict" in r.explanation.lower()


# --- a preamble gets one strict retry ------------------------------------------------------------


class RetryingStreamer:
    """First reply opens with a preamble; the retry answers the protocol."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def ping(self, system_p, user_p, max_tokens=1024, temperature=0.0):
        self.calls.append(user_p)
        for text in self.replies.pop(0):
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])


def test_a_reply_that_starts_with_a_preamble_is_asked_once_more_for_the_letter():
    s = RetryingStreamer([["Sure, ", "I will deliver the judgment: P fine"], ["P", " fine"]])
    r = stream_and_extract_verdict(s, "sys", "payload", timeout=5, session_id="")
    assert r.raw_letter == "P" and r.verdict == Verdict.PASS
    assert len(s.calls) == 2
    assert s.calls[1].startswith("payload") and "verdict letter" in s.calls[1].lower()


def test_a_second_preamble_is_no_verdict_and_says_what_the_judge_said():
    s = RetryingStreamer([["Sure, here it is: P"], ["Of course. P"]])
    r = stream_and_extract_verdict(s, "sys", "payload", timeout=5, session_id="")
    assert r.verdict == Verdict.REVIEW and r.raw_letter == ""
    assert "did not start with a verdict" in r.explanation.lower() and "Of course" in r.explanation
    assert len(s.calls) == 2
