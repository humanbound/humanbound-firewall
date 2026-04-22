# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for data models."""

from humanbound_firewall.models import VERDICT_MAP, Category, EvalResult, Verdict


class TestEvalResult:
    def test_blocked_property(self):
        r = EvalResult(verdict=Verdict.BLOCK, category=Category.VIOLATION)
        assert r.blocked is True
        assert r.passed is False

    def test_passed_property(self):
        r = EvalResult(verdict=Verdict.PASS)
        assert r.passed is True
        assert r.blocked is False

    def test_review_is_not_blocked(self):
        r = EvalResult(verdict=Verdict.REVIEW)
        assert r.blocked is False
        assert r.passed is False


class TestVerdictMap:
    def test_all_letters_mapped(self):
        for letter in ["P", "A", "B", "C", "D"]:
            assert letter in VERDICT_MAP

    def test_p_is_pass(self):
        verdict, category = VERDICT_MAP["P"]
        assert verdict == Verdict.PASS
        assert category == Category.NONE

    def test_a_is_off_topic(self):
        verdict, category = VERDICT_MAP["A"]
        assert verdict == Verdict.BLOCK
        assert category == Category.OFF_TOPIC

    def test_b_is_violation(self):
        verdict, category = VERDICT_MAP["B"]
        assert verdict == Verdict.BLOCK
        assert category == Category.VIOLATION

    def test_c_is_restriction(self):
        verdict, category = VERDICT_MAP["C"]
        assert verdict == Verdict.BLOCK
        assert category == Category.RESTRICTION

    def test_d_is_review(self):
        verdict, category = VERDICT_MAP["D"]
        assert verdict == Verdict.REVIEW
        assert category == Category.UNCERTAIN
