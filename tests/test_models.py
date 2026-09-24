# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for data models."""

from humanbound_firewall.models import VERDICT_MAP, Category, EvalResult, Verdict


class TestEvalResult:
    def test_blocked_and_passed_properties(self):
        r = EvalResult(verdict=Verdict.BLOCK, category=Category.VIOLATION)
        assert r.blocked is True
        assert r.passed is False

        r = EvalResult(verdict=Verdict.PASS)
        assert r.passed is True
        assert r.blocked is False

        # REVIEW is neither blocked nor passed
        r = EvalResult(verdict=Verdict.REVIEW)
        assert r.blocked is False
        assert r.passed is False


class TestVerdictMap:
    def test_every_letter_maps_to_its_verdict_and_category(self):
        assert VERDICT_MAP == {
            "P": (Verdict.PASS, Category.NONE),
            "A": (Verdict.BLOCK, Category.OFF_TOPIC),
            "B": (Verdict.BLOCK, Category.VIOLATION),
            "C": (Verdict.BLOCK, Category.RESTRICTION),
            "D": (Verdict.REVIEW, Category.UNCERTAIN),
        }
