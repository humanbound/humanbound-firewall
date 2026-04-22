# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Tests for metrics tracker."""

import pytest

from humanbound_firewall.metrics import Metrics


class TestMetrics:
    def test_initial_state(self):
        m = Metrics()
        assert m.total_evaluations == 0
        assert m.passed == 0
        assert m.blocked == 0
        assert m.avg_latency_ms == 0

    def test_record_pass(self):
        m = Metrics()
        m.record("pass", "", 100)
        assert m.total_evaluations == 1
        assert m.passed == 1
        assert m.blocked == 0

    def test_record_block(self):
        m = Metrics()
        m.record("block", "violation", 200)
        assert m.blocked == 1
        assert m.by_category == {"violation": 1}

    def test_block_rate(self):
        m = Metrics()
        m.record("pass", "", 100)
        m.record("pass", "", 100)
        m.record("block", "violation", 200)
        assert m.block_rate == pytest.approx(0.3333, abs=0.01)

    def test_avg_latency(self):
        m = Metrics()
        m.record("pass", "", 100)
        m.record("pass", "", 200)
        m.record("pass", "", 300)
        assert m.avg_latency_ms == 200

    def test_reset(self):
        m = Metrics()
        m.record("block", "violation", 100)
        m.reset()
        assert m.total_evaluations == 0
        assert m.blocked == 0
        assert m.by_category == {}

    def test_to_dict(self):
        m = Metrics()
        m.record("pass", "", 100)
        d = m.to_dict()
        assert d["total_evaluations"] == 1
        assert d["passed"] == 1
        assert "avg_latency_ms" in d

    def test_error_tracking(self):
        m = Metrics()
        m.record_error()
        assert m.errors == 1
        assert m.total_evaluations == 1
