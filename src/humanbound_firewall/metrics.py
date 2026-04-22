# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Evaluation metrics tracker."""

import threading


class Metrics:
    """Thread-safe metrics tracker for firewall evaluations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total = 0
        self._passed = 0
        self._blocked = 0
        self._review = 0
        self._errors = 0
        self._by_category: dict[str, int] = {}
        self._latencies: list[int] = []

    @property
    def total_evaluations(self) -> int:
        return self._total

    @property
    def passed(self) -> int:
        return self._passed

    @property
    def blocked(self) -> int:
        return self._blocked

    @property
    def review(self) -> int:
        return self._review

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def by_category(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_category)

    @property
    def avg_latency_ms(self) -> int:
        with self._lock:
            if not self._latencies:
                return 0
            return round(sum(self._latencies) / len(self._latencies))

    @property
    def p99_latency_ms(self) -> int:
        with self._lock:
            if not self._latencies:
                return 0
            sorted_l = sorted(self._latencies)
            idx = int(len(sorted_l) * 0.99)
            return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def block_rate(self) -> float:
        if self._total == 0:
            return 0.0
        return round(self._blocked / self._total, 4)

    def record(self, verdict: str, category: str, latency_ms: int):
        """Record an evaluation result."""
        with self._lock:
            self._total += 1
            self._latencies.append(latency_ms)

            if verdict == "pass":
                self._passed += 1
            elif verdict == "block":
                self._blocked += 1
                if category:
                    self._by_category[category] = self._by_category.get(category, 0) + 1
            elif verdict == "review":
                self._review += 1

            # Keep only last 10,000 latencies to bound memory
            if len(self._latencies) > 10000:
                self._latencies = self._latencies[-5000:]

    def record_error(self):
        """Record an evaluation error."""
        with self._lock:
            self._total += 1
            self._errors += 1

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._total = 0
            self._passed = 0
            self._blocked = 0
            self._review = 0
            self._errors = 0
            self._by_category.clear()
            self._latencies.clear()

    def to_dict(self) -> dict:
        """Export metrics as a dictionary."""
        return {
            "total_evaluations": self._total,
            "passed": self._passed,
            "blocked": self._blocked,
            "review": self._review,
            "errors": self._errors,
            "block_rate": self.block_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "by_category": self.by_category,
        }
