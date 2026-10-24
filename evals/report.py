# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Render metrics as a markdown report: summary table, per-class breakdown, misses and false blocks."""

from __future__ import annotations


def render(metrics: dict) -> str:
    raise NotImplementedError
