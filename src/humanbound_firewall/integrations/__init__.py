# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Framework adapters, reached through `Firewall.adapt_to(framework)`.

One library, no per-framework extras: an adapter is imported lazily, on first use, and the framework
itself is the application's dependency. Without an adapter, the manual tier — `firewall.inspect()`
at each boundary — is the same contract.
"""

from __future__ import annotations

import importlib

ADAPTERS = {
    "langchain": ("humanbound_firewall.integrations.langchain", "HumanboundFirewallMiddleware"),
}


def load_adapter(framework: str):
    """The adapter class for a framework; ValueError if unknown, ImportError if not installed."""
    if framework not in ADAPTERS:
        raise ValueError(
            f"No adapter for '{framework}'. Known adapters: {', '.join(ADAPTERS)}. "
            "Without one, call firewall.inspect(payload, cls=...) at each boundary yourself."
        )
    module_name, attr = ADAPTERS[framework]
    return getattr(importlib.import_module(module_name), attr)
