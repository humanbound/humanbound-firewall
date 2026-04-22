# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""humanbound-firewall: Multi-tier firewall for AI agents.

Tier 0:   Input sanitization
Tier 1:   Basic attack detection (pre-trained, single-turn)
Tier 2:   Agent-specific classification (trained, multi-turn, 3+ turns)
Tier 3:   LLM-as-a-judge (handles uncertain cases)

Designed for low cold-import cost: importing this module loads only stdlib and
pydantic-adjacent dependencies. Heavy components (Firewall, HBFW, LLM clients)
are loaded lazily on first attribute access.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

__version__ = "0.2.0"

from .models import VERDICT_MAP, AgentConfig, Category, EvalResult, Turn, Verdict

_LAZY_ATTRS = {
    "Firewall": ".firewall",
    "AttackDetector": ".firewall",
    "AttackDetectorEnsemble": ".firewall",
    "Provider": ".llm",
    "ProviderIntegration": ".llm",
    "ProviderName": ".llm",
    "get_llm_pinger": ".llm",
    "get_llm_streamer": ".llm",
    "HBFW": ".hbfw",
    "load_model_class": ".hbfw",
    "save_hbfw": ".hbfw",
    "load_hbfw": ".hbfw",
}


def __getattr__(name: str):
    """Lazy attribute loader (PEP 562).

    Defers importing heavy submodules until first attribute access so that
    bare `import humanbound_firewall` stays fast (< 200 ms target).
    """
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


if TYPE_CHECKING:
    # Make the lazy names visible to static type checkers and IDEs.
    from .firewall import AttackDetector, AttackDetectorEnsemble, Firewall
    from .hbfw import HBFW, load_hbfw, load_model_class, save_hbfw
    from .llm import (
        Provider,
        ProviderIntegration,
        ProviderName,
        get_llm_pinger,
        get_llm_streamer,
    )


__all__ = [
    "Firewall",
    "AttackDetector",
    "AttackDetectorEnsemble",
    "EvalResult",
    "AgentConfig",
    "Verdict",
    "Category",
    "Turn",
    "VERDICT_MAP",
    "Provider",
    "ProviderIntegration",
    "ProviderName",
    "get_llm_pinger",
    "get_llm_streamer",
    "HBFW",
    "load_model_class",
    "save_hbfw",
    "load_hbfw",
]

# Backwards-compat shim: legacy imports (`import hb_firewall`) and legacy
# pickled `.hbfw` models that reference the `hb_firewall.*` module path
# continue to work. Scope: 0.2.x only — will be removed in 0.3.
sys.modules.setdefault("hb_firewall", sys.modules[__name__])
