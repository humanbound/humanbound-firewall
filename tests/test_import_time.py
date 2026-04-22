# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Cold-import regression test.

Bare `import humanbound_firewall` must not pull in torch, transformers,
openai, anthropic, or google-generativeai. The lazy `__getattr__` in
`__init__.py` guarantees that — this test guards against regressions.
"""

from __future__ import annotations

import subprocess
import sys

COLD_IMPORT_BUDGET_MS = 500
FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "openai",
    "anthropic",
    "google.generativeai",
    "numpy",  # Only needed for .hbfw save/load
    "sentence_transformers",
    "setfit",
)


def test_cold_import_is_fast_and_minimal():
    """`import humanbound_firewall` should be fast and should NOT load heavy deps."""
    script = (
        "import time, sys\n"
        "t0 = time.perf_counter()\n"
        "import humanbound_firewall\n"
        "dt = (time.perf_counter() - t0) * 1000\n"
        "loaded = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') "
        f"for f in {FORBIDDEN_MODULES!r}))\n"
        "print(f'TIME={dt:.1f}')\n"
        "print(f'LOADED={loaded}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"import failed: {result.stderr}"

    lines = {
        line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.strip().splitlines()
    }
    elapsed_ms = float(lines["TIME"])
    loaded = lines["LOADED"]

    assert elapsed_ms < COLD_IMPORT_BUDGET_MS, (
        f"Cold import took {elapsed_ms:.1f} ms, budget is {COLD_IMPORT_BUDGET_MS} ms. "
        "Something heavy is being imported eagerly — check __init__.py."
    )
    assert loaded == "[]", (
        f"Heavy modules loaded on bare import: {loaded}. "
        "They must stay behind the lazy __getattr__ barrier."
    )


def test_legacy_hb_firewall_alias_resolves():
    """Legacy `import hb_firewall` must keep working in 0.2.x for .hbfw pickles."""
    script = (
        "import humanbound_firewall\n"
        "import hb_firewall\n"
        "assert hb_firewall is humanbound_firewall, 'alias not wired'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"alias test failed: {result.stderr}"
    assert "OK" in result.stdout
