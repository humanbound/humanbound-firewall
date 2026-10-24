# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The commands run the way a person runs them (python -m ...), and every evals file keeps its
license header: a broken entry point or a lost header fails here, not after a push."""

import subprocess
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).parent.parent


@pytest.mark.parametrize("module", ["evals.run", "evals.score", "evals.compare"])
def test_the_command_starts(module):
    out = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        cwd=EVALS.parent,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


def test_every_python_file_has_the_license_header():
    missing = [
        str(p.relative_to(EVALS))
        for p in sorted(EVALS.rglob("*.py"))
        if not p.read_text(encoding="utf-8").startswith("# SPDX-License-Identifier: Apache-2.0")
    ]
    assert missing == []
