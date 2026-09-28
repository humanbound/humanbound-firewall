# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""firewall.adapt_to(framework): the adapter registry. Lazy import, no per-framework extras."""

import importlib
import subprocess
import sys

import pytest

from humanbound_firewall.firewall import Firewall
from humanbound_firewall.models import AgentConfig


def test_unknown_frameworks_are_rejected_naming_the_known_ones():
    with pytest.raises(ValueError) as e:
        Firewall(AgentConfig(business_scope="x")).adapt_to("django")
    assert "langchain" in str(e.value)


def test_adapt_to_returns_a_fresh_adapter_bound_to_this_firewall():
    pytest.importorskip("langchain")
    from humanbound_firewall.integrations.langchain import HumanboundFirewallMiddleware

    fw = Firewall(AgentConfig(business_scope="x"))
    a, b = fw.adapt_to("langchain"), fw.adapt_to("langchain")
    assert isinstance(a, HumanboundFirewallMiddleware) and a.firewall is fw
    assert a is not b


def test_a_missing_framework_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", None)
    monkeypatch.delitem(sys.modules, "humanbound_firewall.integrations.langchain", raising=False)
    with pytest.raises(ImportError) as e:
        Firewall(AgentConfig(business_scope="x")).adapt_to("langchain")
    assert "pip install langchain" in str(e.value)
    importlib.invalidate_caches()


def test_importing_the_package_does_not_import_any_framework():
    probe = (
        "import sys, humanbound_firewall; from humanbound_firewall import Firewall; "
        "print(sorted(n for n in sys.modules if n.startswith(('langchain', 'langgraph'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
