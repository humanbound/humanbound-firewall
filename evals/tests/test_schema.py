# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The case loader: the shipped datasets load, and a broken case fails with its file and line."""

import json

import pytest

from evals.schema import POLICIES_DIR, CaseError, load_cases

GOOD = {"id": "x", "cls": "ingest", "payload": "p", "policy": "banking.yaml", "expected": "pass"}


def _write(path, *lines):
    path.write_text("".join((ln if isinstance(ln, str) else json.dumps(ln)) + "\n" for ln in lines))
    return path


def test_the_shipped_datasets_load_and_every_case_knows_its_suite():
    cases = load_cases()
    assert cases
    assert {c.suite for c in cases} >= {"request", "ingest", "recall", "must_pass", "must_block"}
    assert len({c.id for c in cases}) == len(cases)


def test_ingest_cases_are_never_filed_under_integrity():
    """Integrity is a recall-only finding: the ingest judge's letters are A (off_topic), B
    (violation) and C (restriction), with no integrity step at all, so an ingest case labelled
    integrity can never match. (The recall judge's blocks are all filed as integrity, but a
    Tier 0 block — an invisible-character payload — keeps the tier-0 category, so a recall case
    may legitimately carry a non-integrity category.)"""
    for case in load_cases():
        if case.cls == "ingest":
            assert case.expected_category != "integrity", case.id


def test_the_shipped_policies_load():
    """Every policy parses, not only the ones a case happens to reference: a policy that cannot
    load would otherwise sit in the repo unnoticed until someone wrote a case for it."""
    from humanbound_firewall.config import load_config

    paths = sorted(POLICIES_DIR.glob("*.yaml"))
    assert paths
    for path in paths:
        assert load_config(path).name


def test_a_minimal_case_gets_the_defaults(tmp_path):
    (case,) = load_cases(_write(tmp_path / "t.jsonl", GOOD))
    assert (case.window, case.boundary, case.language, case.split) == ([], None, "en", "dev")
    assert case.suite == "t"


@pytest.mark.parametrize(
    "case, message",
    [
        ({**GOOD, "atack_type": "x"}, "unknown field"),
        ({k: v for k, v in GOOD.items() if k != "policy"}, "missing required"),
        ({**GOOD, "cls": "pigeon"}, "cls"),
        ({**GOOD, "expected": "maybe"}, "expected"),
        ({**GOOD, "expected": "block"}, "attack_type"),
        ({**GOOD, "expected_category": "restriction"}, "pass case"),
        ({**GOOD, "expected": "block", "attack_type": "a", "expected_category": "bad"}, "category"),
        ({**GOOD, "policy": "nope.yaml"}, "not found"),
        ({**GOOD, "window": [{"role": "system", "content": "x"}]}, "role"),
        ({**GOOD, "boundary": {"kind": "tools"}}, "name"),
        ({**GOOD, "suite": "x"}, "unknown field"),
        ({**GOOD, "id": "  "}, "empty"),
    ],
)
def test_a_broken_case_is_rejected_with_its_file_and_line(tmp_path, case, message):
    path = _write(tmp_path / "t.jsonl", GOOD | {"id": "first"}, case)
    with pytest.raises(CaseError, match=message) as e:
        load_cases(path)
    assert "t.jsonl:2" in str(e.value)


def test_invalid_json_names_the_line(tmp_path):
    with pytest.raises(CaseError, match="t.jsonl:2: invalid JSON"):
        load_cases(_write(tmp_path / "t.jsonl", GOOD, "{broken"))


def test_an_id_used_twice_is_rejected_even_across_files(tmp_path):
    _write(tmp_path / "a.jsonl", GOOD)
    _write(tmp_path / "b.jsonl", GOOD)
    with pytest.raises(CaseError, match="duplicate id 'x'"):
        load_cases(tmp_path)


def test_blank_lines_are_skipped(tmp_path):
    assert len(load_cases(_write(tmp_path / "t.jsonl", GOOD, "", "   "))) == 1


# --- a tool's output has the class its policy gives it -------------------------------------------

SHOP = {"cls": "ingest", "payload": "p", "policy": "shopassist.yaml", "expected": "pass"}


@pytest.mark.parametrize(
    "case",
    [
        {**SHOP, "id": "a", "boundary": {"name": "fetch_url", "kind": "tools"}},  # default: ingest
        {**SHOP, "id": "b", "cls": "recall", "boundary": {"name": "lookup_order"}},  # declared
        {**SHOP, "id": "c", "cls": "recall", "boundary": {"name": "notes", "kind": "memory"}},
    ],
)
def test_a_case_whose_class_matches_its_tool_loads(tmp_path, case):
    assert load_cases(_write(tmp_path / "t.jsonl", case))


@pytest.mark.parametrize(
    "case, message",
    [
        # the policy lists lookup_order under tools.recall: its output is our own record
        (
            {**SHOP, "boundary": {"name": "lookup_order", "kind": "tools"}},
            "is recall in shopassist",
        ),
        # an undeclared tool is ingest, never recall
        ({**SHOP, "cls": "recall", "boundary": {"name": "fetch_url"}}, "is ingest in shopassist"),
        # a tool's output is never a principal's request
        ({**SHOP, "cls": "request", "boundary": {"name": "fetch_url"}}, "never a request"),
    ],
)
def test_a_case_whose_class_disagrees_with_its_policy_is_rejected(tmp_path, case, message):
    with pytest.raises(CaseError, match=message):
        load_cases(_write(tmp_path / "t.jsonl", {**case, "id": "x"}))
