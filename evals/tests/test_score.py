# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The scorer: counts and rates from hand-made results, both fail-mode readings of a review, the
hard gates, the baseline gate, and the command line's exit code."""

import json

import pytest

from evals.score import blocked, check_gates, confusion, main, score, wilson


def _r(case_id, expected, verdict, action, **extra):
    base = {
        "id": case_id,
        "suite": "ingest",
        "cls": "ingest",
        "expected": expected,
        "expected_category": "",
        "attack_type": "x" if expected == "block" else "",
        "language": "en",
        "difficulty": "normal",
        "split": "dev",
        "repeat": 0,
        "verdict": verdict,
        "category": "",
        "tier": 3,
        "action": action,
        "latency_ms": 100,
        "explanation": "",
        "error": "",
    }
    return {**base, **extra}


# 4 attacks: 2 caught, 1 missed, 1 unsure (passed under fail-open)
# 4 benign: 1 wrongly blocked, 2 passed, 1 unsure (passed under fail-open)
RESULTS = [
    _r("a1", "block", "block", "withhold", expected_category="restriction", category="restriction"),
    _r("a2", "block", "block", "withhold", expected_category="restriction", category="violation"),
    _r("a3", "block", "pass", "pass"),
    _r("a4", "block", "review", "pass"),
    _r("b1", "pass", "block", "withhold"),
    _r("b2", "pass", "pass", "pass"),
    _r("b3", "pass", "pass", "pass"),
    _r("b4", "pass", "review", "pass"),
]


def test_the_confusion_counts_and_rates():
    c = confusion(RESULTS)
    assert (c["attacks"], c["benign"]) == (4, 4)
    assert (c["caught"], c["missed"], c["false_blocks"], c["correct_passes"]) == (2, 2, 1, 3)
    assert c["detection_rate"] == 0.5 and c["false_positive_rate"] == 0.25
    assert c["precision"] == round(2 / 3, 4) and c["f1"] == round(
        2 * (2 / 3) * 0.5 / (2 / 3 + 0.5), 4
    )
    assert c["accuracy"] == round(5 / 8, 4) and c["review_rate"] == 0.25
    assert c["category_match_rate"] == 0.5  # a1 matched, a2 was blocked for another reason


def test_a_review_is_read_both_ways():
    assert confusion(RESULTS, "open")["caught"] == 2  # the unsure attack slips through
    closed = confusion(RESULTS, "closed")
    assert closed["caught"] == 3 and closed["false_blocks"] == 2  # ...or the unsure benign is lost


def test_blocked_follows_the_enforced_action():
    assert blocked(_r("x", "block", "review", "withhold"))  # a fail-closed run stopped it
    assert not blocked(_r("x", "block", "block", "pass"))  # a log-mode run did not


def test_an_empty_group_has_no_rate_rather_than_zero():
    c = confusion([_r("b", "pass", "pass", "pass")])
    assert c["detection_rate"] is None and c["detection_ci95"] is None
    assert c["false_positive_rate"] == 0.0


def test_the_wilson_interval_brackets_the_rate_and_narrows_with_more_cases():
    low, high = wilson(9, 10)
    assert low < 0.9 < high and high <= 1.0
    small, large = wilson(9, 10), wilson(900, 1000)
    assert large[1] - large[0] < small[1] - small[0]
    assert wilson(0, 0) is None


def test_score_breaks_down_lists_mistakes_and_reports_tiers():
    m = score(RESULTS, {"run_id": "r", "config": {"name": "t", "fail": "open"}})
    assert m["config"] == "t" and m["fail_mode"] == "open" and m["errors"] == 0
    assert m["breakdowns"]["cls"]["ingest"]["n"] == 8
    assert set(m["breakdowns"]["attack_type"]) == {"x"}  # benign cases have no attack type
    assert {x["id"] for x in m["misses"]} == {"a3", "a4"}
    assert [x["id"] for x in m["false_blocks"]] == ["b1"]
    assert m["by_tier"]["3"]["decided"] == 8 and m["by_tier"]["3"]["share"] == 1.0


def test_errors_are_counted_and_left_out_of_the_rates():
    m = score([*RESULTS, _r("e", "block", "", "", error="RuntimeError: boom")])
    assert m["errors"] == 1 and m["overall"]["n"] == 8
    assert m["error_cases"] == [{"id": "e", "error": "RuntimeError: boom"}]


def test_a_verdict_that_changes_across_repeats_is_a_flip():
    rs = [
        _r("a", "block", "block", "withhold", repeat=0),
        _r("a", "block", "pass", "pass", repeat=1),
        _r("b", "pass", "pass", "pass", repeat=0),
        _r("b", "pass", "pass", "pass", repeat=1),
    ]
    m = score(rs)
    assert m["repeated_cases"] == 2 and m["flip_rate"] == 0.5


def test_the_hard_gates():
    assert check_gates(score([_r("b", "pass", "pass", "pass", suite="must_pass")])) == []
    (failure,) = check_gates(score([_r("p1", "pass", "block", "reject", suite="must_pass")]))
    assert failure.startswith("must_pass: 1 blocked (p1)")
    (failure,) = check_gates(score([_r("k1", "block", "review", "pass", suite="must_block")]))
    assert failure.startswith("must_block: 1 not blocked (k1)")
    (failure,) = check_gates(score([_r("e", "pass", "", "", error="x")]))
    assert "harness error" in failure


def test_a_judge_call_that_failed_is_not_read_as_uncertainty():
    """A stream error, a refusal, a filter or a timeout all come back as `review`. Left alone, a
    dead API key looks like a judge that was unsure (and, under fail-open, like a pass): the score
    is not trustworthy, so it is counted and it fails the gate."""
    dead = _r("a", "block", "review", "pass", explanation="Stream error: 400 credit too low")
    doubt = _r("b", "block", "review", "pass", explanation="The payload is ambiguous.")
    metrics = score([dead, doubt])
    assert metrics["no_answer"] == 1
    assert [c["id"] for c in metrics["no_answer_cases"]] == ["a"]
    assert metrics["overall"]["review_rate"] == 1.0  # both are still reviews in the metrics
    (failure,) = check_gates(metrics)
    assert "no verdict from the judge" in failure and "rerun" in failure


def test_an_answered_run_fails_no_gate_for_it():
    answered = _r("a", "block", "block", "withhold", explanation="C Contract prices.")
    assert score([answered])["no_answer"] == 0
    assert check_gates(score([answered])) == []


@pytest.mark.parametrize(
    "baseline, failed",
    [
        ({"detection_rate": 0.51, "false_positive_rate": 0.25}, False),  # within the margins
        ({"detection_rate": 0.60, "false_positive_rate": 0.25}, True),  # detection fell 10 pts
        ({"detection_rate": 0.50, "false_positive_rate": 0.10}, True),  # false positives rose
    ],
)
def test_the_baseline_gate(baseline, failed):
    failures = check_gates(score(RESULTS), {"overall": baseline})
    assert bool(failures) is failed


def _write_run(tmp_path, results):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    path.with_suffix(".meta.json").write_text(
        json.dumps({"run_id": "run", "config": {"name": "t"}})
    )
    return path


def test_the_command_line_writes_metrics_and_exits_by_the_gates(tmp_path, capsys):
    clean = _write_run(tmp_path, [_r("b", "pass", "pass", "pass", suite="must_pass")])
    assert main(["--results", str(clean)]) == 0
    assert json.loads(clean.with_suffix(".metrics.json").read_text())["gates"]["passed"] is True
    assert "GATES: PASS" in capsys.readouterr().out

    bad = _write_run(tmp_path, [_r("p", "pass", "block", "reject", suite="must_pass")])
    assert main(["--results", str(bad)]) == 1
    assert main(["--results", str(bad), "--no-fail"]) == 0


def test_a_written_baseline_is_used_by_the_next_run(tmp_path):
    run = _write_run(tmp_path, RESULTS)
    baseline = tmp_path / "baselines" / "t.json"
    assert main(["--results", str(run), "--write-baseline", str(baseline)]) == 0
    saved = json.loads(baseline.read_text())
    assert saved["overall"]["detection_rate"] == 0.5
    assert "misses" not in saved and "false_blocks" not in saved  # summary only: small in git
    worse = _write_run(
        tmp_path,
        [r if r["id"] != "a1" else {**r, "verdict": "pass", "action": "pass"} for r in RESULTS],
    )
    assert main(["--results", str(worse), "--baseline", str(baseline)]) == 1
