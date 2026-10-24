# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Compare the labels with what one or more judges said, and list what a human should review.

    python -m evals.compare --results evals/results/<judge-openai-run>.jsonl
    python -m evals.compare --results <judge-openai-run>.jsonl <judge-claude-run>.jsonl

A disagreement is not proof that a label is wrong: the judge may be the one that is wrong, and that
is what an eval is for. The list only orders the cases a person should read, most suspicious first:

  label?     every judge, in every repeat, said the opposite of the label: the label may be wrong
  unstable   a judge changed its answer across repeats (or was unsure): the case may be unclear
  split      the judges disagree with each other: more likely a weakness of one judge
  category   right verdict, wrong reason, in every judge: check expected_category
  no answer  every judge failed to answer (refused, filtered, timed out): says nothing about the label
  tier 0     decided by Tier 0 before any judge saw it: says nothing about the label
  error      a harness error: rerun before reading anything into it

A failed answer (a refusal, a provider's content filter, a timeout, a reply with no verdict) is
not an opinion: it is set aside before comparing, and counted per judge in the report.

Never change a label only because a judge disagrees; change it when, having read the case, the
label is wrong. Otherwise the dataset learns to agree with one model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .score import load_results, no_answer

GROUPS = ("label?", "unstable", "split", "category", "no answer", "tier 0", "error")
TITLES = {
    "label?": "LABEL MAY BE WRONG: every judge, every repeat, said the opposite",
    "unstable": "UNSTABLE: a judge changed its answer across repeats, or was unsure",
    "split": "SPLIT: the judges disagree with each other",
    "category": "WRONG REASON: right verdict, different category in every judge",
    "no answer": "NO ANSWER: every judge failed (refused, filtered, timed out); rerun or skip",
    "tier 0": "DECIDED BY TIER 0: no judge saw these; they say nothing about the label",
    "error": "HARNESS ERROR: rerun these",
}


def load_runs(paths: list[str | Path]) -> dict[str, list[dict]]:
    """Results per run, keyed by the run's config name (or file name when there is no meta)."""
    runs: dict[str, list[dict]] = {}
    for path in paths:
        results, meta = load_results(path)
        name = (meta.get("config") or {}).get("name") or Path(path).stem
        if name in runs:
            name = f"{name} ({Path(path).stem})"
        runs[name] = results
    return runs


# The explanations the firewall gives when the judge produced no verdict at all. `no_answer` and
# the strings live in score.py: the scorer needs them too, to tell a failed call from real doubt.


def _majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else ""


def classify(case: dict[str, Any], by_run: dict[str, list[dict]]) -> str:
    """The review group of one case, given its results in every run ("" when all is well)."""
    if any(r.get("error") for rs in by_run.values() for r in rs):
        return "error"
    if any(r.get("tier") == 0 for rs in by_run.values() for r in rs):
        return "tier 0"
    expected = case["expected"]
    opposite = "pass" if expected == "block" else "block"
    verdicts = {run: [r["verdict"] for r in rs] for run, rs in by_run.items()}
    if all(vs and all(v == opposite for v in vs) for vs in verdicts.values()):
        return "label?"
    if any(len(set(vs)) > 1 or "review" in vs for vs in verdicts.values()):
        return "unstable"
    if len({vs[0] for vs in verdicts.values()}) > 1:
        return "split"
    wanted = case.get("expected_category")
    if expected == "block" and wanted:
        cats = [_majority([r.get("category", "") for r in rs]) for rs in by_run.values()]
        if all(c and c != wanted for c in cats):
            return "category"
    return ""


def compare(runs: dict[str, list[dict]]) -> dict[str, Any]:
    """Every case that needs a look, grouped and ordered, plus the counts."""
    per_case: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    first: dict[str, dict] = {}
    for run, results in runs.items():
        for r in results:
            per_case[r["id"]][run].append(r)
            first.setdefault(r["id"], r)

    groups: dict[str, list[dict]] = {g: [] for g in GROUPS}
    agreed = 0
    for case_id in sorted(per_case):
        case, by_run = first[case_id], per_case[case_id]
        answered = {run: [r for r in rs if not no_answer(r)] for run, rs in by_run.items()}
        answered = {run: rs for run, rs in answered.items() if rs}
        if any(r.get("error") or r.get("tier") == 0 for rs in by_run.values() for r in rs):
            group = classify(case, by_run)
        elif not answered:
            group = "no answer"
        else:
            group = classify(case, answered)
        if not group:
            agreed += 1
            continue
        groups[group].append(
            {
                "id": case_id,
                "suite": case.get("suite", ""),
                "cls": case.get("cls", ""),
                "expected": case["expected"],
                "expected_category": case.get("expected_category", ""),
                "judges": {
                    run: {
                        "verdicts": dict(
                            Counter(
                                "no-answer" if no_answer(r) else (r["verdict"] or "error")
                                for r in rs
                            )
                        ),
                        "category": _majority([r.get("category", "") for r in rs]),
                        "tier": rs[0].get("tier"),
                        "explanation": next(
                            (r["explanation"] for r in rs if r.get("explanation")), ""
                        ),
                    }
                    for run, rs in by_run.items()
                },
            }
        )
    return {
        "runs": list(runs),
        "cases": len(per_case),
        "agreed": agreed,
        "counts": {g: len(groups[g]) for g in GROUPS},
        "no_answer_per_judge": {
            run: sum(1 for r in rs if no_answer(r)) for run, rs in runs.items()
        },
        "groups": groups,
    }


def render(report: dict[str, Any], explain_chars: int = 160) -> str:
    """The review list as plain text, most suspicious group first."""
    lines = [
        f"judges: {', '.join(report['runs'])}   cases: {report['cases']}   "
        f"agree with the label: {report['agreed']}",
        "  " + "   ".join(f"{g}: {n}" for g, n in report["counts"].items()),
        "  failed answers set aside (refused, filtered, timed out): "
        + ", ".join(f"{run} {n}" for run, n in report["no_answer_per_judge"].items()),
    ]
    for group in GROUPS:
        items = report["groups"][group]
        if not items:
            continue
        lines += ["", f"== {TITLES[group]} ({len(items)})"]
        for item in items:
            label = item["expected"] + (
                f"/{item['expected_category']}" if item["expected_category"] else ""
            )
            lines.append(f"  {item['id']}  [{item['suite']}, {item['cls']}]  label={label}")
            for run, j in item["judges"].items():
                verdicts = " ".join(f"{v}x{n}" for v, n in sorted(j["verdicts"].items()))
                category = f" category={j['category']}" if j["category"] else ""
                lines.append(f"      {run}: {verdicts}{category} tier={j['tier']}")
                if j["explanation"]:
                    lines.append(f'        "{j["explanation"][:explain_chars]}"')
    if not any(report["counts"].values()):
        lines += ["", "Every judge agreed with every label."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.compare", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--results", nargs="+", required=True, help="one results file per judge")
    parser.add_argument("--json", help="also write the review list here as JSON")
    args = parser.parse_args(argv)

    report = compare(load_runs(args.results))
    print(render(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nreview list -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
