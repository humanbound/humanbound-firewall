"""Tier 2 orchestrator for hb-firewall.

Handles data extraction, training coordination, evaluation, and serialization.
The actual ML model is injected — provide a Python file with a AgentClassifier class:

    class AgentClassifier:
        def __init__(self, name): ...
        def train(self, texts, context=None): ...
        def predict(self, text, context=""): ...  # → (bool, float)
        def export_weights(self): ...             # → dict
        def load_weights(self, weights): ...
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import Optional

MAX_TURNS = 20


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_class(path: str):
    """Load AgentClassifier class from a Python file or module.

    Accepts:
        "./detectors/my_model.py"          — file path
        "my_package.detectors:MyDetector"  — module:class (optional)
    """
    import importlib
    import importlib.util

    if ":" in path:
        module_path, class_name = path.rsplit(":", 1)
        if module_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location("_detector", module_path)
        else:
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
    else:
        class_name = "AgentClassifier"
        spec = importlib.util.spec_from_file_location("_detector", path)

    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load '{path}'. File not found or not a valid Python module.")

    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except FileNotFoundError:
        raise ValueError(f"Cannot load '{path}'. File not found.")

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ValueError(f"'{class_name}' not found in '{path}'")

    for method in ("train", "predict", "export_weights", "load_weights"):
        if not callable(getattr(cls, method, None)):
            raise ValueError(f"'{class_name}' missing required method: {method}")

    return cls


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def format_last_n_turns(conversation: list[dict], n: int = 5) -> str:
    cap = min(n, MAX_TURNS)
    recent = conversation[-cap:]
    parts = []
    for turn in recent:
        a = turn.get("a") or turn.get("assistant") or ""
        u = turn.get("u") or turn.get("user") or ""
        if a:
            parts.append(f"Agent: {a}")
        if u:
            parts.append(f"User: {u}")
    return "\n".join(parts)


def _extract_turns(logs, category_filter, result_filter):
    filtered = [l for l in logs
                if category_filter(l.get("test_category") or "")
                and result_filter(l.get("result") or "")]
    texts = []
    for log in filtered:
        conv = log.get("conversation", [])
        if not conv:
            continue
        n = len(conv)
        start = max(0, n - MAX_TURNS)
        for i in range(start, n):
            user_msg = conv[i].get("u") or conv[i].get("user") or ""
            if not user_msg.strip() or len(user_msg.strip()) < 10:
                continue
            ctx_start = max(0, i - 2)
            parts = []
            for j in range(ctx_start, i + 1):
                t = conv[j]
                a = t.get("a") or t.get("assistant") or ""
                u = t.get("u") or t.get("user") or ""
                if a:
                    parts.append(f"Agent: {a}")
                if u:
                    parts.append(f"User: {u}")
            texts.append("\n".join(parts))
    return texts


def extract_adversarial_turns(logs):
    return _extract_turns(logs,
                          category_filter=lambda c: "adversarial" in c,
                          result_filter=lambda r: r == "fail")


def extract_qa_texts(logs):
    return _extract_turns(logs,
                          category_filter=lambda c: "adversarial" not in c,
                          result_filter=lambda r: r == "pass")


# ---------------------------------------------------------------------------
# Curated data selection
# ---------------------------------------------------------------------------

CURATED_TARGET = 60


def _allocate_stratified(buckets, target):
    """Allocate target budget across buckets preserving distribution.

    Guarantees at least 1 per bucket (if budget allows).
    Remaining budget distributed proportionally.
    """
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return {}

    n_cats = len(buckets)
    allocation = {}

    # Phase 1: guarantee 1 per category
    guaranteed = min(n_cats, target)
    for cat in buckets:
        allocation[cat] = 1
    remaining = target - guaranteed

    # Phase 2: distribute remaining proportionally
    if remaining > 0:
        for cat in buckets:
            share = round(len(buckets[cat]) / total * remaining)
            allocation[cat] += share

    # Cap to available and budget
    total_alloc = sum(allocation.values())
    if total_alloc > target:
        # Trim largest buckets first
        for cat in sorted(allocation, key=lambda c: allocation[c], reverse=True):
            excess = total_alloc - target
            if excess <= 0:
                break
            trim = min(excess, allocation[cat] - 1)
            allocation[cat] -= trim
            total_alloc -= trim

    for cat in allocation:
        allocation[cat] = min(allocation[cat], len(buckets[cat]))

    return allocation


def _curate_attack_logs(logs, target=CURATED_TARGET):
    """Select diverse attack examples stratified by fail_category.

    At least 1 per fail_category, then proportional distribution.
    Sorted by severity DESC within each category.
    Last 2-3 turns per conversation (attack payload).
    """
    from collections import defaultdict

    failed = [l for l in logs
              if "adversarial" in (l.get("test_category") or "")
              and l.get("result") == "fail"]

    if not failed:
        passed_adv = [l for l in logs
                      if "adversarial" in (l.get("test_category") or "")
                      and l.get("result") == "pass"]
        passed_adv.sort(key=lambda l: l.get("confidence", 0), reverse=True)
        return _extract_last_turns(passed_adv[:target])

    buckets = defaultdict(list)
    for log in failed:
        cat = log.get("fail_category") or "other"
        buckets[cat].append(log)

    for cat in buckets:
        buckets[cat].sort(key=lambda l: (l.get("severity", 0), l.get("confidence", 0)),
                          reverse=True)

    allocation = _allocate_stratified(buckets, target)

    selected = []
    for cat, quota in allocation.items():
        selected.extend(buckets[cat][:quota])

    return _extract_last_turns(selected)


def _curate_benign_logs(logs, target=CURATED_TARGET, fallback=None):
    """Select diverse benign examples stratified by gen_category.

    At least 1 per gen_category, then uniform distribution.
    Sorted by confidence DESC within each category.
    """
    from collections import defaultdict

    passed_qa = [l for l in logs
                 if "adversarial" not in (l.get("test_category") or "")
                 and l.get("result") == "pass"]

    if not passed_qa:
        return fallback[:target] if fallback else []

    buckets = defaultdict(list)
    for log in passed_qa:
        cat = log.get("gen_category") or "other"
        buckets[cat].append(log)

    for cat in buckets:
        buckets[cat].sort(key=lambda l: l.get("confidence", 0), reverse=True)

    allocation = _allocate_stratified(buckets, target)

    selected = []
    for cat, quota in allocation.items():
        selected.extend(buckets[cat][:quota])

    return _extract_qa_turns_from_logs(selected)


def _extract_last_turns(logs, n_last=3):
    """Extract last N turns from conversations with context."""
    texts = []
    for log in logs:
        conv = log.get("conversation", [])
        if not conv:
            continue
        start = max(0, len(conv) - n_last)
        for i in range(start, len(conv)):
            user_msg = conv[i].get("u") or conv[i].get("user") or ""
            if not user_msg.strip() or len(user_msg.strip()) < 10:
                continue
            ctx_start = max(0, i - 2)
            parts = []
            for j in range(ctx_start, i + 1):
                t = conv[j]
                a = t.get("a") or t.get("assistant") or ""
                u = t.get("u") or t.get("user") or ""
                if a:
                    parts.append(f"Agent: {a}")
                if u:
                    parts.append(f"User: {u}")
            texts.append("\n".join(parts))
    return texts


def _extract_qa_turns_from_logs(logs):
    """Extract individual turns from QA logs with context."""
    texts = []
    for log in logs:
        conv = log.get("conversation", [])
        for i, turn in enumerate(conv):
            user_msg = turn.get("u") or turn.get("user") or ""
            if not user_msg.strip() or len(user_msg.strip()) < 10:
                continue
            ctx_start = max(0, i - 2)
            parts = []
            for j in range(ctx_start, i + 1):
                t = conv[j]
                a = t.get("a") or t.get("assistant") or ""
                u = t.get("u") or t.get("user") or ""
                if a:
                    parts.append(f"Agent: {a}")
                if u:
                    parts.append(f"User: {u}")
            texts.append("\n".join(parts))
    return texts


# ---------------------------------------------------------------------------
# HBFW orchestrator
# ---------------------------------------------------------------------------

class HBFW:
    """Tier 2 orchestrator. Inject your model, the orchestrator does the rest."""

    def __init__(self, attack_detector, benign_detector):
        self.clf_attack = attack_detector
        self.clf_benign = benign_detector
        self._performance = None
        self._has_qa = False

    def prepare(self, logs, restricted_intents=None, permitted_intents=None):
        attack_texts = extract_adversarial_turns(logs)
        qa_texts = extract_qa_texts(logs)
        self._has_qa = len(qa_texts) > 0

        restricted_texts = [i.lower() for i in (restricted_intents or [])]
        permitted_texts = [i.lower() for i in (permitted_intents or [])]
        benign_texts = qa_texts if qa_texts else permitted_texts

        # Curated subsets: stratified, high-quality examples for few-shot training
        curated_attack = _curate_attack_logs(logs)
        curated_benign = _curate_benign_logs(logs, fallback=permitted_texts)

        return {
            "attack_texts": attack_texts,
            "benign_texts": benign_texts,
            "curated_attack": curated_attack,
            "curated_benign": curated_benign,
            "restricted_texts": restricted_texts,
            "permitted_texts": permitted_texts,
            "has_qa": self._has_qa,
            "stats": {
                "attack_samples": len(attack_texts),
                "benign_samples": len(benign_texts),
                "curated_attack": len(curated_attack),
                "curated_benign": len(curated_benign),
                "restricted_intents": len(restricted_texts),
                "permitted_intents": len(permitted_texts),
            },
        }

    def train(self, data, permitted_intents=None, restricted_intents=None):
        context = {
            "permitted_intents": permitted_intents or [],
            "restricted_intents": restricted_intents or [],
            "all_attack_texts": data["attack_texts"],
            "all_benign_texts": data["benign_texts"],
        }

        # Pass curated data if available, full data as fallback
        attack_train = data.get("curated_attack") or data["attack_texts"]
        benign_train = data.get("curated_benign") or (data["benign_texts"] + data["permitted_texts"])

        if len(attack_train) >= 5:
            self.clf_attack.train(attack_train, context=context)

        if len(benign_train) >= 5:
            self.clf_benign.train(benign_train, context=context)

        self._performance = {
            "stats": data.get("stats", {}),
            "has_qa_data": data.get("has_qa", False),
        }

        # Capture validation metrics from detectors (if they report them)
        if hasattr(self.clf_attack, "metrics") and self.clf_attack.metrics:
            self._performance["validation"] = self.clf_attack.metrics

        return self._performance

    def classify(self, conversation: list[dict]) -> dict:
        ctx_text = format_last_n_turns(conversation, n=5)
        last = conversation[-1] if conversation else {}
        last_msg = last.get("u") or last.get("user") or ""
        turn_text = f"User: {last_msg}" if last_msg.strip() else ctx_text

        is_atk_ctx, s1 = self.clf_attack.predict(ctx_text)
        is_atk_turn, s2 = self.clf_attack.predict(turn_text)
        is_attack = is_atk_ctx or is_atk_turn
        attack_score = max(s1, s2)

        is_ben_ctx, _ = self.clf_benign.predict(ctx_text)
        is_ben_turn, _ = self.clf_benign.predict(turn_text)
        is_benign = is_ben_ctx and is_ben_turn

        if is_attack and not is_benign:
            return {"decision": "BLOCK", "reason": "attack",
                    "attack_probability": round(attack_score, 4), "tier": "2.1"}
        if is_benign and not is_attack:
            return {"decision": "ALLOW", "reason": "benign",
                    "attack_probability": round(attack_score, 4), "tier": "2.2"}

        reason = "conflicting" if (is_attack and is_benign) else "uncertain"
        return {"decision": "ESCALATE", "reason": reason,
                "attack_probability": round(attack_score, 4), "tier": "2"}

    def export(self):
        weights = {}
        weights.update(self.clf_attack.export_weights())
        weights.update(self.clf_benign.export_weights())
        return {
            "config": {
                "version": "2.0",
                "performance": self._performance or {},
            },
            "weights": weights,
        }

    def load(self, config, weights):
        self._performance = config.get("performance", {})
        self.clf_attack.load_weights(weights)
        self.clf_benign.load_weights(weights)


# ---------------------------------------------------------------------------
# .hbfw I/O
# ---------------------------------------------------------------------------

def save_hbfw(model_data, path):
    import numpy as np
    with tempfile.TemporaryDirectory() as tmpdir:
        cp = os.path.join(tmpdir, "config.json")
        wp = os.path.join(tmpdir, "weights.npz")
        with open(cp, "w") as f:
            json.dump(model_data["config"], f, indent=2)
        np.savez(wp, **model_data["weights"])
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(cp, "config.json")
            zf.write(wp, "weights.npz")


def load_hbfw(path):
    import numpy as np
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("config.json") as f:
            config = json.loads(f.read())
        with zf.open("weights.npz") as f:
            weights = dict(np.load(f, allow_pickle=True))
    return config, weights
