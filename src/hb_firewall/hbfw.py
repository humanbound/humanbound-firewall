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

DEFAULT_TIER1_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy_coverage(classify_fn, permitted, restricted):
    results = {"restricted": [], "permitted": []}
    for intent in (restricted or []):
        r = classify_fn([{"u": intent.lower(), "a": ""}])
        results["restricted"].append({
            "intent": intent, "decision": r["decision"],
            "covered": r["decision"] != "ALLOW"})
    for intent in (permitted or []):
        r = classify_fn([{"u": intent.lower(), "a": ""}])
        results["permitted"].append({
            "intent": intent, "decision": r["decision"],
            "correct": r["decision"] == "ALLOW"})

    rc = sum(1 for r in results["restricted"] if r["covered"])
    pc = sum(1 for r in results["permitted"] if r["correct"])
    return {
        "restricted_coverage": round(rc / len(results["restricted"]), 4) if results["restricted"] else None,
        "restricted_covered": rc, "restricted_total": len(results["restricted"]),
        "permitted_coverage": round(pc / len(results["permitted"]), 4) if results["permitted"] else None,
        "permitted_correct": pc, "permitted_total": len(results["permitted"]),
        "gaps": [r for r in results["restricted"] if not r["covered"]],
        "false_blocks": [r for r in results["permitted"] if not r["correct"]],
    }


def _run_attack_benchmark(classify_fn, dataset_name, split, text_field,
                           label_field, attack_label, tier1_model=None):
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        return {"dataset": dataset_name, "error": str(e)[:200]}

    tier1_pipe = None
    if tier1_model:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import pipeline as hf_pipeline
        tier1_pipe = hf_pipeline("text-classification", model=tier1_model,
                                  truncation=True, max_length=512)

    y_true, t1_dec, t2_dec, combined = [], [], [], []
    for s in ds:
        text = s.get(text_field, "")
        if not text or not text.strip():
            continue
        is_attack = 1 if s.get(label_field) == attack_label else 0
        y_true.append(is_attack)

        t1 = None
        if tier1_pipe:
            r = tier1_pipe(text[:512])[0]
            if r["label"] in ("INJECTION", "LABEL_1", "positive", "1") and r["score"] > 0.5:
                t1 = "BLOCK"
        t1_dec.append(t1)

        t2 = classify_fn([{"u": text, "a": ""}])
        t2_dec.append(t2["decision"])
        combined.append(t1 if t1 == "BLOCK" else t2["decision"])

    total = len(y_true)
    atk = sum(1 for t in y_true if t == 1)
    result = {
        "dataset": dataset_name, "total": total,
        "attacks": {
            "total": atk,
            "blocked": sum(1 for t, d in zip(y_true, combined) if t == 1 and d == "BLOCK"),
            "escalated": sum(1 for t, d in zip(y_true, combined) if t == 1 and d == "ESCALATE"),
            "allowed": sum(1 for t, d in zip(y_true, combined) if t == 1 and d == "ALLOW"),
        },
        "benign": {
            "total": total - atk,
            "allowed": sum(1 for t, d in zip(y_true, combined) if t == 0 and d == "ALLOW"),
            "blocked": sum(1 for t, d in zip(y_true, combined) if t == 0 and d == "BLOCK"),
        },
    }
    if tier1_pipe:
        result["tier1"] = {
            "attacks_blocked": sum(1 for t, d in zip(y_true, t1_dec) if t == 1 and d == "BLOCK"),
            "benign_blocked": sum(1 for t, d in zip(y_true, t1_dec) if t == 0 and d == "BLOCK"),
        }
        t2_seen = sum(1 for t, d in zip(y_true, t1_dec) if t == 1 and d != "BLOCK")
        t2_caught = sum(1 for t, d1, d2 in zip(y_true, t1_dec, t2_dec)
                        if t == 1 and d1 != "BLOCK" and d2 == "BLOCK")
        result["tier2"] = {"attacks_seen": t2_seen, "attacks_blocked": t2_caught}
    return result


def _run_benign_benchmark(classify_fn, dataset_name, split, text_field):
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        return {"dataset": dataset_name, "type": "benign", "error": str(e)[:200]}

    decisions = []
    for s in ds:
        text = s.get(text_field, "")
        if not text or not text.strip():
            continue
        decisions.append(classify_fn([{"u": text, "a": ""}])["decision"])

    total = len(decisions)
    return {
        "dataset": dataset_name, "type": "benign", "total": total,
        "allowed": sum(1 for d in decisions if d == "ALLOW"),
        "escalated": sum(1 for d in decisions if d == "ESCALATE"),
        "blocked": sum(1 for d in decisions if d == "BLOCK"),
    }


def evaluate_benchmarks(classify_fn, run_tier1=True, benign_dataset=None):
    tier1 = DEFAULT_TIER1_MODEL if run_tier1 else None
    benchmarks = []
    for ds_name, label in [("deepset/prompt-injections", 1),
                            ("neuralchemy/Prompt-injection-dataset", 1)]:
        benchmarks.append(_run_attack_benchmark(
            classify_fn, ds_name, "test", "text", "label", label, tier1))
    if benign_dataset:
        benchmarks.append(_run_benign_benchmark(
            classify_fn, benign_dataset["dataset"],
            benign_dataset.get("split", "test"),
            benign_dataset.get("text_field", "text")))
    return benchmarks


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

        return {
            "attack_texts": attack_texts,
            "benign_texts": benign_texts,
            "restricted_texts": restricted_texts,
            "permitted_texts": permitted_texts,
            "has_qa": self._has_qa,
            "stats": {
                "attack_samples": len(attack_texts),
                "benign_samples": len(benign_texts),
                "restricted_intents": len(restricted_texts),
                "permitted_intents": len(permitted_texts),
            },
        }

    def train(self, data, permitted_intents=None, restricted_intents=None,
              benign_dataset=None):
        context = {
            "permitted_intents": permitted_intents or [],
            "restricted_intents": restricted_intents or [],
        }

        if len(data["attack_texts"]) >= 10:
            self.clf_attack.train(data["attack_texts"], context=context)

        benign_all = data["benign_texts"] + data["permitted_texts"]
        if len(benign_all) >= 10:
            self.clf_benign.train(benign_all, context=context)

        self._performance = {
            "stats": data.get("stats", {}),
            "has_qa_data": data.get("has_qa", False),
        }

        if permitted_intents or restricted_intents:
            self._performance["policy_coverage"] = evaluate_policy_coverage(
                self.classify, permitted_intents, restricted_intents)

        self._performance["benchmarks"] = evaluate_benchmarks(
            self.classify, benign_dataset=benign_dataset)

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
                "model": getattr(self.clf_attack, '_model_path', ''),
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
