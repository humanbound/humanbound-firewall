# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Run a dataset through one firewall configuration and write one result per case.

    python -m evals.run --config evals/configs/tier0.yaml
    python -m evals.run --config evals/configs/full-openai.yaml --split dev --workers 8

Writes evals/results/<run-id>.jsonl (one line per case and repeat) and <run-id>.meta.json (what was
run: config, dataset version, git commit, firewall version, timings).

The firewall is built only from the config's tiers: unlike Firewall.from_config(), nothing is picked
up from the environment, so a Tier 0 run can never call a paid LLM. Each case is screened with
Firewall.inspect(), the same call an adapter makes, in block mode so the action shows what would be
enforced. LLM judge replies are cached under evals/cache/<config>/, keyed on the model, the prompts
and the repeat number: re-running a config costs nothing, while repeats still sample the judge anew.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import humanbound_firewall
from humanbound_firewall.config import load_config
from humanbound_firewall.firewall import (
    AttackDetector,
    AttackDetectorEnsemble,
    Firewall,
    _extract_token,  # the firewall's own chunk reader, so the cache sees exactly what the judge saw
)
from humanbound_firewall.llm import Provider, ProviderIntegration, get_llm_streamer
from humanbound_firewall.models import AgentConfig

from .schema import DATASETS_DIR, POLICIES_DIR, Case, load_cases

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"
CACHE_DIR = EVALS_DIR / "cache"
EXPLANATION_TIMEOUT = 30  # seconds to wait for a streamed Tier 3 explanation

# Copied from the case into every result, so scoring never needs the dataset.
_CASE_FIELDS = (
    "id",
    "suite",
    "cls",
    "policy",
    "expected",
    "expected_category",
    "attack_type",
    "language",
    "difficulty",
    "split",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Read and check an evals/configs/*.yaml file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict) or not config.get("name"):
        raise ValueError(f"{path}: a run config needs a name")
    tiers = config.setdefault("tiers", {}) or {}
    config["tiers"] = tiers
    unknown = set(tiers) - {"tier1", "tier2", "tier3"}
    if unknown:
        raise ValueError(f"{path}: unknown tiers {sorted(unknown)}")
    config.setdefault("fail", "open")
    config.setdefault("mode", "block")
    config["repeats"] = int(config.get("repeats", 1))
    if config["repeats"] < 1:
        raise ValueError(f"{path}: repeats must be at least 1")
    # How long the judge may take, overriding the policy's own settings.timeout. A policy states
    # the SLA for its deployed agent; offline evaluation of a reasoning judge needs far longer.
    # Unset means "use the policy".
    timeout = config.get("timeout")
    if timeout is not None and (not isinstance(timeout, int) or timeout < 1):
        raise ValueError(f"{path}: timeout must be a positive whole number of seconds")
    config["timeout"] = timeout
    tier3 = tiers.get("tier3")
    if tier3 is not None and not (
        tier3.get("provider") and (tier3.get("model") or tier3.get("model_env"))
    ):
        raise ValueError(f"{path}: tier3 needs a provider and a model (or model_env)")
    return config


# ---------------------------------------------------------------------------
# Building the firewall from a config
# ---------------------------------------------------------------------------


class CachingStreamer:
    """Wraps a Tier 3 streamer: a reply seen before is replayed from disk instead of re-asked.

    Only complete replies are stored. Each repeat gets its own view (for_repeat), and the repeat is
    part of the key, so repeat 2 is a fresh sample of the judge, not a copy of repeat 1. The judge
    is called from the firewall's own thread, so the repeat is fixed on the view, never thread state.
    """

    def __init__(self, inner, model: str, cache_dir: Path, repeat: int = 0):
        self._inner = inner
        self._model = model
        self._dir = cache_dir
        self._repeat = repeat
        self._dir.mkdir(parents=True, exist_ok=True)

    def for_repeat(self, repeat: int) -> CachingStreamer:
        return CachingStreamer(self._inner, self._model, self._dir, repeat)

    def ping(self, system_p, user_p, **kwargs):
        key = hashlib.sha256(
            json.dumps([self._model, self._repeat, system_p, user_p]).encode("utf-8")
        ).hexdigest()
        path = self._dir / f"{key}.txt"
        if path.exists():
            yield path.read_text(encoding="utf-8")
            return
        parts = []
        for chunk in self._inner.ping(system_p=system_p, user_p=user_p, **kwargs):
            token = _extract_token(chunk)
            if token is not None:
                parts.append(token)
            yield chunk
        if not "".join(parts).strip():
            return  # no text (a refusal, a filter, a cut stream): replaying it would lose why
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text("".join(parts), encoding="utf-8")
        tmp.replace(path)  # atomic: a reader never sees half a reply


def _setting(tier3: dict, key: str) -> str | None:
    """A tier3 setting: the environment variable the config names (`key_env`) when it is set,
    else the value written in the config (`key`). Per-account values such as an Azure endpoint
    stay out of the repository; a value in the config is the default."""
    env = tier3.get(f"{key}_env")
    if env and os.environ.get(env):
        return os.environ[env]
    return str(tier3[key]) if tier3.get(key) else None


# The environment variable each provider's SDK reads its endpoint from when the client is built.
_BASE_URL_ENV = {"claude": "ANTHROPIC_BASE_URL", "openai": "OPENAI_BASE_URL"}


def _with_base_url(tier3: dict, build):
    """Build the streamer against another endpoint when the config names one (`base_url`, or
    `base_url_env`): an Anthropic- or OpenAI-compatible API such as Z.ai's. The SDK reads the
    endpoint from its environment variable when the client is built, so it is set only for that
    moment and restored, leaving the firewall's providers untouched."""
    base_url = _setting(tier3, "base_url")
    if not base_url:
        return build()
    env = _BASE_URL_ENV.get(tier3["provider"])
    if env is None:
        raise ValueError(f"base_url is not supported for provider {tier3['provider']!r}")
    previous = os.environ.get(env)
    os.environ[env] = base_url
    try:
        return build()
    finally:
        if previous is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = previous


def build_streamer(tier3: dict | None, config_name: str, use_cache: bool):
    """The Tier 3 judge named in the config, or None. The API key comes from the environment
    variable the config names (default HUMANBOUND_FIREWALL_API_KEY); model, endpoint and
    api_version may also be read from the environment through model_env, endpoint_env and
    api_version_env."""
    if not tier3:
        return None
    key_env = tier3.get("api_key_env", "HUMANBOUND_FIREWALL_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise ValueError(f"tier3 needs an API key: set {key_env}")
    model = _setting(tier3, "model")
    endpoint, api_version = _setting(tier3, "endpoint"), _setting(tier3, "api_version")
    if tier3["provider"] == "azureopenai":
        missing = [
            tier3.get(f"{k}_env") or k
            for k, v in (("endpoint", endpoint), ("api_version", api_version))
            if not v
        ]
        if missing:
            raise ValueError(f"Azure OpenAI needs an endpoint and an api_version: set {missing}")
    provider = Provider(
        name=tier3["provider"],
        integration=ProviderIntegration(
            api_key=api_key, model=model or "", endpoint=endpoint, api_version=api_version
        ),
    )
    streamer = _with_base_url(tier3, lambda: get_llm_streamer(provider))
    if use_cache:
        streamer = CachingStreamer(streamer, model or "", CACHE_DIR / config_name)
    return streamer


def build_ensemble(tier1: dict | None):
    if not tier1:
        return None
    detectors = [AttackDetector(d) for d in tier1.get("detectors") or []]
    if not detectors:
        raise ValueError("tier1 needs at least one detector")
    return AttackDetectorEnsemble(detectors, consensus=int(tier1.get("consensus", 1)))


def build_classifier(tier2: dict | None):
    if not tier2:
        return None
    from humanbound_firewall.hbfw import HBFW, load_hbfw, load_model_class

    if not tier2.get("model_path") or not tier2.get("detector_script"):
        raise ValueError("tier2 needs model_path and detector_script")
    hbfw_config, weights = load_hbfw(str(tier2["model_path"]))
    detector = load_model_class(str(tier2["detector_script"]))
    classifier = HBFW(attack_detector=detector("attack"), benign_detector=detector("benign"))
    classifier.load(hbfw_config, weights)
    return classifier


class Firewalls:
    """One Firewall per policy and repeat, all sharing the config's tiers (built once). A repeat
    has its own Firewall only so that its judge cache view is its own."""

    def __init__(self, config: dict, policies_dir: Path, use_cache: bool):
        tiers = config["tiers"]
        self._config = config
        self._policies_dir = policies_dir
        self.timeout = config.get("timeout")  # None: each policy's own settings.timeout
        self.streamer = build_streamer(tiers.get("tier3"), config["name"], use_cache)
        self._ensemble = build_ensemble(tiers.get("tier1"))
        self._classifier = build_classifier(tiers.get("tier2"))
        self._built: dict[tuple[str, int], Firewall] = {}
        self._policies: dict[str, AgentConfig] = {}
        self._lock = threading.RLock()

    def policy_config(self, policy: str) -> AgentConfig:
        """The parsed policy, cached. The judge input is built from its `expects`, so a case needs
        the policy as well as its own boundary."""
        with self._lock:
            if policy not in self._policies:
                self._policies[policy] = load_config(self._policies_dir / policy)
            return self._policies[policy]

    def get(self, policy: str, repeat: int = 0) -> Firewall:
        with self._lock:
            key = (policy, repeat)
            if key not in self._built:
                streamer = self.streamer
                if isinstance(streamer, CachingStreamer):
                    streamer = streamer.for_repeat(repeat)
                self._built[key] = Firewall(
                    self.policy_config(policy),
                    streamer=streamer,
                    ensemble=self._ensemble,
                    scope_classifier=self._classifier,
                    mode=self._config["mode"],
                    fail=self._config["fail"],
                )
            return self._built[key]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def judge_boundary(firewalls: Firewalls, case: Case) -> dict[str, Any] | None:
    """The case's boundary, plus the policy's `expects` for that tool.

    The judge is told what the boundary it screens normally returns: an adapter sends this in
    production (LangChain's `_tool_boundary` attaches `config.tool_expects`), so a case that omits
    it would measure a different prompt from the one that ships. The case may state its own
    `expects`; the policy's entry is the default.
    """
    if not case.boundary:
        return case.boundary
    expects = firewalls.policy_config(case.policy).tool_expects.get(case.boundary.get("name", ""))
    if not expects or case.boundary.get("expects"):
        return case.boundary
    return {**case.boundary, "expects": expects}


def run_case(firewalls: Firewalls, case: Case, repeat: int) -> dict[str, Any]:
    """Screen one case and return its result line. Never raises: a harness error is recorded."""
    result: dict[str, Any] = {k: getattr(case, k) for k in _CASE_FIELDS}
    result["repeat"] = repeat
    started = time.perf_counter()
    try:
        decision = firewalls.get(case.policy, repeat).inspect(
            case.payload,
            cls=case.cls,
            window=case.window or None,
            boundary=judge_boundary(firewalls, case),
            timeout=firewalls.timeout,
        )
        result.update(
            verdict=decision.verdict.value,
            category=decision.category.value,
            tier=decision.tier,
            action=decision.action,
            mode_applied=decision.mode_applied,
            attack_probability=decision.probabilities.get("attack", 0.0),
            latency_ms=decision.elapsed_ms,
            explanation=decision.wait_explanation(timeout=EXPLANATION_TIMEOUT),
            error="",
        )
    except Exception as e:  # a broken case or harness bug must not stop the whole run
        result.update(
            verdict="",
            category="",
            tier=None,
            action="",
            mode_applied="",
            attack_probability=0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            explanation="",
            error=f"{type(e).__name__}: {e}",
        )
    return result


def run(
    config_path: str | Path,
    dataset: str | Path = DATASETS_DIR,
    *,
    out_dir: str | Path = RESULTS_DIR,
    policies_dir: str | Path = POLICIES_DIR,
    split: str | None = None,
    limit: int | None = None,
    workers: int = 4,
    use_cache: bool = True,
    repeats: int | None = None,
    timeout: int | None = None,
) -> Path:
    """Run every case (times the config's repeats) and return the path of the results file."""
    config = load_run_config(config_path)
    if repeats is not None:  # a one-off override; the meta file records what ran
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        config["repeats"] = repeats
    if timeout is not None:  # same: a one-off override, recorded in the meta file
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        config["timeout"] = timeout
    cases = load_cases(dataset, policies_dir=policies_dir)
    if split:
        cases = [c for c in cases if c.split == split]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no cases to run")

    firewalls = Firewalls(config, Path(policies_dir), use_cache)
    jobs = [(case, r) for case in cases for r in range(config["repeats"])]

    started = datetime.now(timezone.utc)
    run_id = f"{config['name']}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{run_id}.jsonl"

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda job: run_case(firewalls, *job), jobs))
    elapsed = time.perf_counter() - t0

    with open(results_path, "w", encoding="utf-8") as f:
        for line in results:  # pool.map keeps the input order: the file is stable run to run
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    meta = {
        "run_id": run_id,
        "config": config,
        "config_path": str(config_path),
        "dataset": str(dataset),
        "dataset_version": _dataset_version(Path(dataset)),
        "split": split,
        "cases": len(cases),
        "results": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "firewall_version": humanbound_firewall.__version__,
        "git_commit": _git_commit(),
        "started": started.isoformat(),
        "seconds": round(elapsed, 2),
    }
    meta_path = results_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return results_path


def _dataset_version(dataset: Path) -> str:
    version = (dataset if dataset.is_dir() else dataset.parent) / "VERSION"
    return version.read_text(encoding="utf-8").strip() if version.exists() else ""


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=EVALS_DIR,
            timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--config", required=True, help="evals/configs/<name>.yaml")
    parser.add_argument("--dataset", default=str(DATASETS_DIR), help="a .jsonl file or a folder")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="where results are written")
    parser.add_argument("--split", choices=("dev", "test"), help="run only this split")
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--workers", type=int, default=4, help="cases screened in parallel")
    parser.add_argument("--no-cache", action="store_true", help="always ask the LLM judge")
    parser.add_argument("--repeats", type=int, help="override the config's repeats for this run")
    parser.add_argument(
        "--timeout", type=int, help="override the policy's judge timeout (seconds) for this run"
    )
    args = parser.parse_args(argv)

    path = run(
        args.config,
        args.dataset,
        out_dir=args.out,
        split=args.split,
        limit=args.limit,
        workers=args.workers,
        use_cache=not args.no_cache,
        repeats=args.repeats,
        timeout=args.timeout,
    )
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    errors = sum(1 for r in results if r["error"])
    matched = sum(1 for r in results if r["verdict"] == r["expected"])
    print(f"{len(results)} results -> {path}")
    print(f"verdict matched expected: {matched}/{len(results)}  harness errors: {errors}")
    print(f"score it: python -m evals.score --results {path}")


if __name__ == "__main__":
    main()
