# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""The eval case format: one JSON object per line in evals/datasets/*.jsonl.

Required: id, cls, payload, policy, expected.
Optional: window, boundary, expected_category, attack_type, language, source, difficulty, split.

A case is validated when it is loaded, so a broken case fails loudly with its file and line instead
of quietly skewing a score. Each case also records its suite, the name of the file it came from
(e.g. "must_pass"), which the scorer uses for the hard gates.

A tool case must carry the class its policy gives that tool, the way the adapters screen it: every
tool is ingest unless the policy lists it under tools.recall.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from humanbound_firewall.config import load_config
from humanbound_firewall.models import CLASSES, AgentConfig

EXPECTED = ("block", "pass")
SPLITS = ("dev", "test")
DIFFICULTIES = ("easy", "normal", "hard")
ROLES = ("user", "assistant", "tool")
# What a block can be filed under; a pass has no category.
CATEGORIES = ("off_topic", "violation", "restriction", "integrity")

POLICIES_DIR = Path(__file__).parent / "policies"
DATASETS_DIR = Path(__file__).parent / "datasets"

_REQUIRED = ("id", "cls", "payload", "policy", "expected")


class CaseError(ValueError):
    """A case that cannot be used, with where it was found."""


@dataclass
class Case:
    id: str
    cls: str  # request | ingest | recall
    payload: str
    policy: str  # file name under evals/policies/
    expected: str  # block | pass
    window: list[dict] = field(default_factory=list)
    boundary: dict[str, Any] | None = None
    expected_category: str = ""
    attack_type: str = ""  # required for a block, "" for benign cases
    language: str = "en"
    source: str = ""
    difficulty: str = "normal"  # easy | normal | hard
    split: str = "dev"  # dev | test
    suite: str = ""  # set by the loader from the file name, never written in the data

    @classmethod
    def from_dict(cls, data: Any, *, policies_dir: Path = POLICIES_DIR) -> Case:
        """Build a case from one parsed JSON line, or raise CaseError naming what is wrong."""
        if not isinstance(data, dict):
            raise CaseError(f"a case must be a JSON object, got {type(data).__name__}")

        allowed = {f.name for f in fields(cls)} - {"suite"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise CaseError(f"unknown field(s) {unknown}; allowed: {sorted(allowed)}")
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise CaseError(f"missing required field(s) {missing}")

        for key in (*_REQUIRED, "expected_category", "attack_type", "language", "source"):
            if key in data and not isinstance(data[key], str):
                raise CaseError(f"{key} must be a string")
        for key in _REQUIRED:
            if key != "payload" and not data[key].strip():
                raise CaseError(f"{key} must not be empty")

        case = cls(**data)
        case._check(policies_dir)
        return case

    def _check(self, policies_dir: Path) -> None:
        _one_of("cls", self.cls, CLASSES)
        _one_of("expected", self.expected, EXPECTED)
        _one_of("split", self.split, SPLITS)
        _one_of("difficulty", self.difficulty, DIFFICULTIES)
        if not self.language.strip():
            raise CaseError("language must not be empty")

        if self.expected == "block":
            if not self.attack_type.strip():
                raise CaseError("a block case needs an attack_type")
            if self.expected_category:
                _one_of("expected_category", self.expected_category, CATEGORIES)
        elif self.expected_category:
            raise CaseError("a pass case has no expected_category")

        if not (policies_dir / self.policy).is_file():
            raise CaseError(f"policy {self.policy!r} not found in {policies_dir}")

        if not isinstance(self.window, list):
            raise CaseError("window must be a list of messages")
        for i, message in enumerate(self.window):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise CaseError(f"window[{i}] must be {{'role': ..., 'content': ...}}")
            _one_of(f"window[{i}].role", message["role"], ROLES)
            if not isinstance(message["content"], str):
                raise CaseError(f"window[{i}].content must be a string")

        if self.boundary is not None:
            if not isinstance(self.boundary, dict):
                raise CaseError("boundary must be an object")
            name = self.boundary.get("name")
            if not isinstance(name, str) or not name.strip():
                raise CaseError("boundary needs a non-empty name")
            if "args" in self.boundary and not isinstance(self.boundary["args"], dict):
                raise CaseError("boundary.args must be an object")
            if self.boundary.get("kind", "tools") == "tools":
                self._check_tool_class(name, policies_dir)

    def _check_tool_class(self, tool: str, policies_dir: Path) -> None:
        """A tool's output has the class its policy gives it, as the adapters screen it: every
        tool is ingest unless the policy's tools: block says recall. A case that disagrees tests
        something that cannot happen in a real deployment."""
        if self.cls == "request":
            raise CaseError(f"a tool's output ({tool!r}) is never a request; use ingest or recall")
        declared = _policy(policies_dir / self.policy).class_of(tool)
        if declared != self.cls:
            raise CaseError(
                f"tool {tool!r} is {declared} in {self.policy}, but the case says cls={self.cls!r}"
            )


def load_cases(
    path: str | Path = DATASETS_DIR, *, policies_dir: str | Path = POLICIES_DIR
) -> list[Case]:
    """Load and validate every case in a .jsonl file, or in every .jsonl file of a directory.

    Raises CaseError on the first broken case (with its file and line) and on an id used twice,
    across files too.
    """
    path, policies_dir = Path(path), Path(policies_dir)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise CaseError(f"no .jsonl files in {path}")
    elif path.is_file():
        files = [path]
    else:
        raise CaseError(f"{path} does not exist")

    cases: list[Case] = []
    seen: dict[str, str] = {}
    for file in files:
        with open(file, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{file}:{number}"
                try:
                    case = Case.from_dict(json.loads(line), policies_dir=policies_dir)
                except json.JSONDecodeError as e:
                    raise CaseError(f"{where}: invalid JSON: {e.msg}") from e
                except (CaseError, TypeError) as e:
                    raise CaseError(f"{where}: {e}") from e
                if case.id in seen:
                    raise CaseError(f"{where}: duplicate id {case.id!r} (first at {seen[case.id]})")
                seen[case.id] = where
                case.suite = file.stem
                cases.append(case)
    return cases


@functools.cache
def _policy(path: Path) -> AgentConfig:
    """A policy file, parsed once per path for the whole load."""
    return load_config(path)


def _one_of(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise CaseError(f"{name} = {value!r}; expected one of {allowed}")
