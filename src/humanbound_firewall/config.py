# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""YAML configuration loader."""

import warnings
from pathlib import Path

import yaml

from .models import CAPABILITIES, AgentConfig

_TOOL_CLASSES = ("ingest", "recall")  # a tool's output is never a principal's request
_FEW_SHOT_CLASSES = ("request", "ingest")  # the judges that learn from examples; recall takes none
# Parsed for compatibility but never used: the judge runs at temperature 0 and sees every turn
# it is given. Removed in 0.4.
_UNUSED_SETTINGS = ("session_window", "temperature")


def load_config(path: str | Path) -> AgentConfig:
    """Load agent configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}")

    scope = data.get("scope", {})
    intents = data.get("intents", {})
    settings = data.get("settings", {})
    for key in _UNUSED_SETTINGS:
        if key in settings:
            warnings.warn(
                f"settings.{key} in {path} has no effect and will be removed in 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
    capabilities = _capabilities(data.get("capabilities"))
    tool_classes, tool_expects = _tools(data.get("tools"))

    return AgentConfig(
        name=data.get("name", ""),
        version=str(data.get("version", "1.0")),
        business_scope=scope.get("business", ""),
        more_info=scope.get("more_info", ""),
        permitted_intents=intents.get("permitted", []),
        restricted_intents=intents.get("restricted", []),
        timeout=settings.get("timeout", 5),
        mode=settings.get("mode", "block"),
        session_window=settings.get("session_window", 5),
        tier2_min_turns=settings.get("tier2_min_turns", 3),
        temperature=settings.get("temperature", 0.0),
        few_shots=_few_shots(data.get("few_shots")),
        capabilities=capabilities,
        tool_classes=tool_classes,
        tool_expects=tool_expects,
    )


def _few_shots(raw) -> list[dict]:
    """Few-shot examples, each tagged with the class it was learned on (default: request)."""
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("few_shots must be a list of {prompt, verdict, category, class} entries")
    out = []
    for i, ex in enumerate(raw):
        if not isinstance(ex, dict):
            raise ValueError(f"few_shots[{i}] must be a mapping")
        cls = ex.get("class") or "request"
        if cls not in _FEW_SHOT_CLASSES:
            raise ValueError(
                f"few_shots[{i}].class = {cls!r}. Expected one of {_FEW_SHOT_CLASSES}; "
                "the recall judge takes no examples."
            )
        out.append({**ex, "class": cls})
    return out


def _capabilities(raw) -> list[str]:
    if raw is None:
        return ["tools"]
    if not isinstance(raw, list) or not all(isinstance(c, str) for c in raw):
        raise ValueError("capabilities must be a list of names")
    unknown = [c for c in raw if c not in CAPABILITIES]
    if unknown:
        raise ValueError(f"Unknown capabilities {unknown}. Expected a subset of {CAPABILITIES}.")
    return list(raw)


def _tools(raw) -> tuple[dict[str, str], dict[str, str]]:
    """The optional `tools:` block: recall (a list), class (a mapping), expects (a mapping)."""
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        raise ValueError("tools must be a mapping with any of: recall, class, expects")
    unknown = set(raw) - {"recall", "class", "expects"}
    if unknown:
        raise ValueError(f"Unknown keys under tools: {sorted(unknown)}")

    recall = raw.get("recall") or []
    if not isinstance(recall, list) or not all(isinstance(n, str) for n in recall):
        raise ValueError("tools.recall must be a list of tool names")
    classes = raw.get("class") or {}
    if not isinstance(classes, dict):
        raise ValueError("tools.class must map tool names to ingest | recall")
    expects = raw.get("expects") or {}
    if not isinstance(expects, dict):
        raise ValueError("tools.expects must map tool names to a description")

    tool_classes = dict.fromkeys(recall, "recall")
    for name, cls in classes.items():
        if cls not in _TOOL_CLASSES:
            raise ValueError(f"tools.class[{name!r}] = {cls!r}. Expected one of {_TOOL_CLASSES}.")
        if tool_classes.get(name, cls) != cls:
            raise ValueError(f"Tool {name!r} is listed under recall and declared {cls!r}.")
        tool_classes[name] = cls
    return tool_classes, {str(k): str(v) for k, v in expects.items()}
