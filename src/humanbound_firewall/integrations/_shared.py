# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""What every framework adapter shares, and needs no framework for.

- `Lanes`: log mode's background judgements, one lane per conversation (a LangChain thread, a
  Pydantic AI conversation), run one at a time in call order, lanes in parallel;
- `BoundaryInventory`: the trust-boundary inventory (`boundaries`) and its coverage table
  (`report()`), from the policy and the adapter's tool list;
- `json_without_binary`: a non-text payload as JSON for the judge, raw media bytes dropped.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any

logger = logging.getLogger("humanbound_firewall.integrations")


class Lanes:
    """Background judgements for log mode: one lane per conversation. A lane runs its jobs one at a
    time in submission order, each on the session the previous job produced; lanes run in
    parallel. A lane's worker thread exists only while the lane has work."""

    def __init__(self):
        self._cond = threading.Condition()
        self._queues: dict[str, deque] = {}
        self._sessions: dict[str, Any] = {}
        self._pending = 0

    def submit(self, key: str, job) -> None:
        """`job(previous_session_or_None)` judges one payload and returns the session after it."""
        with self._cond:
            self._pending += 1
            queue = self._queues.get(key)
            if queue is not None:  # the lane's worker is running and will get to it
                queue.append(job)
                return
            self._queues[key] = deque([job])
        threading.Thread(
            target=self._drain, args=(key,), daemon=True, name=f"humanbound-firewall-log:{key}"
        ).start()

    def _drain(self, key: str) -> None:
        while True:
            with self._cond:
                queue = self._queues[key]
                if not queue:
                    del self._queues[key]
                    self._cond.notify_all()
                    return
                job = queue.popleft()
                previous = self._sessions.get(key)
            session = None
            try:
                session = job(previous)
            except Exception:  # observe-only: a failed judgement never stops the ones after it
                logger.exception("background firewall judgement failed")
            with self._cond:
                if session is not None:
                    self._sessions[key] = session
                self._pending -= 1
                self._cond.notify_all()

    def session(self, key: str, *, release: bool = False):
        """The latest session a lane produced. `release` forgets it once the lane is idle (it has
        been written back to where the conversation's session lives)."""
        with self._cond:
            session = self._sessions.get(key)
            if release and key not in self._queues:
                self._sessions.pop(key, None)
            return session

    def flush(self, timeout: float | None = None) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)


class BoundaryInventory:
    """`boundaries` and `report()` for an adapter. The adapter sets `firewall`, `_tool_names`,
    `_tool_descriptions`, and the class attributes below."""

    ADAPTER = ""  # the name adapt_to() knows it by
    FRAMEWORK = ""  # the framework, as the report's notes name it
    HUMAN_TURN: dict[str, Any] = {}  # the request boundary: name, kind and hook
    TOOL_HOOK = ""  # the hook every tool boundary is attached at

    firewall: Any
    _tool_names: list[str]
    _tool_descriptions: dict[str, str]

    @property
    def boundaries(self) -> list[dict[str, Any]]:
        """The trust-boundary inventory: what this adapter attaches, with class and mode, and what
        it cannot attach (served by the manual tier)."""
        guard, config = self.firewall.guard, self.firewall.config
        rows: list[dict[str, Any]] = [
            {
                **self.HUMAN_TURN,
                "class": "request",
                "mode": guard.mode_for("request"),
                "expects": "",
            }
        ]
        declared = dict.fromkeys([*config.tool_classes, *config.tool_expects])
        if self._tool_names:
            # The policy can only describe tools the agent has: a stray entry is not a boundary.
            stray = [name for name in declared if name not in self._tool_names]
            if stray:
                logger.warning(
                    "agent.yaml tools: entries name tools this agent does not have: %s",
                    ", ".join(stray),
                )
            names = list(self._tool_names)
        else:
            names = list(declared)  # no tool list given: the policy's entries are the inventory
        for name in names:
            rows.append(self._tool_boundary(name))
        for kind in config.capabilities:
            if kind in ("memory", "inter_agent"):
                rows.append(
                    {
                        "name": kind,
                        "kind": kind,
                        "hook": None,
                        "class": None,
                        "mode": None,
                        "note": (
                            f"not attachable: no generic {self.FRAMEWORK} hook — call "
                            f"firewall.inspect(payload, cls=...) where {kind} content is read"
                        ),
                    }
                )
        return rows

    def report(self) -> str:
        """The coverage table: the inventory as text."""
        config, guard = self.firewall.config, self.firewall.guard
        head = (
            f"Humanbound Firewall · {config.name or 'agent'} · adapter={self.ADAPTER} · "
            f"classes={','.join(guard.classes)}"
        )
        table = [("boundary", "kind", "hook", "class", "mode", "note")]
        for b in self.boundaries:
            notes = []
            if b["class"] == "recall":
                notes.append("integrity mode (declared: recall)")
            if b.get("expects"):
                notes.append(f"expects: {b['expects']}")
            if b.get("note"):
                notes.append(b["note"])
            table.append(
                (
                    b["name"],
                    b["kind"],
                    b["hook"] or "—",
                    b["class"] or "—",
                    b["mode"] or "—",
                    "; ".join(notes),
                )
            )
        widths = [max(len(row[i]) for row in table) for i in range(5)]
        lines = [head]
        for row in table:
            cells = [row[i].ljust(widths[i]) for i in range(5)]
            lines.append(" " + "  ".join(cells) + "  " + row[5])
        return "\n".join(line.rstrip() for line in lines)

    def _tool_boundary(self, name: str, description: str | None = None) -> dict[str, Any]:
        """What the judge is told about a tool boundary: the tool's own description (what the
        agent uses it for) and the policy's `expects`, which defaults to that description. The
        screening call adds `args`, what the tool was called with this time."""
        cls = self.firewall.config.class_of(name)
        description = (
            description if description is not None else self._tool_descriptions.get(name, "")
        )
        return {
            "name": name,
            "kind": "tools",
            "hook": self.TOOL_HOOK,
            "class": cls,
            "mode": self.firewall.guard.mode_for(cls),
            "description": description,
            "expects": self.firewall.config.tool_expects.get(name) or description,
        }


# Dropped from a payload's JSON form: raw media bytes, which carry no text and would swamp the judge.
BINARY_FIELDS = frozenset({"base64", "data_base64", "bytes"})


def json_without_binary(value) -> str:
    """A non-text payload as JSON for the judge, with raw media bytes dropped at any depth."""

    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if k not in BINARY_FIELDS}
        if isinstance(v, list | tuple):
            return [strip(x) for x in v]
        if isinstance(v, bytes | bytearray):
            return None
        return v

    try:
        return json.dumps(strip(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
