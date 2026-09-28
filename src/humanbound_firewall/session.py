# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Session — a pure, serialisable value distilling what the firewall has concluded in a thread.

The firewall is stateless: it never stores a session. The caller carries the token between calls
(in an agent framework's state, a cookie, a database row) and hands it back with the next payload —
token in, token out. A session is built from verdict-time facts only (verdict, category, class,
boundary, a hash of the payload) — never from the payload's text and never from the judge's streamed
explanation — so it is available the moment a verdict is, and it can never leak content.

Scope: the thread (the whole conversation). It also tracks the current run (one user request and
the cascade of tool calls it triggers) through `start_run()`, which pins the request so the judge
keeps seeing the user's intent after the conversation window has scrolled past it.

Posture rule (fixed by design): after a block anywhere in the thread the posture is "elevated" for
the rest of the thread — no time decay, no reset at a run boundary. The consumer (the Tier 2
thresholds) tightens one notch on an elevated posture. An elevated posture never turns uncertain
verdicts into blocks: one false positive must not cascade into refusals.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

VERSION = 1


def _empty_counts() -> dict[str, Any]:
    return {"evaluations": 0, "blocks": 0, "escalations": 0, "by_class": {}, "by_category": {}}


def _entry_key(entry: dict) -> str:
    return entry.get("id") or json.dumps(entry, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Session:
    """Immutable: every operation returns a new Session."""

    RECENT_LIMIT = 20

    run: int = 0
    pinned_request: str = ""
    counts: dict[str, Any] = field(default_factory=_empty_counts)
    recent: tuple = ()
    last_seen: str = ""

    # ------------------------------------------------------------------ construction
    @classmethod
    def new(cls) -> Session:
        return cls()

    @classmethod
    def from_json(cls, token) -> Session:
        """Accepts a dict, a JSON string, or nothing (-> a new session)."""
        if not token:
            return cls.new()
        data = json.loads(token) if isinstance(token, str) else dict(token)
        if not data:
            return cls.new()
        counts = _empty_counts()
        counts.update(data.get("counts") or {})
        return cls(
            run=int(data.get("run", 0)),
            pinned_request=str(data.get("pinned_request", "")),
            counts=counts,
            recent=tuple(dict(e) for e in data.get("recent") or ()),
            last_seen=str(data.get("last_seen", "")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "v": VERSION,
            "run": self.run,
            "pinned_request": self.pinned_request,
            "counts": self.counts,
            "recent": list(self.recent),
            "posture": self.posture,
            "last_seen": self.last_seen,
        }

    @classmethod
    def merge(cls, a, b) -> Session:
        """Join two branches of the same thread that recorded in parallel (e.g. parallel tool
        calls in one graph step). Every verdict the right branch recorded and the left has not
        seen is folded into the left, so nothing is lost and a block on either branch is kept:
        the posture can only rise. Exact as long as a branch's new entries fit the recent
        window, which parallel calls in one step always do."""
        left = a if isinstance(a, Session) else cls.from_json(a)
        right = b if isinstance(b, Session) else cls.from_json(b)
        if left == right:
            return left
        seen = {_entry_key(e) for e in left.recent}
        new = [e for e in right.recent if _entry_key(e) not in seen]
        counts = json.loads(json.dumps(left.counts))
        for e in new:
            counts["evaluations"] += 1
            counts["blocks"] += e.get("v") == "block"
            counts["escalations"] += e.get("v") == "review"
            counts["by_class"][e["cls"]] = counts["by_class"].get(e["cls"], 0) + 1
            if e.get("v") == "block" and e.get("c"):
                counts["by_category"][e["c"]] = counts["by_category"].get(e["c"], 0) + 1
        newer = right if right.last_seen >= left.last_seen else left
        return cls(
            run=max(left.run, right.run),
            pinned_request=newer.pinned_request or left.pinned_request or right.pinned_request,
            counts=counts,
            recent=(*left.recent, *new)[-cls.RECENT_LIMIT :],
            last_seen=max(left.last_seen, right.last_seen),
        )

    # ------------------------------------------------------------------ derived
    @property
    def posture(self) -> str:
        return "elevated" if self.counts.get("blocks", 0) > 0 else "normal"

    # ------------------------------------------------------------------ operations
    def start_run(self, request: str) -> Session:
        """A new run in this thread: the user's request is pinned so intent never ages out."""
        return replace(self, run=self.run + 1, pinned_request=request or "", last_seen=_now())

    def record(self, result, *, cls: str, boundary: str | None = None) -> Session:
        """Fold one verdict in. `result` is an EvalResult; only its verdict-time facts are used."""
        verdict = getattr(result.verdict, "value", result.verdict)
        category = getattr(result.category, "value", result.category) or ""
        counts = json.loads(json.dumps(self.counts))  # deep copy of a small dict
        counts["evaluations"] += 1
        counts["blocks"] += verdict == "block"
        counts["escalations"] += verdict == "review"
        counts["by_class"][cls] = counts["by_class"].get(cls, 0) + 1
        if verdict == "block" and category:
            counts["by_category"][category] = counts["by_category"].get(category, 0) + 1
        digest = hashlib.sha256((result.prompt or "").encode("utf-8")).hexdigest()[:16]
        entry = {
            "id": secrets.token_hex(4),  # tells parallel branches' entries apart when merging
            "h": digest,
            "cls": cls,
            "b": boundary or "",
            "v": verdict,
            "c": category,
        }
        recent = (*self.recent, entry)[-self.RECENT_LIMIT :]
        return replace(self, counts=counts, recent=recent, last_seen=_now())
