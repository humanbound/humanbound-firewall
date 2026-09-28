# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Judge prompt builder — constructs the system prompt for LLM evaluation."""

from pathlib import Path

from .models import AgentConfig, Turn

_PROMPTS = Path(__file__).parent / "prompts"

# One judge per trust class. "request" judges a principal's turn; "ingest" judges outside
# content (tool output, pages, documents) as untrusted data; "recall" checks the integrity of
# the agent's own records coming back. Every firewall class needs an entry here.
_TEMPLATES = {
    "request": (_PROMPTS / "judge.txt").read_text(encoding="utf-8"),
    "ingest": (_PROMPTS / "judge_ingest.txt").read_text(encoding="utf-8"),
    "recall": (_PROMPTS / "judge_recall.txt").read_text(encoding="utf-8"),
}


def build_system_prompt(
    config: AgentConfig,
    few_shots: list[dict] | None = None,
    session_turns: list[Turn] | None = None,
    cls: str = "request",
    boundary: dict | None = None,
) -> str:
    """Build the complete judge system prompt from config, few-shots, and session context.

    `cls` selects the judge that matches the payload's trust class. `boundary` names where the
    payload crossed into the agent ({"name", "kind", "description", "expects"}): the judge is
    told what that boundary is for and what the policy expects it to return, so content that is
    what the boundary normally returns reads as information, not as a directive.
    """
    if cls not in _TEMPLATES:
        raise ValueError(f"No judge for trust class '{cls}'. Expected one of {tuple(_TEMPLATES)}.")

    # Format intents as bullet lists
    permitted = (
        "\n".join(f" - {i}" for i in config.permitted_intents)
        if config.permitted_intents
        else " - (none defined)"
    )
    restricted = (
        "\n".join(f" - {i}" for i in config.restricted_intents)
        if config.restricted_intents
        else " - (none defined)"
    )

    # More info section
    more_info = f"**Additional Info:** {config.more_info}" if config.more_info else ""

    # Few-shot examples: only the ones learned on this class (recall takes none)
    few_shots_text = _format_few_shots(few_shots or config.few_shots, cls)

    # Conversation context (session turns)
    context_text = _format_session_context(session_turns)

    prompt = _TEMPLATES[cls].format(
        business_scope=config.business_scope or "(not defined)",
        more_info=more_info,
        permitted_intents=permitted,
        restricted_intents=restricted,
        few_shots=few_shots_text,
        conversation_context=context_text,
        boundary=_format_boundary(boundary),
    )

    return prompt


def _format_boundary(boundary: dict | None) -> str:
    """The boundary the payload crossed, as facts for the judge (empty when unknown)."""
    if not boundary or not boundary.get("name"):
        return ""
    lines = [f"## BOUNDARY\nThis payload arrived through `{boundary['name']}`"]
    if boundary.get("kind"):
        lines[0] += f" (kind: {boundary['kind']})"
    lines[0] += "."
    if boundary.get("description"):
        lines.append(f"- What the agent uses it for: {boundary['description']}")
    if boundary.get("expects"):
        lines.append(f"- What the policy expects it to return: {boundary['expects']}")
    if boundary.get("args"):
        lines.append(f"- Called with: {_short_args(boundary['args'])}")
    return "\n".join(lines)


def _short_args(args, limit: int = 600) -> str:
    """The call's arguments as `name=value` pairs (no braces: the template is str.format-ed),
    cut so a large argument cannot flood the prompt."""
    import json

    if isinstance(args, dict):
        parts = []
        for key, value in args.items():
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(value)
            parts.append(f"{key}={text}")
        text = ", ".join(parts)
    else:
        text = str(args)
    text = text.replace("{", "(").replace("}", ")")
    return text if len(text) <= limit else text[:limit] + "…"


_FEW_SHOT_HEADERS = {
    "request": "Previously detected attacks in requests to block:",
    "ingest": "Previously detected attacks in fetched content to block:",
}


def _format_few_shots(few_shots: list[dict], cls: str = "request") -> str:
    """Format the few-shot examples learned on this class as attack patterns.

    An example belongs to the class it was found on (`class`, default "request"): a user prompt
    that broke the agent teaches the request judge, a page that did teaches the ingest judge, and
    neither teaches the other. The recall judge takes none: a record either instructs or it does
    not, and examples would only put attack text in front of an integrity check.
    """
    if not few_shots or cls not in _FEW_SHOT_HEADERS:
        return ""
    mine = [ex for ex in few_shots if (ex.get("class") or "request") == cls]
    if not mine:
        return ""

    lines = [f"## LEARNED ATTACK PATTERNS\n\n{_FEW_SHOT_HEADERS[cls]}\n"]
    for ex in mine:
        prompt = ex.get("prompt", "")
        if not prompt:
            continue
        preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        verdict = ex.get("verdict", "block").upper()
        lines.append(f'"{preview}" → {verdict}')

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_session_context(turns: list[Turn] | None) -> str:
    """Format session turns as conversation context for the judge."""
    if not turns:
        return ""

    lines = ["## CONVERSATION CONTEXT (recent turns)\n"]
    for i, turn in enumerate(turns):
        if turn.assistant:
            lines.append(f"Agent (turn {i + 1}): {turn.assistant[:300]}")
        if turn.user:
            lines.append(f"User (turn {i + 1}): {turn.user[:300]}")
        if getattr(turn, "tool", ""):
            lines.append(f"Tool output (turn {i + 1}, untrusted): {turn.tool[:300]}")

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)
