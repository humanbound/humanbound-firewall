# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Judge prompt builder — constructs the system prompt for LLM evaluation."""

from pathlib import Path

from .models import AgentConfig, Turn

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "judge.txt").read_text(encoding="utf-8")


def build_system_prompt(
    config: AgentConfig,
    few_shots: list[dict] | None = None,
    session_turns: list[Turn] | None = None,
) -> str:
    """Build the complete judge system prompt from config, few-shots, and session context."""

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

    # Few-shot examples
    few_shots_text = _format_few_shots(few_shots or config.few_shots)

    # Conversation context (session turns)
    context_text = _format_session_context(session_turns)

    prompt = _PROMPT_TEMPLATE.format(
        business_scope=config.business_scope or "(not defined)",
        more_info=more_info,
        permitted_intents=permitted,
        restricted_intents=restricted,
        few_shots=few_shots_text,
        conversation_context=context_text,
    )

    return prompt


def _format_few_shots(few_shots: list[dict]) -> str:
    """Format few-shot examples as learned attack patterns."""
    if not few_shots:
        return ""

    lines = ["## LEARNED ATTACK PATTERNS\n\nPreviously detected attacks to block:\n"]
    for ex in few_shots:
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

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)
