# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Streaming LLM judge — extracts verdict from first token of streamed response."""

import threading

from .firewall import _extract_token
from .models import VERDICT_MAP, Category, EvalResult, Verdict


def stream_and_extract_verdict(
    streamer, system_prompt: str, user_prompt: str, timeout: int, session_id: str
) -> EvalResult:
    """Stream LLM response, extract verdict letter from first token."""
    result_holder = [None]
    decision_event = threading.Event()

    def _run():
        verdict_letter = None
        explanation_parts = []

        try:
            stream = streamer.ping(
                system_p=system_prompt,
                user_p=user_prompt,
                max_tokens=1024,
                temperature=0.0,
            )

            for chunk in stream:
                token = _extract_token(chunk)
                if token is None:
                    continue

                if verdict_letter is None:
                    letter = next(
                        (c for c in token if c.isalpha() and c.upper() in VERDICT_MAP), None
                    )
                    if letter:
                        verdict_letter = letter.upper()
                        verdict, category = VERDICT_MAP[verdict_letter]
                        result_holder[0] = EvalResult(
                            verdict=verdict,
                            category=category,
                            raw_letter=verdict_letter,
                            session_id=session_id,
                        )
                        decision_event.set()
                        remainder = token[token.index(letter) + 1 :]
                        if remainder.strip():
                            explanation_parts.append(remainder)
                else:
                    explanation_parts.append(token)

        except Exception as e:
            if result_holder[0] is None:
                result_holder[0] = EvalResult(
                    verdict=Verdict.REVIEW,
                    category=Category.UNCERTAIN,
                    explanation=f"Stream error: {str(e)[:200]}",
                    session_id=session_id,
                )
                decision_event.set()
            return

        if result_holder[0]:
            result_holder[0].explanation = "".join(explanation_parts).strip()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if not decision_event.wait(timeout=timeout):
        raise TimeoutError("Evaluation timed out waiting for verdict.")

    return result_holder[0]
