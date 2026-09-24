# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Streaming LLM judge — the verdict is the letter the reply starts with.

The judge answers `<letter> <explanation>`. The letter is read off the start of the reply and only
counts as a standalone word: the first word is read to its boundary before it is trusted, so "C"
followed by "ertainly" is not a restriction verdict, and a letter inside the payload's own text
(a judge captured into echoing it) never becomes one. A reply that opens with anything else — a
preamble, a continuation of the payload — is a protocol violation: the judge is asked once more,
tersely, and if the retry does not start with a letter either the result is "no verdict", which a
fail-closed deployment withholds.
"""

import threading

from .firewall import _extract_token
from .models import VERDICT_MAP, Category, EvalResult, Verdict

_WRAPPERS = " \t\r\n*#`\"'_>-"
_RETRY_SUFFIX = "\n\nReply with the verdict letter only (P, A, B, C or D), then the explanation."


def _read_verdict(head: str, ended: bool = False):
    """Read the verdict off the start of the reply.

    Returns (letter, remainder) once the reply is known to start with a standalone verdict
    letter, (None, None) while more text is needed to know, and (None, "") once it is clear
    the reply does not start with one.
    """
    i = 0
    while i < len(head) and head[i] in _WRAPPERS:
        i += 1
    if i >= len(head):
        return (None, "") if ended else (None, None)  # only wrappers so far
    first = head[i]
    if not first.isalpha() or first.upper() not in VERDICT_MAP:
        return None, ""  # the reply starts with something else
    if i + 1 >= len(head):
        # The word is not finished: wait for its boundary (a one-letter reply resolves at the end).
        return (first.upper(), "") if ended else (None, None)
    if head[i + 1].isalpha():
        return None, ""  # the letter is the start of a word
    rest = head[i + 1 :].lstrip(_WRAPPERS.replace("-", "").replace(">", ""))
    return first.upper(), rest


def stream_and_extract_verdict(
    streamer, system_prompt: str, user_prompt: str, timeout: int, session_id: str
) -> EvalResult:
    """Stream the judge's reply; return as soon as the verdict letter is known."""
    result_holder: list[EvalResult | None] = [None]
    decision_event = threading.Event()
    explanation_ready = threading.Event()

    def stream_once(user_p: str):
        """One judge call. Returns (letter, explanation_parts, head): letter None = no verdict."""
        head, parts, letter = "", [], None
        stream = streamer.ping(
            system_p=system_prompt, user_p=user_p, max_tokens=1024, temperature=0.0
        )
        for chunk in stream:
            token = _extract_token(chunk)
            if token is None:
                continue
            if letter is not None:
                parts.append(token)
                continue
            head += token
            found, remainder = _read_verdict(head)
            if found is None:
                if remainder is not None:
                    return None, [], head  # decided: no verdict; stop reading the stream
                continue
            letter = found
            _decide(letter)
            if remainder.strip():
                parts.append(remainder)
        if letter is None:
            found, remainder = _read_verdict(head, ended=True)
            if found is not None:
                letter = found
                _decide(letter)
        return letter, parts, head

    def _decide(letter):
        verdict, category = VERDICT_MAP[letter]
        result_holder[0] = EvalResult(
            verdict=verdict,
            category=category,
            raw_letter=letter,
            session_id=session_id,
            _explanation_ready=explanation_ready,
        )
        decision_event.set()

    def _run():
        try:
            letter, parts, head = stream_once(user_prompt)
            if letter is None and head.strip():
                # A protocol violation: ask once more, tersely.
                letter, parts, head2 = stream_once(user_prompt + _RETRY_SUFFIX)
                head = head2 if letter is None else head
        except Exception as e:
            if result_holder[0] is None:
                result_holder[0] = EvalResult(
                    verdict=Verdict.REVIEW,
                    category=Category.UNCERTAIN,
                    explanation=f"Stream error: {str(e)[:200]}",
                    session_id=session_id,
                )
                decision_event.set()
            explanation_ready.set()
            return

        if result_holder[0] is not None:
            explanation = "".join(parts).strip()
            result_holder[0].explanation = explanation.lstrip(" \t\r\n*#`\"'_").strip()
        else:
            said = head.strip().replace("\n", " ")[:120]
            explanation = (
                f"Judge did not start with a verdict: '{said}'"
                if said
                else "Judge returned no verdict."
            )
            result_holder[0] = EvalResult(
                verdict=Verdict.REVIEW,
                category=Category.UNCERTAIN,
                explanation=explanation,
                session_id=session_id,
            )
            decision_event.set()
        explanation_ready.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if not decision_event.wait(timeout=timeout):
        raise TimeoutError("Evaluation timed out waiting for verdict.")

    return result_holder[0]
