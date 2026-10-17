# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Streaming LLM judge — the verdict is the letter the reply starts with.

The judge answers `<letter> <explanation>`. The letter is read off the start of the reply and only
counts as a standalone word: the first word is read to its boundary before it is trusted, so "C"
followed by "ertainly" is not a restriction verdict, and a letter inside the payload's own text
(a judge captured into echoing it) never becomes one. A reply that opens with anything else — a
preamble, a continuation of the payload — is a protocol violation: the judge is asked once more,
tersely, and if the retry does not start with a letter either the result is "no verdict", which a
fail-closed deployment withholds. The first reply is dropped as soon as it is known to hold no
verdict, so the retry starts at once; the retry alone is read to its end.

A judge whose provider refuses to evaluate the payload — Claude stopping with `refusal`, an Azure
OpenAI content filter, an OpenAI `refusal` — gives no letter either, but that is not doubt: the
provider's own safety layer judged the text harmful. The firewall's job is to stop exactly such
text, so by default a refusal is a block (`on_refusal="block"`). A deployment that would rather
treat it as an uncertain verdict, decided by its fail mode, sets `on_refusal="review"`.
"""

import threading

from .firewall import _extract_token
from .models import VERDICT_MAP, Category, EvalResult, Verdict

_WRAPPERS = " \t\r\n*#`\"'_>-"
_RETRY_SUFFIX = "\n\nReply with the verdict letter only (P, A, B, C or D), then the explanation."
ON_REFUSAL = ("block", "review")
# What a provider's content filter says when it rejects the request itself (Azure OpenAI's
# prompt filter, OpenAI's moderation), as opposed to any other error.
_FILTER_MARKERS = ("content_filter", "content management policy", "content_policy_violation")


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


def _refusal(chunk) -> str | None:
    """The reason, when a streamed chunk says the provider refused to answer; else None.

    Anthropic ends a refused message with `stop_reason: "refusal"` (and a `stop_details` category);
    OpenAI and Azure OpenAI send a `refusal` text or finish with `content_filter`. Only real strings
    count, so a chunk without these fields is never mistaken for a refusal.
    """
    if getattr(chunk, "type", None) == "message_delta":
        delta = getattr(chunk, "delta", None)
        if getattr(delta, "stop_reason", None) == "refusal":
            details = getattr(delta, "stop_details", None) or getattr(chunk, "stop_details", None)
            category = getattr(details, "category", None)
            return (
                f"provider refusal: {category}" if isinstance(category, str) else "provider refusal"
            )
        return None
    choices = getattr(chunk, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        choice = choices[0]
        refusal = getattr(getattr(choice, "delta", None), "refusal", None)
        if isinstance(refusal, str) and refusal.strip():
            return f"provider refusal: {refusal.strip()[:120]}"
        if getattr(choice, "finish_reason", None) == "content_filter":
            return "provider content filter"
    return None


def _filtered(error: BaseException) -> str | None:
    """The reason, when an error is the provider's content filter rejecting the request; else None."""
    code = getattr(error, "code", None)
    text = str(error)
    if code == "content_filter" or any(marker in text for marker in _FILTER_MARKERS):
        return "provider content filter"
    return None


def stream_and_extract_verdict(
    streamer,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    session_id: str,
    on_refusal: str = "block",
) -> EvalResult:
    """Stream the judge's reply; return as soon as the verdict letter is known."""
    if on_refusal not in ON_REFUSAL:
        raise ValueError(f"Unknown on_refusal '{on_refusal}'. Expected one of {ON_REFUSAL}.")
    result_holder: list[EvalResult | None] = [None]
    decision_event = threading.Event()
    explanation_ready = threading.Event()

    def stream_once(user_p: str, read_to_end: bool = False):
        """One judge call. Returns (letter, explanation_parts, head, refusal): letter None = no
        verdict; refusal names why the provider refused, when it did. A first reply that holds
        no verdict is dropped where that is known, so the retry is not made to wait out a
        preamble; only the retry is read to its end (`read_to_end`), to catch a late refusal."""
        head, parts, letter, no_verdict = "", [], None, False
        stream = streamer.ping(
            system_p=system_prompt, user_p=user_p, max_tokens=1024, temperature=0.0
        )
        for chunk in stream:
            if letter is None and (no_verdict or not head.strip()):
                # A refusal counts while the reply holds no verdict: before the judge has said
                # anything, and after what it has said is known not to be a verdict letter — a
                # preamble answers nothing, so the refusal is the only signal there is. It does
                # not count while a letter is still resolving ("P" before its boundary) or once
                # one has been read: a filter that cuts an explanation short leaves the verdict.
                refused = _refusal(chunk)
                if refused:
                    return None, [], head, refused
            try:
                token = _extract_token(chunk)
            except Exception as e:
                # A chunk that cannot be read is not an answer. Gemini raises here when its
                # safety filter blocked the reply, so the refusal is the chunk itself.
                if letter is not None:
                    break  # the verdict is already in; stop reading the explanation
                return None, [], head, f"provider refusal: unreadable chunk ({type(e).__name__})"
            if no_verdict or token is None:
                # The reply does not start with a letter. Nothing more is read from it, but the
                # stream is followed to its end: a refusal still decides, and arrives last.
                continue
            if letter is not None:
                parts.append(token)
                continue
            head += token
            found, remainder = _read_verdict(head)
            if found is None:
                if remainder is not None:
                    no_verdict = True  # decided: no verdict, but a refusal may still follow
                    if not read_to_end:
                        # Leave at once: the retry must not wait out a long preamble, or a
                        # judge that only thinks out loud costs the caller its whole timeout.
                        return None, [], head, None
                continue
            letter = found
            _decide(letter)
            if remainder.strip():
                parts.append(remainder)
        if letter is None and not no_verdict:
            found, remainder = _read_verdict(head, ended=True)
            if found is not None:
                letter = found
                _decide(letter)
        return letter, parts, head, None

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

    def _refused(reason: str):
        """The provider refused to evaluate the payload: a block by default."""
        if on_refusal == "block":
            result_holder[0] = EvalResult(
                verdict=Verdict.BLOCK,
                category=Category.VIOLATION,
                explanation=f"Judge refused to evaluate the payload ({reason}); treated as an attack.",
                session_id=session_id,
            )
        else:
            result_holder[0] = EvalResult(
                verdict=Verdict.REVIEW,
                category=Category.UNCERTAIN,
                explanation=f"Judge refused to evaluate the payload ({reason}).",
                session_id=session_id,
            )
        decision_event.set()

    def _run():
        try:
            letter, parts, head, refused = stream_once(user_prompt)
            if letter is None and not refused and head.strip():
                # A protocol violation: ask once more, tersely.
                letter, parts, head2, refused = stream_once(
                    user_prompt + _RETRY_SUFFIX, read_to_end=True
                )
                head = head2 if letter is None else head
        except Exception as e:
            if result_holder[0] is None:
                filtered = _filtered(e)
                if filtered:
                    _refused(filtered)
                else:
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
        elif refused:
            _refused(refused)
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
