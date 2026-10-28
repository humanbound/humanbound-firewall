# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Pydantic AI adapter: what `firewall.adapt_to("pydantic_ai")` returns.

A Pydantic AI capability (the framework's middleware) that attaches every boundary Pydantic AI
exposes, calls `Firewall.inspect()` there, enforces the Decision, and keeps the Session per
conversation:

- ``before_run``         starts the run: the conversation's session is read (or created), the run
                         counter advances and the user's request is pinned. The request is then
                         judged as a ``request``; a rejected request ends the run with the
                         replacement text as the agent's reply (``wrap_model_request`` answers
                         with it, so the model is never called);
- ``wrap_tool_execute``  a tool's return value is ``ingest`` — or ``recall`` when the policy's
                         ``tools:`` block says the tool returns our own records. A withheld result
                         is replaced by the template text before the model reads it.

The session lives in the adapter, keyed by the run's ``conversation_id``: Pydantic AI carries that
id on the message history, so passing ``message_history`` (or ``conversation_id``) to the next run
continues the conversation's session, as a LangChain checkpointer does for a thread. Each run works
on its own copy of the adapter, so concurrent runs never share state; tool calls Pydantic AI runs in
parallel each record into the run's session and are merged. `session(conversation_id)` returns a
conversation's session token. Nothing is taken from the caller: the policy file and
`Firewall.from_config()` decide.

In log mode a verdict changes nothing the agent does, so the agent does not wait for it: the tool
result goes to the model at once and the same payload is judged in the background, with the window,
boundary and session captured when the tool returned. A conversation's background judgements run
one at a time, in call order, each on the session the previous one produced, so the verdicts and
the session are exactly those of a blocking run; different conversations are judged in parallel.
Decisions reach ``on_decision`` as they land, and ``flush()`` waits for what is still being judged.
``log_blocking=True`` makes log mode wait, as block mode always does.

An agent with a structured ``output_type`` cannot take the replacement text as its output: a
rejected request then ends the run with Pydantic AI's output error instead, and the model is still
never called.

Usage:
    firewall = Firewall.from_config("agent.yaml")
    agent = Agent(model, tools=tools, capabilities=[firewall.adapt_to("pydantic_ai")])
"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

try:
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.messages import (
        AudioUrl,
        BinaryContent,
        ImageUrl,
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolReturn,
        ToolReturnPart,
        UserPromptPart,
        VideoUrl,
    )
    from pydantic_core import to_jsonable_python
except ImportError as e:
    raise ImportError(
        "The Pydantic AI adapter needs Pydantic AI 2.x in your application: pip install pydantic-ai"
    ) from e

from ..session import Session
from ._shared import BoundaryInventory, Lanes, json_without_binary

HUMAN_TURN = {"name": "human turn", "kind": "request", "hook": "before_run"}


class HumanboundFirewallCapability(BoundaryInventory, AbstractCapability[Any]):
    """Guard every Pydantic AI boundary with a Firewall. Build it with
    `firewall.adapt_to("pydantic_ai")` and pass it in the agent's `capabilities`.

    Args:
        firewall: A configured `humanbound_firewall.Firewall`.
        tools: The agent's tools (functions, `Tool` objects or names), used only for the
            boundary inventory (`boundaries` / `report()`); screening does not need them.
        history_messages: How many of the most recent messages go to the firewall as the
            window alongside a payload. Earlier tool outputs are included, so a multi-hop
            chain (a page that only points at the next page) stays visible.
        history_chars: Longest window entry sent; longer ones are truncated. The payload
            being screened is never truncated.
        log_blocking: In log mode, wait for each verdict before the result reaches the agent
            (default False: judge in the background, since a logged verdict changes nothing).
        max_conversations: How many conversations' sessions are kept; the least recently used
            is forgotten first.
    """

    ADAPTER = "pydantic_ai"
    FRAMEWORK = "Pydantic AI"
    HUMAN_TURN = HUMAN_TURN
    TOOL_HOOK = "wrap_tool_execute"

    def __init__(
        self,
        firewall,
        *,
        tools: Iterable[Any] | None = None,
        history_messages: int = 12,
        history_chars: int = 2000,
        log_blocking: bool = False,
        max_conversations: int = 1024,
    ):
        self.firewall = firewall
        self._tool_names = [_tool_name(t) for t in (tools or [])]
        self._tool_descriptions = {_tool_name(t): _tool_description(t) for t in (tools or [])}
        self._history_messages = history_messages
        self._history_chars = history_chars
        self._log_blocking = log_blocking
        self._max_conversations = max_conversations
        # Shared by every run of this adapter:
        self._lanes = Lanes()
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._sessions_lock = threading.Lock()
        # This run's (set on the copy for_run returns):
        self._key = "default"
        self._session = Session.new()
        self._rejected: str | None = None

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every background (log mode) judgement has been made and reported to
        `on_decision`. Returns False if `timeout` seconds pass first."""
        return self._lanes.flush(timeout)

    def session(self, conversation_id: str) -> dict | None:
        """A conversation's session token (None for a conversation this adapter has not seen),
        with what background judgements have recorded so far."""
        with self._sessions_lock:
            token = self._sessions.get(conversation_id)
        background = self._lanes.session(conversation_id)
        if token is None and background is None:
            return None
        return token if background is None else Session.merge(token, background).to_json()

    # ------------------------------------------------------------------ run lifecycle
    async def for_run(self, ctx) -> HumanboundFirewallCapability:
        """A copy for this run: the run's conversation, its session and a rejected request belong
        to it. The stores and the background lanes stay shared."""
        run = copy.copy(self)
        run._key = str(ctx.conversation_id or ctx.run_id or "default")
        run._session = Session.from_json(self.session(run._key))
        run._rejected = None
        return run

    def _current(self) -> Session:
        """The run's session, joined with what background judgements recorded."""
        background = self._lanes.session(self._key)
        return self._session if background is None else Session.merge(self._session, background)

    def _record(self, session: Session) -> None:
        """Keep the run's session, and store it as the conversation's."""
        self._session = session
        with self._sessions_lock:
            self._sessions[self._key] = session.to_json()
            self._sessions.move_to_end(self._key)
            while len(self._sessions) > self._max_conversations:
                self._sessions.popitem(last=False)

    async def after_run(self, ctx, *, result):
        """The run is over: write back what background judgements recorded so far. Anything still
        being judged is in `session(conversation_id)` once it lands (or wait with `flush()`)."""
        background = self._lanes.session(self._key, release=True)
        if background is not None:
            self._record(Session.merge(self._session, background))
        return result

    # ------------------------------------------------------------------ request: the human turn
    async def before_run(self, ctx) -> None:
        request = _text(ctx.prompt)
        self._record(self._current().start_run(request or ""))
        if request is None or self.firewall.guard.mode_for("request") == "off":
            return
        window = self._window(ctx.messages)
        if self._background("request"):  # log mode: a request is never rejected, so do not wait
            self._submit(request, "request", window, HUMAN_TURN)
            return
        decision = await asyncio.to_thread(
            self.firewall.inspect,
            request,
            cls="request",
            window=window,
            session=self._current(),
            boundary=HUMAN_TURN,
        )
        self._record(decision.session)
        if decision.action == "reject":
            self._rejected = decision.replacement

    async def wrap_model_request(self, ctx, *, request_context, handler):
        if self._rejected is not None:  # the request was rejected: the reply is the firewall's
            return ModelResponse(parts=[TextPart(content=self._rejected)])
        return await handler(request_context)

    # ------------------------------------------------------------------ ingest / recall: tools
    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        result = await handler(args)
        payload = _text(result)
        if payload is None:  # no text to judge (an image or file alone)
            return result
        name = call.tool_name
        cls = self.firewall.config.class_of(name)
        # Everything the judge sees is fixed here, when the tool returned.
        window = self._window(ctx.messages)
        boundary = {
            **self._tool_boundary(name, tool_def.description or ""),
            "args": call.args_as_dict(),
            "tool_call_id": call.tool_call_id,
        }
        if self._background(cls):
            self._submit(payload, cls, window, boundary)
            return result
        # The firewall call is blocking I/O. Run it off the event loop: this run still waits
        # for the verdict before the result reaches the model, but other tasks keep running.
        decision = await asyncio.to_thread(
            self.firewall.inspect,
            payload,
            cls=cls,
            window=window,
            session=self._current(),
            boundary=boundary,
        )
        self._record(Session.merge(self._session, decision.session))  # parallel calls join
        return result if decision.action == "pass" else decision.replacement

    # ------------------------------------------------------------------ log mode
    def _background(self, cls: str) -> bool:
        """Log mode judges off the agent's path unless the caller asked it to wait."""
        return not self._log_blocking and self.firewall.guard.mode_for(cls) == "log"

    def _submit(self, payload, cls, window, boundary) -> None:
        snapshot = self._current().to_json()

        def judge(previous):
            session = snapshot if previous is None else Session.merge(snapshot, previous)
            return self.firewall.inspect(
                payload, cls=cls, window=window, session=session, boundary=boundary
            ).session

        self._lanes.submit(self._key, judge)

    # ------------------------------------------------------------------ helpers
    def _window(self, messages) -> list[dict]:
        """The conversation so far, as the judge's context. The system prompt is left out on
        purpose: it is the operator's trusted prompt, not history, and it would put the agent's
        own instructions in front of the judge."""
        window = []
        for message in messages:
            for part in message.parts:
                if isinstance(message, ModelRequest) and isinstance(part, UserPromptPart):
                    role, text = "user", _text(part.content)
                elif isinstance(part, ToolReturnPart):
                    role, text = "tool", _text(part.content)
                elif isinstance(message, ModelResponse) and isinstance(part, TextPart):
                    role, text = "assistant", part.content
                else:
                    continue
                if text:
                    window.append({"role": role, "content": text[: self._history_chars]})
        return window[-self._history_messages :]


def _tool_name(tool) -> str:
    if isinstance(tool, str):
        return tool
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", tool))


def _tool_description(tool) -> str:
    if isinstance(tool, str):
        return ""
    return str(getattr(tool, "description", None) or getattr(tool, "__doc__", None) or "").strip()


# The content that carries no text to judge: the model reads it as media. Everything else is
# screened, whatever its type — a tool may return a dict, a Pydantic model, a ToolReturn, a document
# — since the model reads the text inside it. Listing what to skip rather than what to read means a
# type added later is screened by default: over-screening is the safe failure.
_MEDIA = (ImageUrl, AudioUrl, VideoUrl)
_TEXT_MEDIA_TYPES = ("text/", "application/json", "application/xml", "application/yaml")


def _text(content) -> str | None:
    """The text a prompt or tool result carries.

    A string as it is; a `ToolReturn`'s return value and content; a list of user content joined
    with newlines (images, audio and video skipped; a text file read as text); anything else —
    a dict, a model, a document — as its JSON form, raw bytes dropped. None when nothing is left.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, ToolReturn):
        parts = [_text(content.return_value), _text(content.content)]
    elif isinstance(content, list | tuple):
        parts = [_text(item) for item in content]
    elif isinstance(content, _MEDIA):
        return None
    elif isinstance(content, BinaryContent):
        if content.is_image or content.is_audio or content.is_video:
            return None
        if content.media_type.startswith(_TEXT_MEDIA_TYPES):
            return content.data.decode("utf-8", errors="replace")
        return json_without_binary({"media_type": content.media_type, "id": content.identifier})
    else:
        try:
            return json_without_binary(to_jsonable_python(content, fallback=str))
        except (TypeError, ValueError):
            return str(content)
    texts = [p for p in parts if p]
    return "\n".join(texts) if texts else None
