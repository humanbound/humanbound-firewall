# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""LangChain adapter: what `firewall.adapt_to("langchain")` returns.

An `AgentMiddleware` that attaches every boundary LangChain exposes, calls `Firewall.inspect()`
there, enforces the Decision, and carries the Session in the graph state:

- ``before_agent``   starts the run: the session is read from the state (or created), the run
                     counter advances and the user's request is pinned;
- ``before_model``   the human turn entering the model is a ``request``. A rejected request ends
                     the run with the replacement text as the agent's reply;
- ``wrap_tool_call`` a tool's return value is ``ingest`` — or ``recall`` when the policy's
                     ``tools:`` block says the tool returns our own records. A withheld result is
                     replaced by the template text; the tool_call_id is preserved.

The session lives under ``humanbound_session`` in the agent state, so a checkpointer persists it
across the runs of a thread, and parallel tool calls in one step merge their branches. Nothing is
taken from the caller: the policy file and `Firewall.from_config()` decide.

In log mode a verdict changes nothing the agent does, so the agent does not wait for it: the tool
result goes to the model at once and the same payload is judged in the background, with the window,
boundary and session captured when the tool returned. A thread's background judgements run one at a
time, in call order, each on the session the previous one produced, so the verdicts and the session
are exactly those of a blocking run; different threads are judged in parallel. Decisions reach
``on_decision`` as they land, the session is written back to the state at the next model turn and
at the end of the run, and ``flush()`` waits for what is still being judged. ``log_blocking=True``
makes log mode wait, as block mode always does.

Usage:
    firewall = Firewall.from_config("agent.yaml")
    agent = create_agent(model, tools, middleware=[firewall.adapt_to("langchain")])
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Annotated, Any

try:
    from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.types import Command
    from typing_extensions import NotRequired
except ImportError as e:
    raise ImportError(
        "The LangChain adapter needs LangChain 1.x in your application: pip install langchain"
    ) from e

from ..session import Session
from ._shared import BoundaryInventory, Lanes, json_without_binary

logger = logging.getLogger(__name__)

SESSION_KEY = "humanbound_session"
HUMAN_TURN = {"name": "human turn", "kind": "request", "hook": "before_model"}

# What goes into the conversation window sent as context with each screened payload.
# SystemMessage is left out on purpose: it is the operator's trusted prompt, not history, and
# it would put the agent's own instructions in front of the judge.
_ROLES = ((HumanMessage, "user"), (AIMessage, "assistant"), (ToolMessage, "tool"))


def _merge_tokens(current, incoming):
    """State reducer: branches that recorded in parallel are joined, never overwritten."""
    return Session.merge(current, incoming).to_json()


def _thread_key() -> str:
    """The LangGraph thread a call belongs to: one background lane per thread. Without a thread id
    (no checkpointer, or a call outside a graph) calls share one lane: still in order, only slower."""
    try:
        from langgraph.config import get_config

        return str((get_config().get("configurable") or {}).get("thread_id") or "default")
    except Exception:  # outside a runnable context
        return "default"


class FirewallAgentState(AgentState):
    humanbound_session: NotRequired[Annotated[dict, _merge_tokens]]


class HumanboundFirewallMiddleware(BoundaryInventory, AgentMiddleware):
    """Guard every LangChain boundary with a Firewall. Build it with `firewall.adapt_to("langchain")`.

    Args:
        firewall: A configured `humanbound_firewall.Firewall`.
        tools: The agent's tools (objects or names), used only for the boundary inventory
            (`boundaries` / `report()`); screening does not need them.
        history_messages: How many of the most recent messages go to the firewall as the
            window alongside a payload. Earlier tool outputs are included, so a multi-hop
            chain (a page that only points at the next page) stays visible.
        history_chars: Longest window entry sent; longer ones are truncated. The payload
            being screened is never truncated.
        log_blocking: In log mode, wait for each verdict before the result reaches the agent
            (default False: judge in the background, since a logged verdict changes nothing).
    """

    state_schema = FirewallAgentState
    ADAPTER = "langchain"
    FRAMEWORK = "LangChain"
    HUMAN_TURN = HUMAN_TURN
    TOOL_HOOK = "wrap_tool_call"

    def __init__(
        self,
        firewall,
        *,
        tools: Iterable[Any] | None = None,
        history_messages: int = 12,
        history_chars: int = 2000,
        log_blocking: bool = False,
    ):
        super().__init__()
        self.firewall = firewall
        self._tool_names = [_tool_name(t) for t in (tools or [])]
        self._tool_descriptions = {
            _tool_name(t): str(getattr(t, "description", "") or "") for t in (tools or [])
        }
        self._history_messages = history_messages
        self._history_chars = history_chars
        self._log_blocking = log_blocking
        self._lanes = Lanes()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every background (log mode) judgement has been made and reported to
        `on_decision`. Returns False if `timeout` seconds pass first."""
        return self._lanes.flush(timeout)

    def _session_in(self, state):
        """The thread's session: the state's, joined with what background judgements recorded."""
        background = self._lanes.session(_thread_key())
        token = state.get(SESSION_KEY)
        return token if background is None else Session.merge(token, background).to_json()

    def _write_back(self, state, *, release=False):
        background = self._lanes.session(_thread_key(), release=release)
        if background is None:
            return None
        return {SESSION_KEY: Session.merge(state.get(SESSION_KEY), background).to_json()}

    # ------------------------------------------------------------------ run lifecycle
    def before_agent(self, state, runtime):
        request = _last_human(state.get("messages", []))
        session = Session.from_json(state.get(SESSION_KEY)).start_run(request)
        return {SESSION_KEY: session.to_json()}

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    # ------------------------------------------------------------------ request: the human turn
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        messages = state.get("messages", [])
        written = self._write_back(state)  # what background judgements recorded since last time
        if not messages or not isinstance(messages[-1], HumanMessage):
            return written  # a later model call in the same run: the turn was judged already
        if self.firewall.guard.mode_for("request") == "off":
            return written
        text = _text(messages[-1].content)
        if text is None:  # no text to judge (an image alone)
            return written
        window = self._window(messages[:-1])
        if self._background("request"):  # log mode: a request is never rejected, so do not wait
            state_session = state.get(SESSION_KEY)

            def judge(previous):
                session = (
                    state_session if previous is None else Session.merge(state_session, previous)
                )
                return self.firewall.inspect(
                    text, cls="request", window=window, session=session, boundary=HUMAN_TURN
                ).session

            self._lanes.submit(_thread_key(), judge)
            return written
        decision = self.firewall.inspect(
            text,
            cls="request",
            window=window,
            session=self._session_in(state),
            boundary=HUMAN_TURN,
        )
        update: dict[str, Any] = {SESSION_KEY: decision.session.to_json()}
        if decision.action == "reject":
            update["messages"] = [AIMessage(content=decision.replacement)]
            update["jump_to"] = "end"
        return update

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return await asyncio.to_thread(self.before_model, state, runtime)

    def after_agent(self, state, runtime):
        """The run is over: write back what background judgements recorded so far. Anything still
        being judged is written back at the thread's next model turn (or wait with `flush()`)."""
        return self._write_back(state, release=True)

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)

    # ------------------------------------------------------------------ ingest / recall: tools
    def wrap_tool_call(self, request, handler):
        return self._screen(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if self._in_background(request):
            return self._screen(request, result)  # hands the judgement off; returns at once
        # The firewall call is blocking I/O. Run it off the event loop: this run still waits
        # for the verdict before the result reaches the agent, but other tasks keep running.
        return await asyncio.to_thread(self._screen, request, result)

    def _in_background(self, request) -> bool:
        return self._background(self.firewall.config.class_of(request.tool_call.get("name", "")))

    def _background(self, cls: str) -> bool:
        """Log mode judges off the agent's path unless the caller asked it to wait."""
        return not self._log_blocking and self.firewall.guard.mode_for(cls) == "log"

    def _screen(self, request, result):
        if not isinstance(result, ToolMessage):
            return result
        payload = _text(result.content)
        if payload is None:  # no text to judge (an image or file alone)
            return result
        name = request.tool_call.get("name", "")
        cls = self.firewall.config.class_of(name)
        description = str(getattr(getattr(request, "tool", None), "description", "") or "")
        # Everything the judge sees is fixed here, when the tool returned.
        window = self._window(request.state.get("messages", []))
        boundary = {
            **self._tool_boundary(name, description),
            "args": request.tool_call.get("args") or {},
            "tool_call_id": request.tool_call.get("id"),
        }
        if self._in_background(request):
            state_session = request.state.get(SESSION_KEY)

            def judge(previous):
                session = (
                    state_session if previous is None else Session.merge(state_session, previous)
                )
                return self.firewall.inspect(
                    payload, cls=cls, window=window, session=session, boundary=boundary
                ).session

            self._lanes.submit(_thread_key(), judge)
            return result
        decision = self.firewall.inspect(
            payload,
            cls=cls,
            window=window,
            session=self._session_in(request.state),
            boundary=boundary,
        )
        message = result
        if decision.action != "pass":
            message = result.model_copy(update={"content": decision.replacement})
        return Command(update={"messages": [message], SESSION_KEY: decision.session.to_json()})

    # ------------------------------------------------------------------ helpers
    def _window(self, messages) -> list[dict]:
        window = []
        for message in messages:
            for kind, role in _ROLES:
                if isinstance(message, kind):
                    text = _text(message.content)
                    if text:
                        window.append({"role": role, "content": text[: self._history_chars]})
        return window[-self._history_messages :]


def _tool_name(tool) -> str:
    return tool if isinstance(tool, str) else str(getattr(tool, "name", tool))


def _last_human(messages) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            text = _text(message.content)
            if text is not None:
                return text
    return ""


# The block types that carry no text to judge: the model reads them as media, and their payload
# is bytes or a URL. Everything else is screened, whatever its type — LangChain keeps document,
# search_result, json, custom_tool_call_output and file blocks as tool content, a human turn can
# carry text-plain, and each one reaches the model as text. Listing what to skip rather than what
# to read means a block type added later is screened by default: over-screening is the safe
# failure, a block that slips past is not.
_MEDIA_BLOCKS = frozenset({"image", "image_url", "audio", "video"})


def _text(content) -> str | None:
    """The text a message carries, whether its content is a string or a list of content blocks.

    LangChain allows both (`str | list[str | dict]`): multimodal messages and MCP tool results with
    several parts or an image arrive as a list. Plain strings and `{"type": "text"}` blocks are
    joined with newlines, so text split across blocks is judged as one payload. Any other block —
    a document, a search result, JSON, a custom tool's output — is judged as its JSON form, since
    the model reads the text inside it. Pure media blocks are skipped. None when nothing is left.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in _MEDIA_BLOCKS:
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(_block_json(block))
    parts = [p for p in parts if p]
    return "\n".join(parts) if parts else None


def _block_json(block: dict) -> str:
    """A non-text block as JSON for the judge, with raw media bytes dropped at any depth."""
    return json_without_binary(block)
