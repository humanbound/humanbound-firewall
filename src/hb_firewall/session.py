"""Session management — sliding window of conversation turns."""

import uuid
from typing import List, Optional

from .models import Turn, EvalResult


class Session:
    """Manages conversation context for multi-turn evaluation.

    Keeps a sliding window of the last N turns so the judge can understand
    conversational flow and detect multi-turn attacks.
    """

    def __init__(self, firewall, window_size: int = 5):
        self.id = str(uuid.uuid4())
        self._firewall = firewall
        self._window_size = window_size
        self._turns: List[Turn] = []
        self._pending_user: Optional[str] = None

    @property
    def turns(self) -> List[Turn]:
        """Current conversation turns in the window."""
        return self._turns[-self._window_size:]

    @property
    def turn_count(self) -> int:
        """Total turns in this session (not just the window)."""
        return len(self._turns)

    def evaluate(
        self,
        user_prompt: str,
        agent_prompt: str = "",
    ) -> EvalResult:
        """Evaluate a user prompt within this session's context.

        Args:
            user_prompt: The user message to evaluate.
            agent_prompt: The agent's preceding message (if any).

        Returns:
            EvalResult with verdict, category, and explanation.
        """
        # If we have an agent prompt, update the last turn's assistant field
        if agent_prompt and self._turns:
            self._turns[-1].assistant = agent_prompt
        elif agent_prompt:
            self._turns.append(Turn(assistant=agent_prompt))

        # Evaluate with session context
        result = self._firewall.evaluate(
            user_prompt=user_prompt,
            agent_prompt=agent_prompt,
            session_turns=self.turns,
            session_id=self.id,
        )

        # Record this turn
        self._pending_user = user_prompt
        self._turns.append(Turn(user=user_prompt))

        return result

    def add_response(self, agent_response: str):
        """Record the agent's response for context in subsequent evaluations.

        Call this after your agent responds, so the next evaluation has
        the full conversation context.
        """
        if self._turns:
            self._turns[-1].assistant = agent_response

    def clear(self):
        """Clear session history."""
        self._turns.clear()
        self._pending_user = None
