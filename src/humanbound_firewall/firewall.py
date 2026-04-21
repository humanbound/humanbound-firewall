# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2024-2026 Humanbound
"""Main Firewall class — multi-tier evaluation of user prompts.

Tier 0:   Input sanitization (instant) — blocks non-visible control characters
Tier 1:   Basic attack detection ensemble (models + APIs, parallel, configurable consensus)
Tier 2.1: Advanced attack detection (trained classifier, catches what Tier 1 misses)
Tier 2.2: Benign request detection (trained classifier, fast-tracks legitimate requests)
Tier 3:   LLM-as-a-judge (deep contextual analysis for uncertain cases)
"""

import re
import time
import threading
import concurrent.futures
import requests
from pathlib import Path
from typing import List, Optional, Union

from .models import AgentConfig, EvalResult, Turn, Verdict, Category, VERDICT_MAP
from .config import load_config
from .judge import build_system_prompt
from .cache import PromptCache
from .metrics import Metrics
from .llm import Provider, ProviderIntegration, ProviderName, get_llm_streamer

_INVISIBLE_CHARS = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f'
    r'\u200b-\u200f\u2028-\u2029\u202a-\u202e'
    r'\u2060-\u2064\ufeff\ufff9-\ufffb]'
)


# ---------------------------------------------------------------------------
# Attack detector interface (Tier 1)
# ---------------------------------------------------------------------------

class AttackDetector:
    """Single attack detector — local model or API endpoint."""

    def __init__(self, config: dict):
        self.name = config.get("name", "detector")
        self._model_name = config.get("model")
        self._endpoint = config.get("endpoint")
        self._method = config.get("method", "POST").upper()
        self._headers = config.get("headers", {})
        self._payload_template = config.get("payload", {})
        self._response_path = config.get("response_path", "")
        self._pipe = None

    def score(self, prompt: str, conversation: str = "") -> float:
        """Return attack probability 0-1."""
        if self._model_name:
            return self._score_local(prompt)
        elif self._endpoint:
            return self._score_api(prompt, conversation)
        return 0.0

    def _score_local(self, prompt: str) -> float:
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "Tier 1 local detectors require the [tier1] extra. "
                    "Install with: pip install humanbound-firewall[tier1]"
                ) from e
            self._pipe = pipeline("text-classification", model=self._model_name,
                                   truncation=True, max_length=512)
        result = self._pipe(prompt)[0]
        if result["label"] in ("INJECTION", "LABEL_1", "positive", "1"):
            return result["score"]
        return 1.0 - result["score"]

    def _score_api(self, prompt: str, conversation: str) -> float:
        payload = {}
        for k, v in self._payload_template.items():
            if isinstance(v, str):
                v = v.replace("$PROMPT", prompt).replace("$CONVERSATION", conversation)
            payload[k] = v

        try:
            resp = requests.request(
                self._method, self._endpoint,
                headers=self._headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return self._extract_score(data)
        except Exception:
            pass
        return 0.0

    def _extract_score(self, data: dict) -> float:
        """Extract score from API response using dot-notation path."""
        value = data
        for key in self._response_path.split("."):
            if isinstance(value, dict):
                value = value.get(key, 0)
            else:
                return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value) if value else 0.0


class AttackDetectorEnsemble:
    """Ensemble of attack detectors with configurable consensus."""

    def __init__(self, detectors: list[AttackDetector], consensus: int = 1):
        self.detectors = detectors
        self.consensus = consensus

    def evaluate(self, prompt: str, conversation: str = "") -> tuple[bool, float]:
        """Run all detectors in parallel. Returns (is_attack, max_score)."""
        if not self.detectors:
            return False, 0.0

        scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.detectors)) as pool:
            futures = {pool.submit(d.score, prompt, conversation): d
                       for d in self.detectors}

            attack_count = 0
            for future in concurrent.futures.as_completed(futures):
                score = future.result()
                scores.append(score)
                if score > 0.5:
                    attack_count += 1
                    if attack_count >= self.consensus:
                        # Early exit — consensus reached
                        pool.shutdown(wait=False, cancel_futures=True)
                        return True, max(scores)

        max_score = max(scores) if scores else 0.0
        attack_count = sum(1 for s in scores if s > 0.5)
        return attack_count >= self.consensus, max_score


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

class Firewall:
    """Multi-tier firewall for AI agents.

    Tier 0:   Input sanitization
    Tier 1:   Basic attack detection ensemble (configurable)
    Tier 2.1: Advanced attack detection (trained, .hbfw)
    Tier 2.2: Benign request detection (trained, .hbfw)
    Tier 3:   LLM-as-a-judge
    """

    def __init__(self, config: AgentConfig, streamer=None,
                 ensemble: AttackDetectorEnsemble = None,
                 scope_classifier=None):
        self._config = config
        self._streamer = streamer
        self._ensemble = ensemble
        self._scope_classifier = scope_classifier
        self._cache = PromptCache()
        self.metrics = Metrics()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        provider: Optional[Provider] = None,
        model_path: Optional[Union[str, Path]] = None,
        detector_script: Optional[Union[str, Path]] = None,
        attack_detectors: Optional[list[dict]] = None,
        consensus: int = 1,
    ) -> "Firewall":
        """Create a Firewall from configuration.

        Args:
            config_path: Path to agent.yaml.
            provider: LLM provider for Tier 3 judge.
            model_path: Path to .hbfw file for Tier 2 classifiers.
            detector_script: Path to AgentClassifier script (required if model_path is set).
            attack_detectors: List of detector configs for Tier 1.
            consensus: Minimum detectors to agree for BLOCK.
        """
        config = load_config(config_path)

        # Tier 3: LLM judge
        streamer = None
        if provider is not None:
            streamer = get_llm_streamer(provider)
        else:
            try:
                provider = _provider_from_env()
                streamer = get_llm_streamer(provider)
            except ValueError:
                pass

        # Tier 1: Attack detection ensemble
        ensemble = None
        if attack_detectors:
            detectors = [AttackDetector(cfg) for cfg in attack_detectors]
            ensemble = AttackDetectorEnsemble(detectors, consensus=consensus)

        # Tier 2: Trained classifiers
        scope_classifier = None
        if model_path is not None:
            from .hbfw import HBFW, load_model_class, load_hbfw

            hbfw_config, hbfw_weights = load_hbfw(str(model_path))

            if not detector_script:
                raise ValueError(
                    "Tier 2 model requires a detector script. "
                    "Pass detector_script= to Firewall.from_config().")

            detector_cls = load_model_class(str(detector_script))
            scope_classifier = HBFW(
                attack_detector=detector_cls("attack"),
                benign_detector=detector_cls("benign"))
            scope_classifier.load(hbfw_config, hbfw_weights)

        return cls(config=config, streamer=streamer, ensemble=ensemble,
                   scope_classifier=scope_classifier)

    def evaluate(
        self,
        user_prompt_or_conversation,
        agent_prompt: str = "",
        session_turns: Optional[List[Turn]] = None,
        session_id: str = "",
        timeout: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate a user prompt through all tiers.

        Accepts either:
            fw.evaluate("user prompt", session_turns=[...])

        Or an OpenAI-format conversation (last user message is evaluated):
            fw.evaluate([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "prompt to evaluate"},
            ])
        """
        # Parse input: conversation list or single prompt
        if isinstance(user_prompt_or_conversation, list):
            user_prompt, session_turns, agent_prompt = self._parse_conversation(
                user_prompt_or_conversation)
        else:
            user_prompt = user_prompt_or_conversation

        if self._config.mode == "passthrough":
            return EvalResult(
                verdict=Verdict.PASS, category=Category.NONE,
                explanation="Passthrough mode.", latency_ms=0,
                session_id=session_id, prompt=user_prompt,
            )

        timeout = timeout or self._config.timeout
        t_start = time.time()

        # --- Tier 0: Sanitization ---
        if _INVISIBLE_CHARS.search(user_prompt):
            return self._result(Verdict.BLOCK, Category.VIOLATION,
                                "Input contains non-visible control characters.",
                                t_start, session_id, user_prompt, tier=0)

        # --- Tier 1: Basic attack detection ensemble ---
        if self._ensemble:
            conversation_text = self._format_session(session_turns, user_prompt)
            is_attack, score = self._ensemble.evaluate(user_prompt, conversation_text)
            if is_attack:
                return self._result(Verdict.BLOCK, Category.VIOLATION,
                                    f"Tier 1: attack detected (p={score:.2f})",
                                    t_start, session_id, user_prompt, tier=1,
                                    attack_probability=score)

        # --- Tier 2: Agent-specific classifiers (requires conversation context) ---
        n_turns = len(session_turns) if session_turns else 0
        if self._scope_classifier and n_turns >= self._config.tier2_min_turns:
            conversation = self._build_conversation(session_turns, user_prompt, agent_prompt)
            result = self._scope_classifier.classify(conversation)
            decision = result.get("decision")

            if decision == "BLOCK":
                return self._result(Verdict.BLOCK, Category.VIOLATION,
                                    f"Tier 2.1: attack detected",
                                    t_start, session_id, user_prompt, tier=2,
                                    attack_probability=result.get("attack_probability", 0))

            if decision == "ALLOW":
                return self._result(Verdict.PASS, Category.NONE,
                                    f"Tier 2.2: benign request",
                                    t_start, session_id, user_prompt, tier=2)

            # ESCALATE (uncertain) → fall through to Tier 3

        # --- Tier 3: LLM judge ---
        if self._streamer:
            return self._run_judge(user_prompt, agent_prompt, session_turns,
                                    session_id, timeout, t_start)

        # No Tier 3 available — return ESCALATE
        return self._result(Verdict.REVIEW, Category.UNCERTAIN,
                            "No LLM judge configured.", t_start,
                            session_id, user_prompt, tier=3)

    def _parse_conversation(self, messages: list) -> tuple:
        """Parse OpenAI-format conversation into (user_prompt, session_turns, agent_prompt).

        Accepts: [{"role": "user"|"assistant", "content": "..."}]
        The last user message is the prompt being evaluated.
        """
        turns = []
        current_user = ""
        current_assistant = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                if current_user:
                    turns.append(Turn(user=current_user, assistant=current_assistant))
                    current_assistant = ""
                current_user = content
            elif role == "assistant":
                current_assistant = content

        # Last user message = prompt being evaluated
        user_prompt = current_user or ""
        agent_prompt = ""

        # Everything before the last user message = session turns
        session_turns = turns if turns else None

        return user_prompt, session_turns, agent_prompt

    def _build_conversation(self, session_turns, user_prompt, agent_prompt):
        """Build conversation list for scope classifier."""
        conversation = []
        if session_turns:
            conversation = [{"u": t.user, "a": t.assistant} for t in session_turns]
        conversation.append({"u": user_prompt, "a": agent_prompt or ""})
        return conversation

    def _format_session(self, session_turns, user_prompt):
        """Format session as text for API-based detectors."""
        parts = []
        if session_turns:
            for t in session_turns[-3:]:
                if t.user:
                    parts.append(f"User: {t.user}")
                if t.assistant:
                    parts.append(f"Agent: {t.assistant}")
        parts.append(f"User: {user_prompt}")
        return "\n".join(parts)

    def _result(self, verdict, category, explanation, t_start, session_id,
                prompt, tier=0, attack_probability=0.0):
        latency = int((time.time() - t_start) * 1000)
        self.metrics.record(verdict.value, category.value, latency)
        return EvalResult(
            verdict=verdict, category=category, explanation=explanation,
            latency_ms=latency, session_id=session_id, prompt=prompt,
            tier=tier, attack_probability=attack_probability,
        )

    def _run_judge(self, user_prompt, agent_prompt, session_turns,
                   session_id, timeout, t_start):
        """Run Tier 3 LLM judge with streaming and timeout."""
        try:
            base_prompt = self._cache.get_or_build(self._config)
            system_prompt = (build_system_prompt(self._config, session_turns=session_turns)
                              if session_turns else base_prompt)
            if agent_prompt:
                system_prompt += f"\n## PROMPT TO WHICH THE USER RESPONDS:\n{agent_prompt}\n"

            result = self._stream_judge(system_prompt, user_prompt, timeout, session_id)
            latency = int((time.time() - t_start) * 1000)
            result.latency_ms = latency
            result.prompt = user_prompt
            result.session_id = session_id
            result.tier = 3
            self.metrics.record(result.verdict.value, result.category.value, latency)
            return result

        except TimeoutError:
            return self._result(Verdict.REVIEW, Category.UNCERTAIN,
                                "Evaluation timed out.", t_start,
                                session_id, user_prompt, tier=3)
        except Exception as e:
            return self._result(Verdict.REVIEW, Category.UNCERTAIN,
                                f"Evaluation error: {str(e)[:200]}", t_start,
                                session_id, user_prompt, tier=3)

    def _stream_judge(self, system_prompt, user_prompt, timeout, session_id):
        """Stream LLM response and extract verdict from first token."""
        from .firewall_judge import stream_and_extract_verdict
        return stream_and_extract_verdict(
            self._streamer, system_prompt, user_prompt, timeout, session_id)

    def reload_config(self, config_path: Union[str, Path]):
        self._config = load_config(config_path)
        self._cache.invalidate()

    @property
    def config(self) -> AgentConfig:
        return self._config


# ---------------------------------------------------------------------------
# Streaming judge extraction (extracted for testability)
# ---------------------------------------------------------------------------

def _extract_token(chunk) -> Optional[str]:
    """Extract text from streaming chunk (handles OpenAI, Anthropic, Gemini formats)."""
    if hasattr(chunk, "choices") and chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
        return None
    if hasattr(chunk, "type"):
        if chunk.type == "content_block_delta" and hasattr(chunk, "delta"):
            return getattr(chunk.delta, "text", None)
        return None
    if hasattr(chunk, "text"):
        return chunk.text
    if isinstance(chunk, str):
        return chunk
    return None


def _env(new_key: str, legacy_key: str, default=None):
    """Read an env var, preferring the `HUMANBOUND_FIREWALL_*` name.

    Falls back to the legacy `HB_FIREWALL_*` name with a one-time
    DeprecationWarning. The legacy names are removed in 0.3.
    """
    import os
    import warnings

    value = os.environ.get(new_key)
    if value is not None:
        return value
    legacy_value = os.environ.get(legacy_key)
    if legacy_value is not None:
        warnings.warn(
            f"Environment variable {legacy_key} is deprecated; "
            f"use {new_key} instead. "
            "The legacy HB_FIREWALL_* names will be removed in 0.3.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy_value
    return default


def _provider_from_env() -> Provider:
    """Build a Provider from HUMANBOUND_FIREWALL_* environment variables.

    Legacy HB_FIREWALL_* names still work for 0.2.x (with a DeprecationWarning);
    they are removed in 0.3.
    """
    provider_name = (_env(
        "HUMANBOUND_FIREWALL_PROVIDER", "HB_FIREWALL_PROVIDER", ""
    ) or "").lower()
    api_key = _env("HUMANBOUND_FIREWALL_API_KEY", "HB_FIREWALL_API_KEY", "")

    if not api_key:
        raise ValueError(
            "No LLM provider configured. "
            "Set HUMANBOUND_FIREWALL_PROVIDER and HUMANBOUND_FIREWALL_API_KEY."
        )

    defaults = {
        "openai": ("gpt-4o-mini", ProviderName.OPENAI),
        "azureopenai": ("gpt-4o-mini", ProviderName.AZURE_OPENAI),
        "claude": ("claude-sonnet-4-6-20250514", ProviderName.CLAUDE),
        "gemini": ("gemini-pro", ProviderName.GEMINI),
    }

    if provider_name not in defaults:
        raise ValueError(f"Unsupported provider: '{provider_name}'.")

    default_model, name_enum = defaults[provider_name]

    return Provider(
        name=name_enum,
        integration=ProviderIntegration(
            api_key=api_key,
            model=_env("HUMANBOUND_FIREWALL_MODEL", "HB_FIREWALL_MODEL", default_model),
            endpoint=_env("HUMANBOUND_FIREWALL_ENDPOINT", "HB_FIREWALL_ENDPOINT"),
            api_version=_env("HUMANBOUND_FIREWALL_API_VERSION", "HB_FIREWALL_API_VERSION"),
        ),
    )
