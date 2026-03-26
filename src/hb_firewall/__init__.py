"""hb-firewall: Multi-tier firewall for AI agents.

Tier 0:   Input sanitization
Tier 1:   Basic attack detection (pre-trained, single-turn)
Tier 2:   Agent-specific classification (trained, multi-turn, 3+ turns)
Tier 3:   LLM-as-a-judge (handles uncertain cases)
"""

__version__ = "0.1.0"

from .firewall import Firewall, AttackDetector, AttackDetectorEnsemble
from .models import EvalResult, AgentConfig, Verdict, Category, Turn
from .llm import Provider, ProviderIntegration, ProviderName, get_llm_pinger, get_llm_streamer

try:
    from .hbfw import HBFW, load_model_class, save_hbfw, load_hbfw
except ImportError:
    HBFW = None
    load_model_class = None
    save_hbfw = None
    load_hbfw = None

__all__ = [
    "Firewall", "AttackDetector", "AttackDetectorEnsemble",
    "EvalResult", "AgentConfig", "Verdict", "Category", "Turn",
    "Provider", "ProviderIntegration", "ProviderName",
    "get_llm_pinger", "get_llm_streamer",
    "HBFW", "load_model_class", "save_hbfw", "load_hbfw",
]
