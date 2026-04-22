# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""Example AgentClassifier scaffold for humanbound-firewall Tier 2.

Copy this file and implement your own training and inference logic.
The orchestrator creates two instances — one for attack detection,
one for benign detection — and handles everything else.

Usage:
    hb firewall train --model detectors/example_classifier.py

The orchestrator provides:
    texts:   curated training texts (attack turns or benign turns with context)
    context: {"permitted_intents": [...], "restricted_intents": [...],
              "all_attack_texts": [...], "all_benign_texts": [...]}

Your classifier decides how to process them (embeddings, fine-tuning, NLI, etc).
"""


class AgentClassifier:
    def __init__(self, name):
        """Called twice: name="attack" and name="benign".

        The attack instance is trained on curated adversarial turns.
        The benign instance is trained on curated QA turns.
        """
        self.name = name
        # Initialize your model here

    def train(self, texts, context=None):
        """Train on curated texts from platform logs.

        Args:
            texts: list of strings — each is a user turn with up to 3 turns
                   of conversational context. For the attack instance, these
                   are turns from failed adversarial conversations (stratified
                   by fail_category). For the benign instance, these are turns
                   from passed QA conversations (stratified by gen_category).
            context: dict with additional data:
                - permitted_intents: list of permitted intent descriptions
                - restricted_intents: list of restricted intent descriptions
                - all_attack_texts: full (uncurated) attack turn list
                - all_benign_texts: full (uncurated) benign turn list
        """
        raise NotImplementedError("Implement your training logic")

    def predict(self, text, context=""):
        """Classify a single text input.

        Args:
            text: user turn with conversational context, or isolated user message.
                  Called twice per classification — once with full context,
                  once with the last user turn only.

        Returns:
            (is_match, score): bool and float confidence score.
            For the attack instance: is_match=True means attack detected.
            For the benign instance: is_match=True means benign detected.
        """
        raise NotImplementedError("Implement your inference logic")

    def export_weights(self):
        """Export model state for serialization into .hbfw file.

        Returns:
            dict of numpy arrays (or numpy-compatible values).
            Keys should be prefixed with self.name to avoid collisions
            between attack and benign instances.
        """
        raise NotImplementedError("Implement weight export")

    def load_weights(self, weights):
        """Restore model state from .hbfw file.

        Args:
            weights: dict of numpy arrays, as returned by export_weights().
        """
        raise NotImplementedError("Implement weight loading")
