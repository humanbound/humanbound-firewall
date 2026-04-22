# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""SetFit-based AgentClassifier for Tier 2 of humanbound-firewall.

Trains a single SetFit model (attack vs benign) using contrastive learning
on curated examples from adversarial + QA test logs. The attack detector
trains the model; the benign detector reuses it (opposite interpretation).

Validation is handled by the orchestrator via conversation replay.

Requires: setfit==1.1.3, sentence-transformers>=2.2, scikit-learn>=1.3

Usage:
    hb firewall train --model detectors/setfit_classifier.py
"""

ST_MODEL = "all-MiniLM-L6-v2"
MAX_EXAMPLES = 60

# Workaround: the orchestrator creates two AgentClassifier instances (attack + benign)
# expecting two independent models. SetFit trains a single binary model that handles
# both roles — the benign instance reuses the attack instance's model with the
# opposite probability interpretation. The shared state avoids training twice.
_shared_model = None


def _patch_compat():
    """Pin compatibility: setfit 1.1.3 + transformers 5.x."""
    import transformers.training_args as ta

    if not hasattr(ta, "default_logdir"):
        import datetime
        import os

        ta.default_logdir = lambda: os.path.join(
            "runs", datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        )


class AgentClassifier:
    def __init__(self, name, model_name=ST_MODEL):
        self.name = name
        self.model_name = model_name
        self._model = None
        self._model_dir = None

    def train(self, texts, context=None):
        global _shared_model

        if self.name == "benign":
            if _shared_model is not None:
                self._model = _shared_model
            return

        _patch_compat()

        from datasets import Dataset
        from sentence_transformers import SentenceTransformer
        from setfit import SetFitModel, Trainer, TrainingArguments
        from sklearn.linear_model import LogisticRegression

        pos = texts[:MAX_EXAMPLES]
        neg = (context or {}).get("all_benign_texts", [])[:MAX_EXAMPLES]

        if len(pos) < 5 or len(neg) < 5:
            return

        ds = Dataset.from_dict(
            {
                "text": pos + neg,
                "label": [1] * len(pos) + [0] * len(neg),
            }
        )

        st = SentenceTransformer(self.model_name)
        self._model = SetFitModel(model_body=st, model_head=LogisticRegression())

        trainer = Trainer(
            model=self._model,
            args=TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20),
            train_dataset=ds,
        )
        trainer.train()

        _shared_model = self._model

    def predict(self, text, context=""):
        if self._model is None:
            return False, 0.0

        prob = float(self._model.predict_proba([text])[0][1])

        if self.name == "attack":
            return prob > 0.5, prob
        else:
            benign_prob = 1.0 - prob
            return benign_prob > 0.5, benign_prob

    def export_weights(self):
        if self._model is None:
            return {}
        if self.name != "attack":
            return {}

        import json
        import os
        import tempfile

        import numpy as np

        # Save SetFit model using its native format (safetensors)
        self._model_dir = tempfile.mkdtemp(prefix="hbfw_setfit_")
        self._model.save_pretrained(self._model_dir)

        # Read all files from the saved directory into a dict
        model_files = {}
        for root, dirs, files in os.walk(self._model_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self._model_dir)
                with open(fpath, "rb") as f:
                    model_files[rel] = f.read()

        # Store file manifest + binary data
        manifest = list(model_files.keys())
        result = {
            "setfit_manifest": json.dumps(manifest),
            "setfit_model_name": self.model_name,
        }

        # Store each file as numpy void (binary safe for npz)
        for rel_path in manifest:
            safe_key = "setfit_file_" + rel_path.replace("/", "__").replace(".", "_")
            result[safe_key] = np.void(model_files[rel_path])

        return result

    def load_weights(self, weights):
        if "setfit_manifest" not in weights:
            return

        _patch_compat()

        import json
        import os
        import tempfile

        from setfit import SetFitModel

        manifest = json.loads(str(weights["setfit_manifest"]))
        model_dir = tempfile.mkdtemp(prefix="hbfw_setfit_load_")

        for rel_path in manifest:
            safe_key = "setfit_file_" + rel_path.replace("/", "__").replace(".", "_")
            if safe_key not in weights:
                continue

            fpath = os.path.join(model_dir, rel_path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "wb") as f:
                f.write(bytes(weights[safe_key]))

        self._model = SetFitModel.from_pretrained(model_dir)
