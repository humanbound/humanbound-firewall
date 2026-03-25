"""SetFit-based AgentClassifier for Tier 2 of hb-firewall.

Trains a SINGLE SetFit model (attack vs benign) using contrastive learning.
The attack detector trains the model. The benign detector reuses it
(same boundary, opposite interpretation).

Usage:
    hb firewall train --model detectors/setfit_classifier.py
"""

import os
import datetime
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    _ta.default_logdir = lambda: os.path.join(
        "runs", datetime.datetime.now().strftime("%b%d_%H-%M-%S"))

ST_MODEL = "all-MiniLM-L6-v2"
MAX_EXAMPLES = 60

# Shared model between attack and benign instances
_shared_model = None


class AgentClassifier:
    def __init__(self, name, model_name=ST_MODEL):
        self.name = name
        self.model_name = model_name
        self._model = None

    def train(self, texts, context=None):
        global _shared_model

        if self.name == "benign":
            # Benign detector reuses the model trained by attack detector
            if _shared_model is not None:
                self._model = _shared_model
            return

        # Attack detector trains the shared model
        from setfit import SetFitModel, Trainer, TrainingArguments
        from sentence_transformers import SentenceTransformer
        from datasets import Dataset

        pos = texts[:MAX_EXAMPLES]
        neg = (context or {}).get("all_benign_texts", [])[:MAX_EXAMPLES]

        if len(pos) < 5 or len(neg) < 5:
            return

        ds = Dataset.from_dict({
            "text": pos + neg,
            "label": [1] * len(pos) + [0] * len(neg),
        })

        from sklearn.linear_model import LogisticRegression
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
            # Class 1 = attack. High prob = attack detected.
            return prob > 0.5, prob
        else:
            # Class 0 = benign. Low attack prob = benign detected.
            benign_prob = 1.0 - prob
            return benign_prob > 0.5, benign_prob

    def export_weights(self):
        import numpy as np, pickle
        if self._model is None:
            return {}
        # Only attack detector saves — benign reuses
        if self.name == "attack":
            return {f"setfit_model": np.void(pickle.dumps(self._model))}
        return {}

    def load_weights(self, weights):
        import pickle
        if "setfit_model" in weights:
            self._model = pickle.loads(bytes(weights["setfit_model"]))
