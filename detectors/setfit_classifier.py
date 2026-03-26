"""SetFit-based AgentClassifier for Tier 2 of hb-firewall.

Trains a single SetFit model (attack vs benign) using contrastive learning
on curated examples from adversarial + QA test logs. The attack detector
trains the model; the benign detector reuses it (opposite interpretation).

Training includes 80/20 stratified split with validation metrics
(precision, recall, F1) reported and stored in the .hbfw config.

Requires: setfit==1.1.3, sentence-transformers>=2.2, scikit-learn>=1.3

Usage:
    hb firewall train --model detectors/setfit_classifier.py
"""

ST_MODEL = "all-MiniLM-L6-v2"
MAX_EXAMPLES = 60
VAL_SPLIT = 0.2

# Workaround: the orchestrator creates two AgentClassifier instances (attack + benign)
# expecting two independent models. SetFit trains a single binary model that handles
# both roles — the benign instance reuses the attack instance's model with the
# opposite probability interpretation. The shared state avoids training twice.
_shared_model = None
_shared_metrics = None


def _patch_compat():
    """Pin compatibility: setfit 1.1.3 + transformers 5.x."""
    import transformers.training_args as ta
    if not hasattr(ta, "default_logdir"):
        import os, datetime
        ta.default_logdir = lambda: os.path.join(
            "runs", datetime.datetime.now().strftime("%b%d_%H-%M-%S"))


class AgentClassifier:
    def __init__(self, name, model_name=ST_MODEL):
        self.name = name
        self.model_name = model_name
        self._model = None
        self._model_dir = None
        self.metrics = None

    def train(self, texts, context=None):
        global _shared_model, _shared_metrics

        if self.name == "benign":
            if _shared_model is not None:
                self._model = _shared_model
                self.metrics = _shared_metrics
            return

        _patch_compat()

        from setfit import SetFitModel, Trainer, TrainingArguments
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score, f1_score
        from datasets import Dataset
        import random

        pos = texts[:MAX_EXAMPLES]
        neg = (context or {}).get("all_benign_texts", [])[:MAX_EXAMPLES]

        if len(pos) < 5 or len(neg) < 5:
            return

        # 80/20 stratified split
        random.seed(42)
        random.shuffle(pos)
        random.shuffle(neg)

        val_pos_n = max(1, int(len(pos) * VAL_SPLIT))
        val_neg_n = max(1, int(len(neg) * VAL_SPLIT))

        train_texts = pos[val_pos_n:] + neg[val_neg_n:]
        train_labels = [1] * (len(pos) - val_pos_n) + [0] * (len(neg) - val_neg_n)

        val_texts = pos[:val_pos_n] + neg[:val_neg_n]
        val_labels = [1] * val_pos_n + [0] * val_neg_n

        train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

        st = SentenceTransformer(self.model_name)
        self._model = SetFitModel(model_body=st, model_head=LogisticRegression())

        trainer = Trainer(
            model=self._model,
            args=TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20),
            train_dataset=train_ds,
        )
        trainer.train()

        # Validation metrics on holdout
        val_preds = self._model.predict(val_texts).tolist()
        self.metrics = {
            "val_samples": len(val_labels),
            "precision": round(precision_score(val_labels, val_preds, zero_division=0), 4),
            "recall": round(recall_score(val_labels, val_preds, zero_division=0), 4),
            "f1": round(f1_score(val_labels, val_preds, zero_division=0), 4),
        }

        _shared_model = self._model
        _shared_metrics = self.metrics

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

        import tempfile, os, json
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

        # Store validation metrics
        if self.metrics:
            result["setfit_metrics"] = json.dumps(self.metrics)

        return result

    def load_weights(self, weights):
        if "setfit_manifest" not in weights:
            return

        _patch_compat()

        import json, tempfile, os
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

        if "setfit_metrics" in weights:
            self.metrics = json.loads(str(weights["setfit_metrics"]))
