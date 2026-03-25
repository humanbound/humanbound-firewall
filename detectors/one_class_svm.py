"""One-class SVM detector for Tier 2 of hb-firewall.

Learns the boundary around positive examples using frozen sentence embeddings
and a one-class SVM. No negative class needed.

Usage:
    hb firewall train --model detectors/one_class_svm.py
"""

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


class AgentClassifier:
    def __init__(self, name, embed_model=DEFAULT_EMBED_MODEL, nu=0.05):
        self.name = name
        self.nu = nu
        self.embed_model_name = embed_model
        self._embed_model = None
        self._clf = None
        self._scaler = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    def train(self, texts, context=None):
        import numpy as np
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler

        if len(texts) < 10:
            return

        embeddings = self._get_embed_model().encode(texts, show_progress_bar=True)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(embeddings)

        self._clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale")
        self._clf.fit(X)

    def predict(self, text, context=""):
        if self._clf is None:
            return False, 0.0

        embedding = self._get_embed_model().encode([text])[0]
        X = self._scaler.transform(embedding.reshape(1, -1))
        score = float(self._clf.decision_function(X)[0])
        return score > 0, score

    def export_weights(self):
        import numpy as np
        import pickle
        if self._clf is None:
            return {}
        p = self.name
        return {
            f"{p}_clf": np.void(pickle.dumps(self._clf)),
            f"{p}_scaler": np.void(pickle.dumps(self._scaler)),
            f"{p}_embed_model": self.embed_model_name,
        }

    def load_weights(self, weights):
        import pickle

        p = self.name
        if f"{p}_clf" not in weights:
            return

        self._clf = pickle.loads(bytes(weights[f"{p}_clf"]))
        self._scaler = pickle.loads(bytes(weights[f"{p}_scaler"]))
        if f"{p}_embed_model" in weights:
            self.embed_model_name = str(weights[f"{p}_embed_model"])
