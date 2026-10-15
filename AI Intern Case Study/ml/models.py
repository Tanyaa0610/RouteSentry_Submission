"""Thin, documented wrappers around scikit-learn's IsolationForest and
LocalOutlierFactor so the chronological evaluation harness (ml/chronological.py)
and the synthetic suite (ml/synthetic.py) share one scoring convention:

    HIGHER anomaly_score ALWAYS means MORE anomalous, for both models.

This is the opposite sign of scikit-learn's own `decision_function` /
`score_samples` (where higher = more normal / inlier), so every score
returned from this module is negated. Binary predictions are also
normalised to Python bool: True = anomaly (sklearn's internal -1), False =
normal (sklearn's internal +1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    LOF_CONTAMINATION,
    LOF_K_DIVISOR,
    LOF_K_MAX,
    LOF_K_MIN,
    RANDOM_SEED,
)


def choose_lof_k(train_size: int) -> int:
    """n_neighbors for LOF, scaled to the training fold's size. Documented
    heuristic (not tuned against any particular outcome): a fixed fraction
    of the training set, clamped to [LOF_K_MIN, LOF_K_MAX], and always
    strictly less than train_size (sklearn's hard requirement)."""
    k = max(LOF_K_MIN, min(LOF_K_MAX, train_size // LOF_K_DIVISOR))
    return max(1, min(k, train_size - 1))


@dataclass
class FittedScaler:
    scaler: StandardScaler

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "FittedScaler":
        scaler = StandardScaler().fit(X_train)
        return cls(scaler=scaler)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


def fit_isolation_forest(
    X_train_scaled: np.ndarray,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = RANDOM_SEED,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train_scaled)
    return model


def score_isolation_forest(model: IsolationForest, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = -model.decision_function(X_scaled)  # flip sign: higher = more anomalous
    predictions = model.predict(X_scaled) == -1  # True = anomaly
    return scores, predictions


def fit_lof(
    X_train_scaled: np.ndarray,
    contamination: float = LOF_CONTAMINATION,
) -> LocalOutlierFactor:
    k = choose_lof_k(len(X_train_scaled))
    model = LocalOutlierFactor(n_neighbors=k, contamination=contamination, novelty=True)
    model.fit(X_train_scaled)
    return model


def score_lof(model: LocalOutlierFactor, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = -model.score_samples(X_scaled)  # flip sign: higher = more anomalous
    predictions = model.predict(X_scaled) == -1  # True = anomaly
    return scores, predictions
