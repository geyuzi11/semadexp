"""Prediction baselines: GBDT (LightGBM when libomp is available, else sklearn
HistGradientBoosting) and logistic regression, with AUC / log-loss evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def fit_predictor(kind: str = "gbdt"):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000)
    try:
        import lightgbm

        return lightgbm.LGBMClassifier(
            n_estimators=120, learning_rate=0.08, num_leaves=15, verbosity=-1, random_state=0
        )
    except Exception:
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=15, random_state=0
        )


def run_prediction_ablation(X_trad: np.ndarray, X_fused: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_trad = cross_val_predict(fit_predictor(), X_trad, y, cv=cv, method="predict_proba")[:, 1]
    p_fused = cross_val_predict(fit_predictor(), X_fused, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "auc_trad": float(roc_auc_score(y, p_trad)),
        "auc_fused": float(roc_auc_score(y, p_fused)),
        "logloss_trad": float(log_loss(y, p_trad)),
        "logloss_fused": float(log_loss(y, p_fused)),
        "entropy_gain": float(log_loss(y, p_trad) - log_loss(y, p_fused)),
        "auc_gain": float(roc_auc_score(y, p_fused) - roc_auc_score(y, p_trad)),
    }

