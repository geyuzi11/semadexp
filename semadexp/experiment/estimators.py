"""Effect estimators: diff-in-means (cluster-robust), CUPED, randomization inference,
and exposure-mapping adjustment for interference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _robust_se(estimates: np.ndarray, clusters: np.ndarray, arm: np.ndarray, n: int) -> float:
    """Cluster-robust variance: aggregate residuals by cluster, CR-style sandwich."""
    m = int(clusters.max()) + 1
    residual = estimates - estimates.mean()
    s = np.zeros(m)
    for c in range(m):
        s[c] = residual[clusters == c].sum()
    v = (m / (m - 1)) * (s @ s) / n**2
    return float(np.sqrt(max(v, 1e-18)))


def diff_in_means(df: pd.DataFrame, clusters: np.ndarray) -> dict:
    y = df["y"].to_numpy(float)
    arm = df["arm"].to_numpy(int)
    t, c = y[arm == 1], y[arm == 0]
    est = t.mean() - c.mean()
    se = _robust_se(y - np.where(arm == 1, t.mean(), c.mean()), clusters, arm, len(y))
    return {
        "estimator": "diff_in_means",
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "p_value": 2 * (1 - stats.norm.cdf(abs(est) / (se + 1e-18))),
    }


def cuped(df: pd.DataFrame, clusters: np.ndarray, pre: np.ndarray | None = None) -> dict:
    y = df["y"].to_numpy(float)
    arm = df["arm"].to_numpy(int)
    if pre is None:
        out = diff_in_means(df, clusters)
        out["estimator"] = "cuped"
        return out
    x = _signed_log1p(pre.astype(float))
    control = arm == 0
    xc = x[control] - x[control].mean(axis=0)
    beta, *_ = np.linalg.lstsq(xc, y[control] - y[control].mean(), rcond=None)
    arm_mean = np.zeros_like(x)
    arm_mean[arm == 1] = x[arm == 1].mean(axis=0)
    arm_mean[arm == 0] = x[arm == 0].mean(axis=0)
    adj = y - (x - arm_mean) @ beta
    out = diff_in_means(pd.DataFrame({"y": adj, "arm": arm}), clusters)
    out["estimator"] = "cuped"
    return out


def randomization_inference(
    df: pd.DataFrame, clusters: np.ndarray, n_perm: int = 200, seed: int = 0
) -> dict:
    y = df["y"].to_numpy(float)
    arm = df["arm"].to_numpy(int)
    rng = np.random.default_rng(seed)
    m = int(clusters.max()) + 1
    units = np.arange(len(y))
    obs = y[arm == 1].mean() - y[arm == 0].mean()
    n_t = int(arm.sum())
    perms = np.empty(n_perm)
    for p in range(n_perm):
        perm = np.zeros(len(y), dtype=int)
        chosen = rng.choice(m, size=round(m * n_t / len(y)), replace=False)
        for c in chosen:
            perm[units[clusters == c]] = 1
        perms[p] = y[perm == 1].mean() - y[perm == 0].mean()
    p_value = (np.abs(perms) >= abs(obs)).mean()
    lo, hi = np.percentile(perms, [2.5, 97.5])
    return {
        "estimator": "randomization_inference",
        "estimate": obs,
        "se": float(perms.std()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
    }


def exposure_mapping(df: pd.DataFrame, exposure: np.ndarray | None = None) -> dict:
    """Linear-in-means exposure adjustment: y ~ treat + competitor_treat_share."""
    y = df["y"].to_numpy(float)
    arm = df["arm"].to_numpy(int)
    if exposure is None:
        exposure = np.full(len(y), 0.5)
    A = np.column_stack([np.ones(len(y)), arm, exposure])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    est = float(beta[1])
    return {
        "estimator": "exposure_mapping",
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "p_value": 2 * (1 - stats.norm.cdf(abs(est) / (se + 1e-18))),
    }


def estimate_all(
    df: pd.DataFrame,
    clusters: np.ndarray,
    pre: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rows = [
        diff_in_means(df, clusters),
        cuped(df, clusters, pre),
        randomization_inference(df, clusters, n_perm=n_perm, seed=seed),
        exposure_mapping(df, exposure),
    ]
    return pd.DataFrame(rows)
