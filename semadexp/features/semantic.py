"""LLM semantic feature layer: creative tags + embeddings -> advertiser profiles,
plus incremental-information ablation against traditional behavior features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..data.corpora import AdvertiserSeed
from ..llm.client import LLMClient


TAG_KEYS = ["selling_points", "target_audiences", "tone", "objective"]


@dataclass
class AdvertiserProfile:
    advertiser_id: int
    category: str
    tier: str
    audience_tags: list[str]
    tag_counts: dict[str, list[str]]
    embedding: np.ndarray


def extract_creative_semantics(
    advertisers: list[AdvertiserSeed], llm: LLMClient
) -> dict[int, dict]:
    """Structured tags + embedding for each advertiser's creative corpus."""
    system = (
        "你是广告素材语义分析器。从广告文案中抽取：selling_points(卖点)、target_audiences(目标人群)、"
        "tone(情绪基调)、objective(投放目标)、categories(品类)、price_tier(low/mid/high)，"
        "并给出 summary。所有字段为字符串数组。"
    )
    out: dict[int, dict] = {}
    for ad in advertisers:
        text = " ".join(ad.creatives)
        tags = llm.extract_structured(system, text)
        emb = llm.embed([text])[0]
        out[ad.id] = {"tags": tags, "embedding": np.asarray(emb, dtype=np.float32)}
    return out


def build_profiles(
    advertisers: list[AdvertiserSeed],
    llm: LLMClient,
    dim: int = 24,
) -> pd.DataFrame:
    """Advertiser-level profile matrix: semantic embedding (PCA-reduced) + tag/category/tier features."""
    semantics = extract_creative_semantics(advertisers, llm)
    rows: list[dict] = []
    embs: list[np.ndarray] = []
    for ad in advertisers:
        s = semantics[ad.id]
        tags = s["tags"]
        emb = s["embedding"]
        embs.append(emb)
        row = {
            "advertiser_id": ad.id,
            "category": ad.category,
            "category_idx": ad.category_idx,
            "tier": ad.tier,
            "audience_tags": ad.audience_tags,
            "objective": ad.objective,
            **{f"tag_{k}": ";".join(v) for k, v in tags.items() if k in TAG_KEYS},
            "price_tier": tags.get("price_tier", "mid"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    mat = np.vstack(embs)
    if mat.shape[1] > dim:
        pca = PCA(n_components=min(dim, mat.shape[0] - 1))
        mat = pca.fit_transform(mat)
    elif mat.shape[1] < dim:
        mat = np.pad(mat, ((0, 0), (0, dim - mat.shape[1])))
    df["embedding"] = list(mat.astype(np.float32))
    return df


def profile_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix for bucketing: embedding + one-hot category/tier/price + tag presence."""
    from sklearn.preprocessing import OneHotEncoder

    cols = []
    cols.append(np.vstack(df["embedding"].values))
    cat = OneHotEncoder(sparse_output=False).fit_transform(df[["category"]])
    tier = OneHotEncoder(sparse_output=False).fit_transform(df[["tier"]])
    price = OneHotEncoder(sparse_output=False).fit_transform(df[["price_tier"]])
    cols.extend([cat, tier, price])
    for k in TAG_KEYS:
        vocab = sorted({w for v in df[f"tag_{k}"].str.split(";") for w in v if w})
        m = np.zeros((len(df), len(vocab)))
        for i, v in enumerate(df[f"tag_{k}"].values):
            for w in str(v).split(";"):
                if w in vocab:
                    m[i, vocab.index(w)] = 1.0
        cols.append(m)
    return np.hstack(cols).astype(np.float32)


def incremental_information_eval(
    behavior_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Ablation: traditional behavior features vs behavior + LLM semantic features.
    Reports AUC / log-loss / entropy gain (log-loss reduction) with 5-fold CV."""
    from ..models.predictors import fit_predictor

    trad_cols = [
        c
        for c in [
            "pre_spend",
            "pre_conversions",
            "pre_ctr",
            "budget",
            "quality",
            "tier_S",
            "tier_L",
        ]
        if c in behavior_df.columns
    ]
    X_trad = np.log1p(np.clip(behavior_df[trad_cols].astype(float).values, 0, None))
    y = behavior_df["y_click_next"].values
    emb = np.vstack(profiles_df["embedding"].values)[:, :8]
    cat = pd.get_dummies(profiles_df["category"], prefix="cat")
    tier = pd.get_dummies(profiles_df["tier"], prefix="tier")
    price = pd.get_dummies(profiles_df["price_tier"], prefix="price")
    sem = np.hstack([emb, cat.astype(float).values, tier.astype(float).values, price.astype(float).values])
    X_fused = np.hstack([X_trad, sem])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_trad = _cv_logistic(X_trad, y, cv)
    p_fused = _cv_logistic(X_fused, y, cv)
    return {
        "auc_trad": roc_auc_score(y, p_trad),
        "auc_fused": roc_auc_score(y, p_fused),
        "logloss_trad": log_loss(y, p_trad),
        "logloss_fused": log_loss(y, p_fused),
        "entropy_gain": log_loss(y, p_trad) - log_loss(y, p_fused),
        "auc_gain": roc_auc_score(y, p_fused) - roc_auc_score(y, p_trad),
        "n_trad_features": X_trad.shape[1],
        "n_semantic_features": sem.shape[1],
    }


def _cv_logistic(X: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    """Standardized, strongly regularized logistic regression with out-of-fold predictions."""
    preds = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.05, max_iter=3000, random_state=0).fit(sc.transform(X[tr]), y[tr])
        preds[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return preds
