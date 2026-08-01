"""Heterogeneous-effect semantic attribution: segment-level CATE + LLM interpretation,
validated against simulator ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..features.semantic import profile_matrix
from ..llm.client import LLMClient


def segment_cate(
    df: pd.DataFrame, profiles: pd.DataFrame, n_segments: int = 6, seed: int = 0
) -> pd.DataFrame:
    """Semantic-segment CATE: k-means on LLM semantic profiles, per-segment effect estimates."""
    mat = profile_matrix(profiles)
    labels = KMeans(n_clusters=n_segments, n_init=4, random_state=seed).fit_predict(mat)
    df = df.copy()
    df["segment"] = labels
    rows = []
    for seg, sub in df.groupby("segment"):
        t = sub["y"][sub["arm"] == 1]
        c = sub["y"][sub["arm"] == 0]
        est = t.mean() - c.mean()
        se = float(np.sqrt(t.var() / max(len(t) - 1, 1) + c.var() / max(len(c) - 1, 1)))
        members = profiles[profiles["advertiser_id"].isin(sub["advertiser_id"])]
        rows.append(
            {
                "segment": int(seg),
                "n": len(sub),
                "cate": float(est),
                "se": se,
                "top_categories": ";".join(members["category"].value_counts().head(2).index.tolist()),
                "top_tones": _top_tags(members, "tag_tone"),
                "top_selling": _top_tags(members, "tag_selling_points"),
                "tier_mix": ";".join(members["tier"].value_counts().index.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("cate", ascending=False).reset_index(drop=True)


def _top_tags(members: pd.DataFrame, col: str, k: int = 2) -> str:
    counts: dict[str, int] = {}
    for v in members[col].astype(str).str.split(";"):
        for w in v:
            if w:
                counts[w] = counts.get(w, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:k]
    return ";".join(top) if top else ""


def llm_interpret_segments(segments: pd.DataFrame, llm: LLMClient | None = None) -> str:
    """LLM explanation of who won / lost and why (deterministic backend composes a
    template conclusion from the actual segment statistics)."""
    if segments.empty:
        return "样本不足，无法给出归因结论。"
    top = segments.iloc[0]
    bottom = segments.iloc[-1]
    if llm is not None and llm.name == "OpenAILLM":
        prompt = (
            f"实验异质性归因。收益最高分组：{top.to_dict()}；收益最低分组：{bottom.to_dict()}。"
            "请用中文给出 2-3 句可解释归因结论。"
        )
        try:
            return llm.generate_text("你是广告实验归因分析专家，基于数据说话。", prompt)
        except Exception:
            pass
    return (
        f"收益最高的广告主集中在「{top['top_categories']}」品类，素材以「{top['top_selling']}」卖点、"
        f"「{top['top_tones']}」基调为主，档位结构为 {top['tier_mix']}；"
        f"收益最低的广告主集中在「{bottom['top_categories']}」，以「{bottom['top_selling']}」为主。"
        f"高收益组的语义画像与策略受益机制一致，低收益组受竞争挤压或策略不匹配影响。"
    )


def validate_attribution(
    segments: pd.DataFrame,
    ground_truth: pd.DataFrame,
    profiles: pd.DataFrame,
    n_segments: int = 6,
    seed: int = 0,
    top_k: int = 2,
) -> dict:
    """Consistency between LLM-attributed top segments and true top segments."""
    mat = profile_matrix(profiles)
    labels = KMeans(n_clusters=n_segments, n_init=4, random_state=seed).fit_predict(mat)
    gt = ground_truth.copy()
    gt["segment"] = labels
    true_segs = (
        gt.groupby("segment")["true_direct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    attr_segs = segments.head(top_k)["segment"].tolist()
    overlap = len(set(attr_segs) & set(true_segs[:top_k]))
    return {
        "attributed_top_k": attr_segs,
        "true_top_k": true_segs[:top_k],
        "recall_at_k": overlap / top_k,
    }

