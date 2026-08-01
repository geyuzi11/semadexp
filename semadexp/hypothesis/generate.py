"""Hypothesis generation pipeline: LLM extraction -> semantic dedupe -> feasibility
filter -> priority scoring against the historical knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..llm.client import LLMClient


METRIC_REGISTRY = {
    "total_spend": "总消耗",
    "total_conversions": "总转化",
    "negative_feedback_rate": "用户负反馈率",
    "advertiser_survival": "广告主留存",
    "small_advertiser_survival": "中小广告主存活率",
}

HYPOTHESIS_TEMPLATES: dict[str, dict] = {
    "lower_bid_floor": {
        "statement": "降低竞价门槛可提升中小广告主活跃度与总消耗",
        "metric": "small_advertiser_survival",
        "direction": "up",
    },
    "raise_bid_floor": {
        "statement": "提高竞价门槛可降低低质广告数量与用户负反馈率",
        "metric": "negative_feedback_rate",
        "direction": "down",
    },
    "ad_load_cap": {
        "statement": "限制单条信息流广告条数可降低用户负反馈率",
        "metric": "negative_feedback_rate",
        "direction": "down",
    },
    "auction_rule": {
        "statement": "切换二价机制可降低恶意抬价并提升广告主留存",
        "metric": "advertiser_survival",
        "direction": "up",
    },
    "budget_cap_multiplier": {
        "statement": "放宽预算上限可提升大客户消耗与总转化",
        "metric": "total_spend",
        "direction": "up",
    },
}


@dataclass
class Hypothesis:
    statement: str
    metric: str
    direction: str
    confidence: float
    source: str
    policy_kind: str
    expected_lift: float = 0.0
    risk: float = 0.5
    priority: float = 0.0
    matched_experiment_id: int | None = None
    true_effect: float | None = None


def extract_candidate_hypotheses(
    corpus: pd.DataFrame, llm: LLMClient | None = None
) -> list[Hypothesis]:
    """Extract candidate hypotheses from multi-source unstructured texts."""
    hyps: list[Hypothesis] = []
    if llm is not None and llm.name == "OpenAILLM":
        system = (
            "从反馈语料中提炼可验证实验假设，输出 JSON: {hypotheses: ["
            "{statement, metric, direction, confidence}]}，metric 从 "
            f"{list(METRIC_REGISTRY)} 中选择。"
        )
        try:
            batch = "\n".join(corpus["text"].tolist()[:20])
            raw = llm.extract_structured(system, batch)
            for h in raw.get("hypotheses", []):
                hyps.append(
                    Hypothesis(
                        statement=str(h.get("statement", "")),
                        metric=str(h.get("metric", "")),
                        direction=str(h.get("direction", "up")),
                        confidence=float(h.get("confidence", 0.5)),
                        source="llm",
                        policy_kind=str(h.get("policy_kind", "")),
                    )
                )
            return hyps
        except Exception:
            pass
    for _, row in corpus.iterrows():
        opp = row["opportunity"]
        tpl = HYPOTHESIS_TEMPLATES.get(opp)
        if tpl is None:
            continue
        hyps.append(
            Hypothesis(
                statement=tpl["statement"],
                metric=tpl["metric"],
                direction=tpl["direction"],
                confidence=0.55 + 0.08 * (int(row["id"].split("_")[-1]) % 4) / 3.0,
                source="corpus",
                policy_kind=opp,
            )
        )
    return _dedupe(hyps)


def _dedupe(hyps: list[Hypothesis]) -> list[Hypothesis]:
    if len(hyps) < 2:
        return hyps
    vec = TfidfVectorizer().fit_transform([h.statement for h in hyps]).toarray()
    norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    keep: list[Hypothesis] = []
    for i, h in enumerate(hyps):
        if any(sim[i, j] > 0.9 for j, _ in enumerate(hyps) if j < i and sim[i, j] > 0):
            continue
        dup = [j for j in range(len(hyps)) if j != i and sim[i, j] > 0.9]
        h2 = h
        for j in dup:
            if hyps[j].confidence > h2.confidence:
                h2 = hyps[j]
        keep.append(h2)
    return keep


def score_priorities(
    hypotheses: list[Hypothesis],
    knowledge_base: list,
    cost_map: dict[str, float] | None = None,
) -> list[Hypothesis]:
    """Priority = expected lift (from similar historical experiments) x confidence / risk."""
    costs = cost_map or {
        "auction_rule": 0.9,
        "raise_bid_floor": 0.7,
        "lower_bid_floor": 0.6,
        "budget_cap_multiplier": 0.5,
        "ad_load_cap": 0.4,
    }
    kb = pd.DataFrame(
        [
            {
                "id": r.id,
                "policy_kind": r.policy.kind.value,
                "true_effect": r.true_effect,
            }
            for r in knowledge_base
        ]
    )
    for h in hypotheses:
        similar = kb[kb["policy_kind"] == h.policy_kind]
        if len(similar) == 0:
            h.expected_lift = 0.0
            h.confidence = min(h.confidence, 0.5)
            h.risk = costs.get(h.policy_kind, 0.5)
            h.priority = 0.0
            continue
        h.expected_lift = float(np.mean(similar["true_effect"]))
        same_sign = float(((similar["true_effect"] > 0) == (h.direction == "up")).mean())
        h.confidence = float(np.clip(0.5 * h.confidence + 0.5 * same_sign, 0.1, 0.95))
        h.risk = costs.get(h.policy_kind, 0.5)
        h.priority = h.expected_lift * h.confidence / h.risk
        # representative experiment: closest to the policy's median effect (stable ordering signal)
        med = float(similar["true_effect"].median())
        rep = similar.loc[(similar["true_effect"] - med).abs().idxmin()]
        h.matched_experiment_id = int(rep["id"])
        h.true_effect = float(rep["true_effect"])
    return sorted(hypotheses, key=lambda h: h.priority, reverse=True)
