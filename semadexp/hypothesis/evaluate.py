"""Quantified evaluation of the hypothesis layer: recall, ranking correlation,
top-K value capture and experiments-to-target vs random / manual baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .generate import Hypothesis


@dataclass
class HypothesisEvalReport:
    recall_at_top_x: float
    spearman_priority_vs_true: float
    spearman_p_value: float
    top_k_capture: dict[int, float]
    random_baseline_capture: dict[int, float]
    manual_baseline_capture: dict[int, float]
    experiments_to_50pct: int
    random_experiments_to_50pct: int
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_generation(
    ranked: list[Hypothesis],
    knowledge_base: list,
    top_x: int = 8,
    capture_k: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20),
    seed: int = 0,
) -> HypothesisEvalReport:
    """Opportunities = top-X historical experiments by absolute true effect.
    A hypothesis 'covers' an opportunity when its policy_kind matches."""
    rng = np.random.default_rng(seed)
    kb = pd.DataFrame(
        [
            {
                "id": r.id,
                "policy_kind": r.policy.kind.value,
                "true_effect": r.true_effect,
                "value": abs(r.true_effect),
            }
            for r in knowledge_base
        ]
    )
    opp = kb.sort_values("value", ascending=False).head(top_x).reset_index(drop=True)
    covered = {kind: False for kind in opp["policy_kind"].unique()}
    for h in ranked:
        if h.policy_kind in covered:
            covered[h.policy_kind] = True
    recall = sum(covered.values()) / max(len(covered), 1)

    matched = []
    for h in ranked:
        row = kb[kb["id"] == h.matched_experiment_id] if h.matched_experiment_id is not None else None
        if row is not None and len(row):
            matched.append({"priority": h.priority, "true_effect": h.true_effect or 0.0})
    m = pd.DataFrame(matched)
    rho, pval = (0.0, 1.0)
    if len(m) > 3 and m["priority"].nunique() > 1:
        rho, pval = spearmanr(m["priority"], m["true_effect"])

    total_value = opp["value"].sum()
    capture: dict[int, float] = {}
    seen: set[str] = set()
    cum = 0.0
    for k in capture_k:
        for h in ranked[: max(k, len(ranked))]:
            if h.policy_kind in opp["policy_kind"].tolist() and h.policy_kind not in seen:
                v = opp.loc[opp["policy_kind"] == h.policy_kind, "value"].max()
                cum += v
                seen.add(h.policy_kind)
        capture[k] = cum / max(total_value, 1e-9)

    rand_cap: dict[int, float] = {}
    for k in capture_k:
        vals = []
        for _ in range(40):
            sample = kb.sample(min(k, len(kb)), random_state=int(rng.integers(1e6))).drop_duplicates("policy_kind")
            vals.append(sample["policy_kind"].isin(opp["policy_kind"]).sum() / max(len(opp), 1))
        rand_cap[k] = float(np.mean(vals))

    manual_cap = {k: min(k, len(opp)) / max(len(opp), 1) for k in capture_k}

    def experiments_to_target(capture_map: dict[int, float], target: float = 0.5) -> int:
        for k in capture_k:
            if capture_map.get(k, 0.0) >= target:
                return k
        return int(capture_k[-1])

    return HypothesisEvalReport(
        recall_at_top_x=float(recall),
        spearman_priority_vs_true=float(rho),
        spearman_p_value=float(pval),
        top_k_capture=capture,
        random_baseline_capture=rand_cap,
        manual_baseline_capture=manual_cap,
        experiments_to_50pct=experiments_to_target(capture),
        random_experiments_to_50pct=experiments_to_target(rand_cap),
        detail=m,
    )

