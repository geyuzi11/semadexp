"""Historical experiment knowledge base: simulator experiments with known true
effects + generated review / ticket texts, used as the quantification ground truth
for the hypothesis-generation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import PolicyChange, PolicyKind, SimulationConfig
from ..data.corpora import generate_advertisers
from ..features.semantic import build_profiles
from ..llm.client import LLMClient
from ..simulator.market import run_ground_truth


CANDIDATE_POLICIES: list[PolicyChange] = [
    PolicyChange(kind=PolicyKind.LOWER_BID_FLOOR, param=0.6, name="降低竞价门槛"),
    PolicyChange(kind=PolicyKind.RAISE_BID_FLOOR, param=1.6, name="提高竞价门槛"),
    PolicyChange(kind=PolicyKind.AD_LOAD_CAP, param=3, name="限制广告条数"),
    PolicyChange(kind=PolicyKind.AUCTION_RULE, param=1, name="切换二价机制"),
    PolicyChange(kind=PolicyKind.BUDGET_CAP_MULTIPLIER, param=1.2, name="放宽预算上限"),
]


@dataclass
class ExperimentRecord:
    id: int
    policy: PolicyChange
    metric: str
    true_effect: float
    true_direct: float
    true_indirect: float
    per_type_effect: dict[str, float]
    review_text: str
    tickets: list[str]


def build_knowledge_base(
    n_experiments: int = 60,
    seed: int = 17,
    n_advertisers: int = 90,
    days: int = 6,
    sessions_per_day: int = 400,
    llm: LLMClient | None = None,
) -> list[ExperimentRecord]:
    rng = np.random.default_rng(seed)
    ads = generate_advertisers(n_advertisers, seed)
    records: list[ExperimentRecord] = []
    for i in range(n_experiments):
        policy = CANDIDATE_POLICIES[i % len(CANDIDATE_POLICIES)]
        cfg = SimulationConfig(
            market={
                "n_advertisers": n_advertisers,
                "days": days,
                "sessions_per_day": sessions_per_day,
                "seed": seed + i,
                "interference_strength": float(rng.choice([0.6, 1.0, 1.5])),
            },
            policy=policy,
        )
        gt = run_ground_truth(cfg, ads)
        g = gt["ground_truth"]
        true_direct = float(g["true_direct"].mean())
        true_indirect = float(g["true_indirect"].mean())
        total = true_direct + true_indirect
        per_type = {}
        for tname, mask in [("S", g["tier"] == 0), ("M", g["tier"] == 1), ("L", g["tier"] == 2)]:
            per_type[tname] = float(g.loc[mask, "true_direct"].mean())
        review = _review_text(i, policy, total, true_direct, true_indirect, per_type)
        tickets = [
            f"复盘 {i}: {policy.label()} 实验真实总效应 {total:+.3f}，直接效应 {true_direct:+.3f}，"
            f"广告主反馈间接效应 {true_indirect:+.3f}；中小广告主受益 {per_type['S']:+.3f}。"
        ]
        records.append(
            ExperimentRecord(
                id=i,
                policy=policy,
                metric="conversions",
                true_effect=total,
                true_direct=true_direct,
                true_indirect=true_indirect,
                per_type_effect=per_type,
                review_text=review,
                tickets=tickets,
            )
        )
    return records


def _review_text(
    i: int,
    policy: PolicyChange,
    total: float,
    direct: float,
    indirect: float,
    per_type: dict[str, float],
) -> str:
    direction = "正向" if total > 0 else "负向"
    return (
        f"实验#{i}「{policy.label()}」复盘：整体效果{direction}（{total:+.3f}），"
        f"其中策略直接效应 {direct:+.3f}、广告主反馈效应 {indirect:+.3f}；"
        f"中小/中型/大型广告主直接效应分别为 {per_type['S']:+.3f} / {per_type['M']:+.3f} / {per_type['L']:+.3f}。"
    )

