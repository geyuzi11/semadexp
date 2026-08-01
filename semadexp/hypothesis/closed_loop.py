"""Closed-loop demo: hypothesis -> equilibrium simulation -> graph bucketing ->
experiment -> attribution -> knowledge base feedback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    BehaviorKind,
    EquilibriumScenario,
    ExperimentConfig,
    PolicyChange,
    PolicyKind,
    SimulationConfig,
)
from ..data.corpora import generate_advertisers
from ..experiment.attribution import llm_interpret_segments, segment_cate
from ..experiment.bucketing import assign_buckets, assignment_metrics
from ..experiment.decomposition import decompose_total, fit_response_model, parse_operation_events
from ..experiment.estimators import estimate_all
from ..features.competition_graph import build_competition_graph
from ..features.semantic import build_profiles
from ..simulator.equilibrium import run_equilibrium
from ..simulator.market import run_experiment, run_ground_truth
from .generate import Hypothesis


def run_closed_loop(
    hypothesis: Hypothesis,
    profiles: pd.DataFrame,
    graph,
    ground_truth: dict,
    advertisers,
    sim_cfg: SimulationConfig | None = None,
    seed: int = 0,
) -> dict:
    """One full iteration: simulate equilibrium, design the experiment, run it,
    estimate/decompose/attribute, and report what was learned."""
    policy = PolicyChange(kind=PolicyKind(hypothesis.policy_kind), param=_default_param(hypothesis.policy_kind))
    if sim_cfg is None:
        sim_cfg = SimulationConfig(
            market={
                "n_advertisers": len(advertisers),
                "days": 6,
                "sessions_per_day": 400,
                "seed": seed,
            },
            policy=policy,
        )
    else:
        sim_cfg = sim_cfg.model_copy(deep=True)
        sim_cfg.policy = policy
    eq = run_equilibrium(
        EquilibriumScenario(policy=policy, seed=seed), n_advertisers=len(advertisers),
        days_per_iter=4, sessions_per_day=300, seed=seed,
    )
    assignment = assign_buckets(
        profiles, graph, ExperimentConfig().bucketing, seed=seed
    )
    outcome = run_experiment(sim_cfg, assignment, advertisers, seed=seed + 1)
    df = outcome.advertiser_outcomes.rename(columns={"conversions": "y"})[["advertiser_id", "arm", "y"]]
    budgets = np.array([a.daily_budget for a in advertisers])
    cats = np.array([a.category_idx for a in advertisers])
    exposure = np.zeros(len(advertisers))
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        total = budgets[idx].sum()
        exposure[idx] = budgets[idx] * assignment[idx] @ budgets[idx] / max(total, 1e-9)
    gt = ground_truth["ground_truth"]
    true_mean = float(gt["true_direct"].mean())
    cluster_labels = np.zeros(len(df), dtype=int)
    comm_map = {node: cid for cid, nodes in graph.communities.items() for node in nodes}
    for i, aid in enumerate(df["advertiser_id"]):
        cluster_labels[i] = comm_map.get(int(aid), 0)
    est = estimate_all(df, cluster_labels, exposure=exposure, n_perm=100, seed=seed)
    control_mean = float(df.loc[df["arm"] == 0, "y"].mean())
    df_eff = df[df["arm"] == 1].copy()
    df_eff["y"] = df_eff["y"] - control_mean
    treated_ids = set(df_eff["advertiser_id"])
    gt_eff = gt[gt["advertiser_id"].isin(treated_ids)].copy()
    logs = []
    text_map = {
        BehaviorKind.RAISE_BUDGET: ("提升日预算", "这周预算加了，多投一些"),
        BehaviorKind.RAISE_BID: ("调高出价", "竞争太激烈，把出价往上调了"),
        BehaviorKind.CHANGE_CREATIVE: ("更换素材", "旧素材点击率太低，换了新文案"),
        BehaviorKind.EXPAND_TARGETING: ("扩定向", "人群太窄，扩展了定向人群"),
        BehaviorKind.EXIT: ("暂停投放", "效果不好，先暂停投放"),
    }
    rng = np.random.default_rng(seed)
    for ad in advertisers:
        if ad.id not in treated_ids:
            continue
        for resp in sim_cfg.responses:
            if not _type_matches(ad, resp.advertiser_type):
                continue
            if rng.random() > resp.probability:
                continue
            head, note = text_map.get(resp.behavior, (resp.behavior.value, ""))
            logs.append(
                {
                    "advertiser_id": ad.id,
                    "event_type": resp.behavior.value,
                    "magnitude": resp.magnitude,
                    "day": 2,
                    "text": f"[{head}] {note}，幅度约{resp.magnitude:.0%}",
                }
            )
    ev = parse_operation_events(logs, None)
    model = fit_response_model(gt_eff, ev)
    decomp = decompose_total(df_eff, ev, model, gt_eff)
    segs = segment_cate(df, profiles, n_segments=5, seed=seed)
    narrative = llm_interpret_segments(segs, None)
    return {
        "hypothesis": hypothesis.statement,
        "policy": hypothesis.policy_kind,
        "equilibrium": eq,
        "design": assignment_metrics(profiles, graph, assignment),
        "estimates": est,
        "true_effect": true_mean,
        "decomposition": decomp,
        "segments": segs,
        "attribution": narrative,
    }


def _type_matches(ad, advertiser_type: str) -> bool:
    if advertiser_type in ("*", "all"):
        return True
    if advertiser_type in ("S", "M", "L"):
        return ad.tier == advertiser_type
    return ad.category == advertiser_type


def _default_param(kind: str) -> float:
    return {
        "lower_bid_floor": 0.6,
        "raise_bid_floor": 1.6,
        "ad_load_cap": 3,
        "auction_rule": 1.0,
        "budget_cap_multiplier": 1.2,
    }.get(kind, 1.0)
