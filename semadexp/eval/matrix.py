"""Scenario matrix: interference x heterogeneity x bucketing policy x estimator,
producing bias / variance / RMSE / coverage / power evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BucketingPolicy, MarketConfig, PolicyChange, PolicyKind, SimulationConfig
from ..data.corpora import generate_advertisers
from ..experiment.bucketing import assign_buckets, assignment_metrics
from ..experiment.estimators import estimate_all
from ..features.competition_graph import build_competition_graph
from ..features.semantic import build_profiles
from ..simulator.market import run_experiment, run_ground_truth


def _cluster_labels(n: int, graph) -> np.ndarray:
    comm_map = {node: cid for cid, nodes in graph.communities.items() for node in nodes}
    return np.array([comm_map.get(i, 0) for i in range(n)], dtype=int)


def _exposure(advertisers, assignment: np.ndarray) -> np.ndarray:
    budgets = np.array([a.daily_budget for a in advertisers])
    cats = np.array([a.category_idx for a in advertisers])
    out = np.zeros(len(advertisers))
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        total = budgets[idx].sum()
        out[idx] = budgets[idx] * assignment[idx] @ budgets[idx] / max(total, 1e-9)
    return out


def run_scenario_matrix(
    scenarios: list[dict],
    n_advertisers: int = 110,
    days: int = 8,
    sessions_per_day: int = 500,
    n_reps: int = 10,
    seed: int = 7,
    policies: tuple[BucketingPolicy, ...] = (
        BucketingPolicy.STRUCTURED_STRATIFIED,
        BucketingPolicy.SEMANTIC_STRATIFIED,
        BucketingPolicy.GRAPH_CLUSTER,
    ),
) -> dict:
    rows: list[dict] = []
    design_rows: list[dict] = []
    for sc in scenarios:
        mcfg = MarketConfig(
            n_advertisers=n_advertisers,
            days=days,
            sessions_per_day=sessions_per_day,
            interference_strength=sc["interference"],
            heterogeneity=sc["heterogeneity"],
            seed=seed,
        )
        policy = PolicyChange(kind=PolicyKind.AD_LOAD_CAP, param=3, name="限制广告条数")
        cfg = SimulationConfig(market=mcfg, policy=policy)
        advertisers = generate_advertisers(n_advertisers, seed)
        profiles = build_profiles(advertisers, _noop_llm())
        graph = build_competition_graph(profiles)
        gt = run_ground_truth(cfg, advertisers)
        # burn-in covariates: run the market without policy to obtain predictive pre-experiment features
        burn = run_experiment(
            SimulationConfig(market=mcfg, policy=None),
            np.zeros(n_advertisers, dtype=int),
            advertisers,
            seed=seed + 500,
        )
        b = burn.advertiser_outcomes
        pre = np.column_stack([b["spend"], b["conversions"], b["impressions"], b["profit"]]).astype(float)
        true = gt["ground_truth"]["true_direct"].to_numpy()
        clusters = _cluster_labels(n_advertisers, graph)
        for policy_enum in policies:
            est_holder: dict[str, list] = {
                "diff_in_means": [],
                "cuped": [],
                "randomization_inference": [],
                "exposure_mapping": [],
            }
            design_acc = {}
            for rep in range(n_reps):
                assignment = assign_buckets(profiles, graph, policy_enum, seed=seed + rep)
                outcome = run_experiment(cfg, assignment, advertisers, seed=seed + rep * 7 + 1)
                df = outcome.advertiser_outcomes.rename(columns={"conversions": "y"})[["advertiser_id", "arm", "y"]]
                expo = _exposure(advertisers, assignment)
                ests = estimate_all(df, clusters, pre=pre, exposure=expo, n_perm=80, seed=rep)
                for _, row in ests.iterrows():
                    est_holder[row["estimator"]].append(
                        {
                            "estimate": row["estimate"],
                            "se": row["se"],
                            "ci_low": row["ci_low"],
                            "ci_high": row["ci_high"],
                            "p": row["p_value"],
                        }
                    )
                m = assignment_metrics(profiles, graph, assignment)
                for k, v in m.items():
                    design_acc.setdefault(k, []).append(v)
            true_mean = float(true.mean())
            for est_name, vals in est_holder.items():
                ests = np.array([v["estimate"] for v in vals])
                ses = np.array([v["se"] for v in vals])
                bias = ests.mean() - true_mean
                coverage = np.mean(
                    [(v["ci_low"] <= true_mean) & (v["ci_high"] >= true_mean) for v in vals]
                )
                power = np.mean([abs(v["estimate"]) / max(v["se"], 1e-12) > 1.96 for v in vals])
                rows.append(
                    {
                        "scenario": sc["name"],
                        "interference": sc["interference"],
                        "heterogeneity": sc["heterogeneity"],
                        "bucketing": policy_enum.value,
                        "estimator": est_name,
                        "bias": bias,
                        "variance": float(ests.var()),
                        "rmse": float(np.sqrt(np.mean((ests - true_mean) ** 2))),
                        "coverage": float(coverage),
                        "power": float(power),
                        "true_effect": true_mean,
                    }
                )
            design_rows.append(
                {
                    "scenario": sc["name"],
                    "bucketing": policy_enum.value,
                    "max_smd": float(np.mean(design_acc["max_smd"])),
                    "cross_arm_edge_ratio": float(np.mean(design_acc["cross_arm_edge_ratio"])),
                    "internal_density": float(np.mean(design_acc["internal_density"])),
                }
            )
    return {
        "results": pd.DataFrame(rows),
        "design": pd.DataFrame(design_rows),
        "ground_truth_aux": {"true_effect": float(true.mean())},
    }


def _noop_llm():
    from ..llm.client import DeterministicLLM

    return DeterministicLLM()
