"""Counterfactual equilibrium simulation: policy change + LLM advertiser-response hypotheses,
iterated until ecosystem metrics converge, with sensitivity bands and a static contrast."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import BehaviorKind, EquilibriumScenario, PolicyChange, ResponseHypothesis, SimulationConfig
from ..data.corpora import generate_advertisers
from ..llm.client import LLMClient
from .market import Market


@dataclass
class EquilibriumResult:
    scenario: EquilibriumScenario
    iterations: pd.DataFrame
    converged: bool
    final_metrics: dict[str, float]
    static_metrics: dict[str, float]
    sensitivity: pd.DataFrame
    response_hypotheses: list[ResponseHypothesis]


DEFAULT_RESPONSES: list[ResponseHypothesis] = [
    ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.18, probability=0.8),
    ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BID, magnitude=0.10, probability=0.6),
    ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.08, probability=0.6),
    ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.CHANGE_CREATIVE, magnitude=0.15, probability=0.5),
    ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.EXPAND_TARGETING, magnitude=0.10, probability=0.7),
]

POLICY_AWARE_RESPONSES: dict[str, list[ResponseHypothesis]] = {
    "lower_bid_floor": [
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.20, probability=0.8),
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.EXPAND_TARGETING, magnitude=0.10, probability=0.5),
        ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.CHANGE_CREATIVE, magnitude=0.15, probability=0.4),
        ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.05, probability=0.5),
    ],
    "raise_bid_floor": [
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BID, magnitude=0.15, probability=0.5),
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.EXIT, magnitude=0.3, probability=0.6),
        ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.10, probability=0.6),
        ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.EXPAND_TARGETING, magnitude=0.10, probability=0.5),
    ],
    "ad_load_cap": [
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.10, probability=0.6),
        ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.CHANGE_CREATIVE, magnitude=0.15, probability=0.5),
        ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.RAISE_BID, magnitude=0.08, probability=0.5),
    ],
    "auction_rule": [
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.15, probability=0.7),
        ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.RAISE_BID, magnitude=0.08, probability=0.5),
        ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.EXPAND_TARGETING, magnitude=0.10, probability=0.6),
    ],
    "budget_cap_multiplier": [
        ResponseHypothesis(advertiser_type="L", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.20, probability=0.8),
        ResponseHypothesis(advertiser_type="M", behavior=BehaviorKind.RAISE_BUDGET, magnitude=0.15, probability=0.6),
        ResponseHypothesis(advertiser_type="S", behavior=BehaviorKind.CHANGE_CREATIVE, magnitude=0.15, probability=0.5),
    ],
}


def llm_response_hypotheses(scenario: EquilibriumScenario, llm: LLMClient | None = None) -> list[ResponseHypothesis]:
    """Ask the LLM for advertiser-type response hypotheses; deterministic backend returns templates."""
    if scenario.response_hypotheses:
        return scenario.response_hypotheses
    if llm is None:
        return POLICY_AWARE_RESPONSES.get(scenario.policy.kind.value, DEFAULT_RESPONSES)
    kinds = "、".join([k.value for k in BehaviorKind])
    prompt = (
        f"策略变更为 {scenario.policy.label()}。请输出中小/中/大型广告主可能的主动策略调整，"
        f"每类给出行为({kinds})、幅度(0-0.5)、概率(0-1)，用 JSON。"
    )
    try:
        raw = llm.extract_structured("你是广告平台策略仿真专家。", prompt)
        hyps: list[ResponseHypothesis] = []
        for item in raw.get("responses", []):
            hyps.append(
                ResponseHypothesis(
                    advertiser_type=str(item.get("type", "*")),
                    behavior=BehaviorKind(str(item.get("behavior", "raise_budget"))),
                    magnitude=float(item.get("magnitude", 0.1)),
                    probability=float(item.get("probability", 0.5)),
                    source="llm",
                )
            )
        return hyps or POLICY_AWARE_RESPONSES.get(scenario.policy.kind.value, DEFAULT_RESPONSES)
    except Exception:
        return POLICY_AWARE_RESPONSES.get(scenario.policy.kind.value, DEFAULT_RESPONSES)


def _ecosystem_metrics(outcome) -> dict[str, float]:
    df = outcome.advertiser_outcomes
    small = df[df["tier"] == 0]
    return {
        "total_spend": float(df["spend"].sum()),
        "total_conversions": float(df["conversions"].sum()),
        "negative_feedback_rate": float(outcome.market_metrics["negative_feedback_rate"]),
        "advertiser_survival": float(outcome.market_metrics["advertiser_survival"]),
        "small_advertiser_survival": float((small["active_days"] > 0.5).mean()) if len(small) else 1.0,
    }


def _metric_delta(a: dict[str, float], b: dict[str, float], scale: dict[str, float]) -> float:
    return max(abs(a[k] - b[k]) / (abs(scale[k]) + 1e-9) for k in a)


def run_equilibrium(
    scenario: EquilibriumScenario,
    llm: LLMClient | None = None,
    n_advertisers: int = 120,
    days_per_iter: int = 6,
    sessions_per_day: int = 600,
    seed: int | None = None,
) -> EquilibriumResult:
    """Iterate policy + advertiser responses until ecosystem metrics converge."""
    rng = np.random.default_rng(scenario.seed)
    hypotheses = llm_response_hypotheses(scenario, llm)
    ads = generate_advertisers(n_advertisers, scenario.seed)
    cfg = SimulationConfig(
        market={
            "n_advertisers": n_advertisers,
            "days": days_per_iter,
            "sessions_per_day": sessions_per_day,
            "seed": scenario.seed,
        },
        policy=scenario.policy,
        responses=hypotheses,
    )
    mkt = Market(ads, cfg.market)
    rows: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    exit_mask = np.zeros(len(ads), dtype=bool)
    metrics: dict[str, float] = {}
    converged = False
    scale_metrics: dict[str, float] | None = None
    for it in range(scenario.max_iters):
        outcome = mkt.run(
            cfg.policy,
            responses=hypotheses,
            seed=scenario.seed + it,
            initial_exit=exit_mask,
        )
        metrics = _ecosystem_metrics(outcome)
        if scale_metrics is None:
            scale_metrics = metrics
        rows.append({"iter": it, **metrics})
        # advertiser churn: advertisers below the ROI threshold exit, with a
        # minimum-active guard so the market keeps a functioning advertiser base
        roi = outcome.advertiser_outcomes["roi"].to_numpy()
        active = int((~exit_mask).sum())
        if active > 20:
            low = roi < scenario.churn_threshold_roi
            cap = max(1, int(0.2 * active))
            if low.sum() > cap:
                worst = np.argsort(roi)[:cap]
                exit_mask[worst] = True
            else:
                exit_mask = exit_mask | low
        if prev is not None and _metric_delta(metrics, prev, scale_metrics) < scenario.tolerance:
            converged = True
            break
        prev = metrics
    # static contrast: same equilibrium/churn dynamics but NO advertiser strategic responses
    static_exit = np.zeros(len(ads), dtype=bool)
    static_metrics: dict[str, float] = {}
    for it in range(scenario.max_iters):
        outcome = mkt.run(
            cfg.policy,
            responses=[],
            seed=scenario.seed + 900 + it,
            initial_exit=static_exit,
        )
        static_metrics = _ecosystem_metrics(outcome)
        roi = outcome.advertiser_outcomes["roi"].to_numpy()
        active = int((~static_exit).sum())
        if active > 20:
            low = roi < scenario.churn_threshold_roi
            cap = max(1, int(0.2 * active))
            if low.sum() > cap:
                worst = np.argsort(roi)[:cap]
                static_exit[worst] = True
            else:
                static_exit = static_exit | low
    sens_rows: list[dict[str, float]] = []
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5):
        scaled = [
            ResponseHypothesis(
                advertiser_type=h.advertiser_type,
                behavior=h.behavior,
                magnitude=min(h.magnitude * mult, 0.6),
                probability=h.probability,
                source=f"perturb_{mult}",
            )
            for h in hypotheses
        ]
        sc = cfg.model_copy(update={"responses": scaled})
        m = _ecosystem_metrics(mkt.run(sc.policy, responses=scaled, seed=scenario.seed + int(mult * 100)))
        m["magnitude_multiplier"] = mult
        sens_rows.append(m)
    return EquilibriumResult(
        scenario=scenario,
        iterations=pd.DataFrame(rows),
        converged=converged,
        final_metrics=metrics,
        static_metrics=static_metrics,
        sensitivity=pd.DataFrame(sens_rows),
        response_hypotheses=hypotheses,
    )


def sensitivity_band(result: EquilibriumResult, key: str = "total_spend") -> tuple[float, float]:
    s = result.sensitivity
    return float(s[key].min()), float(s[key].max())
