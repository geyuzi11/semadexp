"""Two-sided market simulator: advertisers x users x auction, with budget competition,
user fatigue and supply substitution. The simulator exposes ground-truth potential
outcomes via global counterfactual runs (control / treatment-no-response / treatment-with-response).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    AuctionType,
    BehaviorKind,
    MarketConfig,
    PolicyChange,
    ResponseHypothesis,
    SimulationConfig,
)
from ..data.corpora import AdvertiserSeed, generate_advertisers


@dataclass
class MarketOutcome:
    advertiser_outcomes: pd.DataFrame
    market_metrics: dict[str, float]


def _semantic_vectors(advertisers: list[AdvertiserSeed], dim: int = 16, seed: int = 0) -> np.ndarray:
    """Deterministic category-prototype semantics used when no LLM profile vectors are supplied."""
    rng = np.random.default_rng(seed)
    n_cat = max(ad.category_idx for ad in advertisers) + 1
    protos = rng.normal(size=(n_cat, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    style = rng.normal(0, 0.25, size=(len(advertisers), dim))
    vecs = protos[[ad.category_idx for ad in advertisers]] + style
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs.astype(np.float32)


class Market:
    """Runs the daily auction market. ``arm`` is None (global mode) or a 0/1 array
    assigning each advertiser to control/treatment within one shared market."""

    def __init__(
        self,
        advertisers: list[AdvertiserSeed],
        config: MarketConfig | None = None,
        ad_vectors: np.ndarray | None = None,
    ):
        self.ads = advertisers
        self.cfg = config or MarketConfig()
        self.vec = (
            ad_vectors.astype(np.float32)
            if ad_vectors is not None
            else _semantic_vectors(advertisers, seed=self.cfg.seed)
        )
        self.n = len(advertisers)
        self.cat_idx = np.array([a.category_idx for a in advertisers], dtype=int)
        self.tier_idx = np.array([0 if a.tier == "S" else (1 if a.tier == "M" else 2) for a in advertisers])
        self.bids = np.array([a.initial_bid for a in advertisers], dtype=float)
        self.quality = np.array([a.quality for a in advertisers], dtype=float)
        self.budgets = np.array([a.daily_budget for a in advertisers], dtype=float)
        self.base_ctr = np.array([a.base_ctr for a in advertisers], dtype=float)
        self.value = np.array([a.value_per_conversion for a in advertisers], dtype=float)
        self.audience = [set(a.audience_tags) for a in advertisers]

    # ------------------------------------------------------------------ setup
    def _session_interest(self, rng: np.random.Generator) -> np.ndarray:
        alpha = np.full(self.cfg.n_categories, 0.8 * self.cfg.heterogeneity + 0.3)
        w = rng.dirichlet(alpha)
        noise = rng.normal(0, 0.2, self.vec.shape[1])
        u = np.zeros(self.vec.shape[1], dtype=float)
        for ci, wi in enumerate(w):
            mask = self.cat_idx == ci
            if mask.any():
                u += wi * self.vec[mask].mean(axis=0)
        u += noise
        n = np.linalg.norm(u)
        return u / (n + 1e-9)

    # ------------------------------------------------------------------ run
    def run(
        self,
        policy: PolicyChange | None = None,
        responses: list[ResponseHypothesis] | None = None,
        arm: np.ndarray | None = None,
        seed: int | None = None,
        initial_exit: np.ndarray | None = None,
    ) -> MarketOutcome:
        cfg = self.cfg
        rng = np.random.default_rng(seed if seed is not None else cfg.seed)
        n = self.n
        if arm is None:
            arm = np.zeros(n, dtype=int)
        arm = np.asarray(arm, dtype=int)

        budget_mult = np.ones(n)
        floor_factor = np.ones(n)
        load_cap = np.full(n, cfg.candidates_per_session)
        gsp_mask = np.zeros(n, dtype=bool)
        if policy is not None:
            treated = arm == 1
            if policy.kind.value == "budget_cap_multiplier":
                budget_mult[treated] = policy.param
            elif policy.kind.value == "lower_bid_floor":
                floor_factor[treated] = policy.param
            elif policy.kind.value == "raise_bid_floor":
                floor_factor[treated] = policy.param
            elif policy.kind.value == "ad_load_cap":
                load_cap[treated] = max(1, int(policy.param))
            elif policy.kind.value == "auction_rule":
                gsp_mask = treated

        # advertiser active responses (post-policy adaptation)
        resp_mult = np.ones((n, 3))  # budget, bid, relevance refresh
        exit_mask = (
            np.asarray(initial_exit, dtype=bool).copy()
            if initial_exit is not None
            else np.zeros(n, dtype=bool)
        )
        targeting_expand = np.zeros(n, dtype=bool)
        for resp in responses or []:
            if resp.probability < 1.0 and rng.random() > resp.probability:
                continue
            if resp.behavior == BehaviorKind.RAISE_BUDGET:
                m = self._type_mask(resp.advertiser_type)
                resp_mult[m, 0] *= 1.0 + resp.magnitude
            elif resp.behavior == BehaviorKind.RAISE_BID:
                m = self._type_mask(resp.advertiser_type)
                resp_mult[m, 1] *= 1.0 + resp.magnitude
            elif resp.behavior == BehaviorKind.CHANGE_CREATIVE:
                m = self._type_mask(resp.advertiser_type)
                resp_mult[m, 2] *= 1.0 + 0.5 * resp.magnitude
            elif resp.behavior == BehaviorKind.EXPAND_TARGETING:
                m = self._type_mask(resp.advertiser_type)
                targeting_expand[m] = True
            elif resp.behavior == BehaviorKind.EXIT:
                m = self._type_mask(resp.advertiser_type)
                exit_mask[m] = True

        budgets = self.budgets * budget_mult * resp_mult[:, 0]
        bids = self.bids * floor_factor * resp_mult[:, 1]
        quality = self.quality * (1.0 + 0.3 * (resp_mult[:, 2] - 1.0))
        vec = self.vec * (1.0 + 0.4 * (resp_mult[:, 2] - 1.0))[:, None]
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        reach_boost = 1.0 + 0.35 * targeting_expand.astype(float)

        horizon = cfg.days
        impressions = np.zeros(n)
        conversions = np.zeros(n)
        spend = np.zeros(n)
        neg_feedback = 0.0
        active_days = np.zeros(n)

        floor = cfg.bid_floor
        sessions = cfg.sessions_per_day
        k_cand = cfg.candidates_per_session
        inter = cfg.interference_strength

        for day in range(horizon):
            remaining = budgets.copy()
            for _ in range(sessions):
                interest = self._session_interest(rng)
                relevance = ((self.vec @ interest) + 1.0) / 2.0
                relevance = np.clip(relevance, 0.05, 1.0)
                score = bids * quality * relevance * reach_boost
                eligible = (remaining > 0.0) & (score >= floor) & (~exit_mask)
                if not eligible.any():
                    continue
                idx = np.where(eligible)[0]
                # stochastic candidate selection keeps long-tail advertisers competitive
                p = np.exp((score[idx] - score[idx].max()) / 0.4)
                p /= p.sum()
                k = min(k_cand, len(idx))
                chosen = rng.choice(idx, size=k, replace=False, p=p)
                shown = chosen[np.argsort(score[chosen])[::-1]]
                # per-advertiser ad load cap (treated ads only)
                if policy is not None and policy.kind.value == "ad_load_cap" and int(policy.param) < k_cand:
                    cap = int(policy.param)
                    t_show = shown[arm[shown] == 1]
                    if len(t_show) > cap:
                        keep_t = t_show[:cap]
                        c_show = shown[arm[shown] == 0]
                        shown = np.concatenate([c_show, keep_t]) if len(c_show) else keep_t
                if len(shown) == 0:
                    continue
                fatigue = 0.0
                for pos, adi in enumerate(shown):
                    p_ctr = self.base_ctr[adi] * relevance[adi] * float(np.exp(-0.15 * fatigue * inter))
                    impressions[adi] += 1
                    active_days[adi] += 1 / horizon
                    if rng.random() < p_ctr:
                        if rng.random() < 0.5 * quality[adi]:
                            conversions[adi] += 1
                    if pos > 0 and rng.random() < 0.02 * inter * pos:
                        neg_feedback += 1.0
                    if gsp_mask[adi] and pos + 1 < len(shown):
                        price = min(bids[adi] * (score[shown[pos + 1]] / max(score[adi], 1e-9)), bids[adi])
                        price = max(price, floor)
                    else:
                        price = bids[adi]
                    price = min(price, remaining[adi])
                    spend[adi] += price
                    remaining[adi] -= price
                    fatigue += 1.0

        neg_rate = neg_feedback / (sessions * horizon)
        profit = conversions * self.value - spend
        roi = np.where(spend > 0, (conversions * self.value) / np.maximum(spend, 1e-9), 0.0)
        out = pd.DataFrame(
            {
                "advertiser_id": np.arange(n),
                "category": self.cat_idx,
                "tier": self.tier_idx,
                "arm": arm,
                "impressions": impressions,
                "conversions": conversions,
                "spend": spend,
                "profit": profit,
                "roi": roi,
                "active_days": active_days,
            }
        )
        return MarketOutcome(
            advertiser_outcomes=out,
            market_metrics={
                "total_spend": float(spend.sum()),
                "total_conversions": float(conversions.sum()),
                "negative_feedback_rate": float(neg_rate),
                "advertiser_survival": float((active_days > 0.5).mean()),
            },
        )

    def _type_mask(self, advertiser_type: str) -> np.ndarray:
        if advertiser_type in ("*", "all"):
            return np.ones(self.n, dtype=bool)
        if advertiser_type in ("S", "M", "L"):
            return self.tier_idx == {"S": 0, "M": 1, "L": 2}[advertiser_type]
        return self.cat_idx == self._cat_index(advertiser_type)

    @staticmethod
    def _cat_index(name: str) -> int:
        names = ["美妆个护", "3C数码", "服饰鞋包", "食品饮料", "本地生活", "游戏娱乐", "金融信贷", "教育培训"]
        return names.index(name) if name in names else 0


def run_market(
    config: SimulationConfig,
    advertisers: list[AdvertiserSeed] | None = None,
    arm: np.ndarray | None = None,
    ad_vectors: np.ndarray | None = None,
    seed: int | None = None,
) -> MarketOutcome:
    ads = advertisers if advertisers is not None else generate_advertisers(config.market.n_advertisers, config.market.seed)
    mkt = Market(ads, config.market, ad_vectors=ad_vectors)
    return mkt.run(config.policy, config.responses, arm=arm, seed=seed)


def run_ground_truth(
    config: SimulationConfig,
    advertisers: list[AdvertiserSeed] | None = None,
    ad_vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Global counterfactual runs: control (C), treatment-no-response (T0), treatment-with-response (T1).
    Returns per-advertiser true direct effect (T0-C) and true indirect effect (T1-T0)."""
    c = run_market(config, advertisers, ad_vectors=ad_vectors, seed=config.market.seed + 1000)
    t0_cfg = config.model_copy(update={"responses": []})
    t0 = run_market(t0_cfg, advertisers, ad_vectors=ad_vectors, seed=config.market.seed + 2000)
    t1 = run_market(config, advertisers, ad_vectors=ad_vectors, seed=config.market.seed + 3000)
    df = c.advertiser_outcomes[["advertiser_id", "category", "tier"]].copy()
    df["y_control"] = c.advertiser_outcomes["conversions"]
    df["y_t0"] = t0.advertiser_outcomes["conversions"]
    df["y_t1"] = t1.advertiser_outcomes["conversions"]
    df["true_direct"] = df["y_t0"] - df["y_control"]
    df["true_indirect"] = df["y_t1"] - df["y_t0"]
    df["true_total"] = df["true_direct"] + df["true_indirect"]
    return {
        "control": c,
        "t0": t0,
        "t1": t1,
        "ground_truth": df,
    }


def run_experiment(
    config: SimulationConfig,
    assignment: np.ndarray,
    advertisers: list[AdvertiserSeed] | None = None,
    ad_vectors: np.ndarray | None = None,
    seed: int | None = None,
) -> MarketOutcome:
    """One experimental run with a shared market and an advertiser-level treatment assignment."""
    return run_market(config, advertisers, arm=np.asarray(assignment, dtype=int), ad_vectors=ad_vectors, seed=seed)
