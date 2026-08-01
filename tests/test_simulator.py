import numpy as np

from semadexp.config import PolicyChange, PolicyKind, SimulationConfig
from semadexp.data.corpora import generate_advertisers
from semadexp.simulator.market import Market, run_ground_truth, run_market


def _cfg(n=40, days=3, sessions=200, seed=1, **kw):
    return SimulationConfig(
        market={"n_advertisers": n, "days": days, "sessions_per_day": sessions, "seed": seed, **kw}
    )


def test_budget_constraint_holds():
    cfg = _cfg()
    ads = generate_advertisers(cfg.market.n_advertisers, cfg.market.seed)
    out = run_market(cfg, ads)
    budgets = np.array([a.daily_budget * cfg.market.days for a in ads])
    assert np.all(out.advertiser_outcomes["spend"] <= budgets + 1e-6)


def test_no_policy_recovery():
    """With no policy change, per-advertiser effects should be near zero."""
    cfg = _cfg()
    gt = run_ground_truth(cfg)
    g = gt["ground_truth"]
    assert abs(g["true_direct"].mean()) < 1.0


def test_policy_changes_market():
    cfg = _cfg()
    t = run_market(cfg)
    cfg2 = cfg.model_copy(deep=True)
    cfg2.policy = PolicyChange(kind=PolicyKind.RAISE_BID_FLOOR, param=2.0)
    t2 = run_market(cfg2)
    assert t2.market_metrics["total_spend"] <= t.market_metrics["total_spend"] + 1e-6


def test_experiment_arms_interact():
    cfg = _cfg()
    ads = generate_advertisers(cfg.market.n_advertisers, cfg.market.seed)
    rng = np.random.default_rng(0)
    arm = rng.integers(0, 2, len(ads))
    cfg.policy = PolicyChange(kind=PolicyKind.BUDGET_CAP_MULTIPLIER, param=2.0)
    out = run_market(cfg, ads, arm=arm)
    assert set(out.advertiser_outcomes["arm"].unique()) <= {0, 1}

