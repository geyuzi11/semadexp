from semadexp.config import PolicyChange, PolicyKind, SimulationConfig
from semadexp.data.corpora import generate_advertisers, generate_operation_logs
from semadexp.experiment.decomposition import (
    decompose_total,
    fit_response_model,
    parse_operation_events,
)
from semadexp.simulator.market import run_ground_truth


def test_decomposition_recovers_known_split():
    cfg = SimulationConfig(
        market={"n_advertisers": 50, "days": 4, "sessions_per_day": 300, "seed": 4},
        policy=PolicyChange(kind=PolicyKind.AD_LOAD_CAP, param=3),
        responses=[
            {
                "advertiser_type": "S",
                "behavior": "raise_budget",
                "magnitude": 0.2,
                "probability": 1.0,
            }
        ],
    )
    ads = generate_advertisers(50, 4)
    gt = run_ground_truth(cfg, ads)
    g = gt["ground_truth"]
    logs = generate_operation_logs(
        ads,
        [(parse_behavior("raise_budget"), 0.2)],
        seed=4,
        advertiser_type="S",
        probability=1.0,
    )
    events = parse_operation_events(logs)
    model = fit_response_model(g, events)
    df = g.rename(columns={"true_total": "y"})[["advertiser_id", "y"]].copy()
    df["arm"] = 1
    dec = decompose_total(df, events, model, g)
    assert dec.validation["indirect_mean_error"] < 0.5
    assert dec.validation["direct_mean_error"] < 0.5
    assert abs(dec.total_effect - (dec.direct_effect + dec.indirect_effect)) < 1e-9


def parse_behavior(name: str):
    from semadexp.config import BehaviorKind

    return BehaviorKind(name)
