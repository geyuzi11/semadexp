from semadexp.config import EquilibriumScenario, PolicyChange, PolicyKind
from semadexp.simulator.equilibrium import run_equilibrium


def test_equilibrium_runs_and_converges():
    eq = run_equilibrium(
        EquilibriumScenario(
            policy=PolicyChange(kind=PolicyKind.LOWER_BID_FLOOR, param=0.6),
            max_iters=4,
            seed=2,
        ),
        n_advertisers=50,
        days_per_iter=3,
        sessions_per_day=250,
        seed=2,
    )
    assert len(eq.iterations) >= 1
    assert "total_spend" in eq.final_metrics
    assert len(eq.sensitivity) == 5


def test_static_vs_dynamic_differs():
    eq = run_equilibrium(
        EquilibriumScenario(
            policy=PolicyChange(kind=PolicyKind.RAISE_BID_FLOOR, param=1.5),
            seed=3,
        ),
        n_advertisers=50,
        days_per_iter=3,
        sessions_per_day=250,
        seed=3,
    )
    assert eq.final_metrics["total_spend"] != eq.static_metrics["total_spend"]

