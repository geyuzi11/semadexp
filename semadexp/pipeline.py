"""End-to-end pipeline orchestrating all four layers with real numbers."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EquilibriumScenario, PolicyChange, PolicyKind
from .data.corpora import generate_advertisers, generate_feedback_corpus, synthetic_behavior_dataset
from .eval.matrix import run_scenario_matrix
from .experiment.attribution import segment_cate
from .features.competition_graph import build_competition_graph
from .features.semantic import build_profiles, incremental_information_eval
from .hypothesis.closed_loop import run_closed_loop
from .hypothesis.closed_loop import _default_param
from .hypothesis.evaluate import evaluate_generation
from .hypothesis.generate import extract_candidate_hypotheses, score_priorities
from .hypothesis.knowledge_base import build_knowledge_base
from .llm.client import get_llm
from .render import render_report
from .simulator.equilibrium import run_equilibrium
from .simulator.equilibrium import llm_response_hypotheses
from .simulator.market import run_ground_truth


SCALE_DEFAULTS = {
    "quick": {"n": 80, "days": 5, "sessions": 300, "reps": 4, "kb": 30, "kb_days": 5, "kb_sessions": 300},
    "full": {"n": 110, "days": 8, "sessions": 500, "reps": 10, "kb": 60, "kb_days": 6, "kb_sessions": 400},
}


def run_pipeline(scale: str = "full", out_dir: str = "results", seed: int = 7) -> dict:
    cfg = SCALE_DEFAULTS[scale]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    llm = get_llm()

    # --- support: profiles, graph, prediction ablation ---
    advertisers = generate_advertisers(cfg["n"], seed)
    profiles = build_profiles(advertisers, llm)
    graph = build_competition_graph(profiles)
    behavior = synthetic_behavior_dataset(cfg["n"], seed)
    ablation = incremental_information_eval(behavior, profiles, seed=seed)

    # --- layer: design & estimation (scenario matrix) ---
    scenarios = [
        {"name": "低干扰-低异质性", "interference": 0.6, "heterogeneity": 0.6},
        {"name": "低干扰-高异质性", "interference": 0.6, "heterogeneity": 1.8},
        {"name": "高干扰-低异质性", "interference": 1.6, "heterogeneity": 0.6},
        {"name": "高干扰-高异质性", "interference": 1.6, "heterogeneity": 1.8},
    ]
    matrix = run_scenario_matrix(
        scenarios,
        n_advertisers=cfg["n"],
        days=cfg["days"],
        sessions_per_day=cfg["sessions"],
        n_reps=cfg["reps"],
        seed=seed,
    )

    # --- layer: hypothesis generation + quantification ---
    kb = build_knowledge_base(
        cfg["kb"], seed=seed, n_advertisers=cfg["n"], days=cfg["kb_days"], sessions_per_day=cfg["kb_sessions"], llm=llm
    )
    corpus = generate_feedback_corpus(seed=seed)
    hyps = extract_candidate_hypotheses(corpus, llm)
    ranked = score_priorities(hyps, kb)
    hyp_eval = evaluate_generation(ranked, kb, seed=seed)

    # --- layer: equilibrium simulation ---
    eq_policy = PolicyChange(kind=PolicyKind.LOWER_BID_FLOOR, param=0.6, name="降低竞价门槛")
    eq = run_equilibrium(
        EquilibriumScenario(policy=eq_policy, seed=seed),
        llm=llm,
        n_advertisers=cfg["n"],
        days_per_iter=4,
        sessions_per_day=300,
        seed=seed,
    )

    # --- layer: closed loop on top hypothesis ---
    sim_policy = ranked[0] if ranked else None
    loop = {}
    if sim_policy is not None:
        from .config import SimulationConfig

        gt_cfg = SimulationConfig(
            market={
                "n_advertisers": cfg["n"],
                "days": cfg["days"],
                "sessions_per_day": cfg["sessions"],
                "seed": seed,
            },
            policy=PolicyChange(
                kind=PolicyKind(sim_policy.policy_kind),
                param=_default_param(sim_policy.policy_kind),
            ),
            responses=llm_response_hypotheses(
                EquilibriumScenario(
                    policy=PolicyChange(
                        kind=PolicyKind(sim_policy.policy_kind),
                        param=_default_param(sim_policy.policy_kind),
                    ),
                    seed=seed,
                ),
                llm,
            ),
        )
        gt = run_ground_truth(gt_cfg, advertisers)
        loop = run_closed_loop(
            sim_policy, profiles, graph, gt, advertisers, sim_cfg=gt_cfg, seed=seed
        )

    report = render_report(
        {
            "ablation": ablation,
            "matrix": matrix,
            "hypothesis_eval": hyp_eval,
            "equilibrium": eq,
            "closed_loop": loop,
            "graph": graph,
            "profiles": profiles,
        },
        out,
    )

    # persist artifacts
    matrix["results"].to_csv(out / "matrix_results.csv", index=False)
    matrix["design"].to_csv(out / "matrix_design.csv", index=False)
    pd.DataFrame([hyp.__dict__ for hyp in ranked]).to_csv(out / "ranked_hypotheses.csv", index=False)
    pd.DataFrame(
        [
            {
                "k": k,
                "llm_capture": v,
                "random_capture": hyp_eval.random_baseline_capture[k],
                "manual_capture": hyp_eval.manual_baseline_capture[k],
            }
            for k, v in hyp_eval.top_k_capture.items()
        ]
    ).to_csv(out / "capture_curve.csv", index=False)
    summary = {
        "meta": {
            "scale": scale,
            "seed": seed,
            "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "ablation": ablation,
        "matrix": {
            "results": json.loads(matrix["results"].to_json(orient="records", force_ascii=False)),
            "design": json.loads(matrix["design"].to_json(orient="records", force_ascii=False)),
        },
        "hypothesis_eval": {
            "recall_at_top_x": hyp_eval.recall_at_top_x,
            "spearman": hyp_eval.spearman_priority_vs_true,
            "spearman_p": hyp_eval.spearman_p_value,
            "experiments_to_50pct": hyp_eval.experiments_to_50pct,
            "random_experiments_to_50pct": hyp_eval.random_experiments_to_50pct,
            "top_k_capture": {str(k): float(v) for k, v in hyp_eval.top_k_capture.items()},
            "random_baseline_capture": {str(k): float(v) for k, v in hyp_eval.random_baseline_capture.items()},
            "manual_baseline_capture": {str(k): float(v) for k, v in hyp_eval.manual_baseline_capture.items()},
        },
        "ranked_hypotheses": [_jsonable(h.__dict__) for h in ranked],
        "equilibrium": {
            "policy": eq.scenario.policy.label(),
            "converged": eq.converged,
            "final_metrics": _jsonable(eq.final_metrics),
            "static_metrics": _jsonable(eq.static_metrics),
            "sensitivity": json.loads(eq.sensitivity.to_json(orient="records", force_ascii=False)),
            "response_hypotheses": _jsonable([h.model_dump() for h in eq.response_hypotheses]),
        },
        "closed_loop": _closed_loop_summary(loop),
    }
    (out / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "ablation": ablation,
        "matrix": matrix,
        "ranked_hypotheses": ranked,
        "hypothesis_eval": hyp_eval,
        "equilibrium": eq,
        "closed_loop": loop,
        "report": report,
        "summary": summary,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return obj


def _closed_loop_summary(loop: dict) -> dict:
    if not loop:
        return {}
    d = loop.get("decomposition")
    return {
        "hypothesis": loop.get("hypothesis"),
        "policy": loop.get("policy"),
        "true_effect": loop.get("true_effect"),
        "design": _jsonable(loop.get("design")),
        "estimates": json.loads(loop["estimates"].to_json(orient="records", force_ascii=False)),
        "decomposition": {
            "total_effect": d.total_effect,
            "direct_effect": d.direct_effect,
            "indirect_effect": d.indirect_effect,
            "validation": _jsonable(d.validation),
        }
        if d is not None
        else {},
        "attribution": loop.get("attribution"),
    }
