"""Command-line interface: python -m semadexp.cli <command>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    BucketingPolicy,
    EquilibriumScenario,
    PolicyChange,
    PolicyKind,
)
from .data.corpora import generate_advertisers, generate_feedback_corpus
from .eval.matrix import run_scenario_matrix
from .experiment.attribution import llm_interpret_segments, segment_cate
from .experiment.bucketing import assign_buckets, assignment_metrics
from .experiment.decomposition import decompose_total, fit_response_model, parse_operation_events
from .experiment.estimators import estimate_all
from .features.competition_graph import build_competition_graph
from .features.semantic import build_profiles
from .hypothesis.evaluate import evaluate_generation
from .hypothesis.generate import extract_candidate_hypotheses, score_priorities
from .hypothesis.knowledge_base import build_knowledge_base
from .llm.client import get_llm
from .pipeline import run_pipeline
from .simulator.equilibrium import run_equilibrium
from .simulator.market import run_ground_truth


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=110)


def cmd_build_graph(args) -> None:
    ads = generate_advertisers(args.n, args.seed)
    profiles = build_profiles(ads, get_llm())
    graph = build_competition_graph(profiles)
    print(
        json.dumps(
            {
                "advertisers": len(graph.advertiser_ids),
                "edges": len(graph.edges),
                "communities": len(graph.communities),
                "internal_density": graph.internal_density(),
                "cross_density": graph.cross_density(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_bucket(args) -> None:
    ads = generate_advertisers(args.n, args.seed)
    profiles = build_profiles(ads, get_llm())
    graph = build_competition_graph(profiles)
    assignment = assign_buckets(profiles, graph, BucketingPolicy(args.policy), seed=args.seed)
    print(json.dumps(assignment_metrics(profiles, graph, assignment), ensure_ascii=False, indent=2))


def cmd_decompose(args) -> None:
    from .config import SimulationConfig

    ads = generate_advertisers(args.n, args.seed)
    cfg = SimulationConfig(
        market={"n_advertisers": args.n, "days": 6, "sessions_per_day": 400, "seed": args.seed},
        policy=PolicyChange(kind=PolicyKind.AD_LOAD_CAP, param=3),
    )
    gt = run_ground_truth(cfg, ads)
    g = gt["ground_truth"]
    logs = []
    for _, row in g.iterrows():
        if abs(row["true_indirect"]) > 1e-9:
            logs.append(
                {
                    "advertiser_id": int(row["advertiser_id"]),
                    "event_type": "raise_budget",
                    "magnitude": min(abs(row["true_indirect"]) * 0.2, 0.3),
                    "day": 2,
                    "text": f"[提升日预算] 预算加了，幅度约{min(abs(row['true_indirect']) * 0.2, 0.3):.0%}",
                }
            )
    events = parse_operation_events(logs, get_llm())
    model = fit_response_model(g, events)
    df = g.rename(columns={"true_total": "y"})[["advertiser_id", "y"]].copy()
    df["arm"] = 1
    dec = decompose_total(df, events, model, g)
    print(
        json.dumps(
            {
                "response_model": model,
                "total": dec.total_effect,
                "direct": dec.direct_effect,
                "indirect": dec.indirect_effect,
                "validation": dec.validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_equilibrium(args) -> None:
    policy = PolicyChange(kind=PolicyKind(args.policy), param=args.param, name=args.name)
    eq = run_equilibrium(
        EquilibriumScenario(policy=policy, seed=args.seed),
        llm=get_llm(),
        n_advertisers=args.n,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "converged": eq.converged,
                "final": eq.final_metrics,
                "static": eq.static_metrics,
                "sensitivity": eq.sensitivity.to_dict(orient="records"),
                "hypotheses": [h.model_dump() for h in eq.response_hypotheses],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_hypotheses(args) -> None:
    kb = build_knowledge_base(args.n_experiments, seed=args.seed, llm=get_llm())
    corpus = generate_feedback_corpus(seed=args.seed)
    hyps = extract_candidate_hypotheses(corpus, get_llm())
    ranked = score_priorities(hyps, kb)
    for h in ranked:
        print(
            f"{h.priority:+.4f}  {h.statement}  [metric={h.metric}, confidence={h.confidence:.2f}, risk={h.risk:.2f}]"
        )


def cmd_eval_hypotheses(args) -> None:
    kb = build_knowledge_base(args.n_experiments, seed=args.seed, llm=get_llm())
    corpus = generate_feedback_corpus(seed=args.seed)
    ranked = score_priorities(extract_candidate_hypotheses(corpus, get_llm()), kb)
    rep = evaluate_generation(ranked, kb, seed=args.seed)
    print(
        json.dumps(
            {
                "recall_at_top_x": rep.recall_at_top_x,
                "spearman": rep.spearman_priority_vs_true,
                "spearman_p": rep.spearman_p_value,
                "experiments_to_50pct_llm": rep.experiments_to_50pct,
                "experiments_to_50pct_random": rep.random_experiments_to_50pct,
                "capture": rep.top_k_capture,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_batch(args) -> None:
    scenarios = [
        {"name": "低干扰-低异质性", "interference": 0.6, "heterogeneity": 0.6},
        {"name": "高干扰-高异质性", "interference": 1.6, "heterogeneity": 1.8},
    ]
    out = run_scenario_matrix(
        scenarios,
        n_advertisers=args.n,
        days=args.days,
        sessions_per_day=args.sessions,
        n_reps=args.reps,
        seed=args.seed,
    )
    print(out["results"].to_string(index=False))


def cmd_render(args) -> None:
    from .pipeline import run_pipeline

    results = run_pipeline(scale=args.scale, out_dir=args.out, seed=args.seed)
    print(f"report -> {results['report']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="semadexp", description="SemAdExp CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build-graph", help="build semantic competition graph")
    _common(sp)
    sp.set_defaults(func=cmd_build_graph)

    sp = sub.add_parser("bucket", help="assign buckets under a policy")
    _common(sp)
    sp.add_argument("--policy", choices=[e.value for e in BucketingPolicy], default="graph_cluster")
    sp.set_defaults(func=cmd_bucket)

    sp = sub.add_parser("decompose", help="decompose total effect into direct + indirect")
    _common(sp)
    sp.set_defaults(func=cmd_decompose)

    sp = sub.add_parser("simulate-equilibrium", help="counterfactual equilibrium simulation")
    _common(sp)
    sp.add_argument("--policy", choices=[e.value for e in PolicyKind], default="lower_bid_floor")
    sp.add_argument("--param", type=float, default=0.6)
    sp.add_argument("--name", type=str, default=None)
    sp.set_defaults(func=cmd_equilibrium)

    sp = sub.add_parser("generate-hypotheses", help="generate and rank hypotheses")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--n-experiments", type=int, default=30)
    sp.set_defaults(func=cmd_hypotheses)

    sp = sub.add_parser("eval-hypotheses", help="quantified hypothesis-layer evaluation")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--n-experiments", type=int, default=30)
    sp.set_defaults(func=cmd_eval_hypotheses)

    sp = sub.add_parser("batch", help="scenario matrix")
    _common(sp)
    sp.add_argument("--days", type=int, default=6)
    sp.add_argument("--sessions", type=int, default=400)
    sp.add_argument("--reps", type=int, default=6)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("render-report", help="run pipeline and render the technical report")
    sp.add_argument("--scale", choices=["quick", "full"], default="full")
    sp.add_argument("--out", default="results")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("run-all", help="full pipeline")
    sp.add_argument("--scale", choices=["quick", "full"], default="full")
    sp.add_argument("--out", default="results")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=lambda a: print(f"report -> {run_pipeline(a.scale, a.out, a.seed)['report']}"))

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

