# %% [markdown]
# # 05 · 闭环演示：假设 → 仿真 → 分桶 → 估计 → 归因
# 
# 从假设生成层取 Top-1 假设，跑通全链路，展示每层的产物与证据，量化闭环效率。

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from semadexp.config import EquilibriumScenario, PolicyChange, PolicyKind, SimulationConfig
from semadexp.data.corpora import generate_advertisers, generate_feedback_corpus
from semadexp.features.competition_graph import build_competition_graph
from semadexp.features.semantic import build_profiles
from semadexp.hypothesis.closed_loop import run_closed_loop
from semadexp.hypothesis.closed_loop import _default_param
from semadexp.hypothesis.generate import extract_candidate_hypotheses, score_priorities
from semadexp.hypothesis.knowledge_base import build_knowledge_base
from semadexp.llm.client import DeterministicLLM
from semadexp.simulator.market import run_ground_truth
from semadexp.simulator.equilibrium import llm_response_hypotheses

# %%
seed = 7
llm = DeterministicLLM()
ads = generate_advertisers(70, seed)
profiles = build_profiles(ads, llm)
graph = build_competition_graph(profiles)

kb = build_knowledge_base(20, seed=seed, n_advertisers=60, days=4, sessions_per_day=250)
ranked = score_priorities(extract_candidate_hypotheses(generate_feedback_corpus(seed=seed), llm), kb)
hyp = ranked[0]
print("Top-1 生成假设:", hyp.statement)

# %%
gt_cfg = SimulationConfig(
    market={"n_advertisers": 70, "days": 5, "sessions_per_day": 300, "seed": seed},
    policy=PolicyChange(kind=PolicyKind(hyp.policy_kind), param=_default_param(hyp.policy_kind)),
    responses=llm_response_hypotheses(
        EquilibriumScenario(policy=PolicyChange(kind=PolicyKind(hyp.policy_kind), param=_default_param(hyp.policy_kind))),
        llm,
    ),
)
gt = run_ground_truth(gt_cfg, ads)
loop = run_closed_loop(hyp, profiles, graph, gt, ads, sim_cfg=gt_cfg, seed=seed)

# %%
print("== 设计 ==")
print("最大 SMD %.3f | 跨组强竞争边 %.3f" % (loop["design"]["max_smd"], loop["design"]["cross_arm_edge_ratio"]))
print("== 估计 ==")
print(loop["estimates"][["estimator", "estimate", "se", "ci_low", "ci_high"]].to_string(index=False))
d = loop["decomposition"]
print("== 效应剥离 ==")
print("总 %+.4f = 直接 %+.4f + 间接 %+.4f；间接均值恢复误差 %.4f"
      % (d.total_effect, d.direct_effect, d.indirect_effect, d.validation["indirect_mean_error"]))
print("== LLM 归因 ==")
print(loop["attribution"])

# %%
print("""
闭环完成：假设 → 均衡仿真（风险预筛）→ 竞争图谱分桶 → 实验估计 →
效应剥离 → 语义归因，全链路证据可直接写入项目报告。
""")
