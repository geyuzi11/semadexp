# %% [markdown]
# # 04 · 反事实均衡模拟 + 自动假设生成与量化评估
# 
# LLM 广告主响应假设 → 均衡迭代（含流失）→ 生态指标与灵敏度带；
# 非结构化语料 → 候选假设 → 去重/过滤/优先级打分 → 召回率、排序相关、Top-K 价值捕获。

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from semadexp.config import EquilibriumScenario, PolicyChange, PolicyKind
from semadexp.data.corpora import generate_feedback_corpus
from semadexp.hypothesis.evaluate import evaluate_generation
from semadexp.hypothesis.generate import extract_candidate_hypotheses, score_priorities
from semadexp.hypothesis.knowledge_base import build_knowledge_base
from semadexp.llm.client import DeterministicLLM
from semadexp.simulator.equilibrium import run_equilibrium

# %%
eq = run_equilibrium(
    EquilibriumScenario(policy=PolicyChange(kind=PolicyKind.LOWER_BID_FLOOR, param=0.6, name="降低竞价门槛")),
    n_advertisers=70,
    days_per_iter=4,
    sessions_per_day=300,
    seed=7,
)
print("均衡收敛:", eq.converged)
print("动态（含响应）:", {k: round(v, 3) for k, v in eq.final_metrics.items()})
print("静态（忽略响应）:", {k: round(v, 3) for k, v in eq.static_metrics.items()})
print("灵敏度带 total_spend: [%.1f, %.1f]" % (eq.sensitivity.total_spend.min(), eq.sensitivity.total_spend.max()))

# %%
kb = build_knowledge_base(30, seed=7, n_advertisers=60, days=4, sessions_per_day=250)
corpus = generate_feedback_corpus(seed=7)
ranked = score_priorities(extract_candidate_hypotheses(corpus, DeterministicLLM()), kb)
rep = evaluate_generation(ranked, kb, seed=7)

# %%
print("== 假设生成层量化 ==")
print("Top-8 机会召回率: %.0f%%" % (rep.recall_at_top_x * 100))
print("优先级 vs 真实效应 Spearman: %+.3f" % rep.spearman_priority_vs_true)
print("50%% 价值捕获所需实验: LLM %d 个 vs 随机基线 %d 个" % (rep.experiments_to_50pct, rep.random_experiments_to_50pct))
print("Top-K 捕获:", {k: round(v, 3) for k, v in rep.top_k_capture.items()})
print()
print("优先级排序:")
for h in ranked:
    print("  %+.4f  %s" % (h.priority, h.statement))

# %%
print("""
结论：LLM 从工单/负反馈/复盘等非结构化文本提炼假设并排序，显著压缩
找到高价值策略所需的实验次数；均衡模拟证明忽略广告主响应会系统性误估生态结果。
""")
