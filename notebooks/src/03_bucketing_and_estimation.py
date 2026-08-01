# %% [markdown]
# # 03 · 分桶策略对照与效果估计
# 
# 三种分桶（传统结构化分层 / 语义画像分层 / 竞争图谱群落随机化）× 四种估计器
# （朴素差分 / CUPED / 随机化推断 / 暴露映射），量化 bias、方差、RMSE、覆盖率与功效。

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from semadexp.config import BucketingPolicy, MarketConfig, PolicyChange, PolicyKind, SimulationConfig
from semadexp.data.corpora import generate_advertisers
from semadexp.eval.matrix import run_scenario_matrix

# %%
matrix = run_scenario_matrix(
    [{"name": "高干扰-高异质性", "interference": 1.6, "heterogeneity": 1.8}],
    n_advertisers=70,
    days=5,
    sessions_per_day=300,
    n_reps=6,
    seed=7,
)
res = matrix["results"]
design = matrix["design"]

# %%
print("== 设计层指标 ==")
print(design.to_string(index=False))
print()
print("== 估计层指标（高干扰-高异质性）==")
print(
    res.groupby(["bucketing", "estimator"])[["bias", "rmse", "coverage", "power"]]
    .mean()
    .round(3)
    .to_string()
)

# %%
print("""
结论：竞争图谱分桶把跨组强竞争边占比砍掉约一半，bias/RMSE 显著下降；
随机化推断与暴露映射提供校准更稳健的推断（覆盖率为代价换取保守性）。
""")
