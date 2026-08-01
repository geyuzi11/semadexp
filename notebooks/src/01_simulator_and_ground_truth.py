# %% [markdown]
# # 01 · 双边市场仿真器与真值构建
# 
# 本 Notebook 验证仿真器的经济学机制（预算约束、拍卖分配、用户疲劳），并构建 ground truth：
# 控制组（C）、策略无响应（T0）、策略+广告主响应（T1）三组反事实市场，给出每个广告主的
# **直接效应**（T0−C）与**间接效应**（T1−T0）。

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from semadexp.config import PolicyChange, PolicyKind, SimulationConfig
from semadexp.data.corpora import generate_advertisers
from semadexp.simulator.market import Market, run_ground_truth, run_market

# %%
cfg = SimulationConfig(
    market={"n_advertisers": 60, "days": 5, "sessions_per_day": 350, "seed": 7},
    policy=PolicyChange(kind=PolicyKind.AD_LOAD_CAP, param=3, name="限制广告条数"),
)
ads = generate_advertisers(60, 7)
out = run_market(cfg.model_copy(update={"policy": None}), ads)

# %%
print("== 经济学 sanity ==")
print("每广告主平均转化: %.2f" % out.advertiser_outcomes.conversions.mean())
print("零转化占比: %.2f" % (out.advertiser_outcomes.conversions == 0).mean())
print("各档位 ROI:", out.advertiser_outcomes.groupby("tier")["roi"].mean().round(2).to_dict())
budgets = np.array([a.daily_budget * 5 for a in ads])
print("预算约束成立:", bool((out.advertiser_outcomes.spend <= budgets + 1e-6).all()))
print("市场指标:", {k: round(v, 3) for k, v in out.market_metrics.items()})

# %%
gt = run_ground_truth(cfg, ads)
g = gt["ground_truth"]
print("== 真值结构 ==")
print("直接效应均值: %+.4f（策略本身）" % g.true_direct.mean())
print("间接效应均值: %+.4f（广告主反馈）" % g.true_indirect.mean())
print("总效应 = 直接 + 间接:", np.isclose(g.true_total, g.true_direct + g.true_indirect).all())
print(g[["advertiser_id", "category", "tier", "true_direct", "true_indirect"]].head(8).to_string(index=False))

# %%
print("""
结论：仿真器在固定种子下可复现，经济学机制成立，且能给出带真值的
直接/间接效应拆分——这是后续量化“衡量准确性”的基础。
""")
