# %% [markdown]
# # 02 · LLM 语义特征与竞争图谱
# 
# 广告文案 → 结构化语义标签 + 语义向量 → 广告主画像 → 语义竞争图谱。
# 本 Notebook 还给出“语义增量信息”的量化证据（AUC / 对数损失 / 熵增益）。

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from semadexp.data.corpora import generate_advertisers, synthetic_behavior_dataset
from semadexp.features.competition_graph import build_competition_graph
from semadexp.features.semantic import build_profiles, incremental_information_eval
from semadexp.llm.client import DeterministicLLM

# %%
llm = DeterministicLLM()  # 无 API Key 时离线可用；设置 OPENAI_API_KEY 后自动切换真实模型
ads = generate_advertisers(80, 7)
profiles = build_profiles(ads, llm)
print("广告主画像字段:", [c for c in profiles.columns if c not in ("embedding",)])
print("示例画像:")
print(profiles[["advertiser_id", "category", "tier", "tag_selling_points"]].head(5).to_string(index=False))

# %%
beh = synthetic_behavior_dataset(80, 7)
abl = incremental_information_eval(beh, profiles, seed=7)
print("== 语义增量信息（5 折交叉验证）==")
print("AUC: 传统 %.4f → 融合 %.4f（提升 %+.4f）" % (abl["auc_trad"], abl["auc_fused"], abl["auc_gain"]))
print("LogLoss: %.4f → %.4f（熵增益 %+.4f nats）" % (abl["logloss_trad"], abl["logloss_fused"], abl["entropy_gain"]))

# %%
graph = build_competition_graph(profiles)
print("== 竞争图谱 ==")
print("广告主节点:", len(graph.advertiser_ids), "强竞争边:", len(graph.edges))
print("竞争群落数:", len(graph.communities))
print("群落内边密度: %.3f | 跨群落边密度: %.3f" % (graph.internal_density(), graph.cross_density()))

# %%
print("""
结论：LLM 语义特征携带行为特征之外的增量信息；竞争图谱识别出强竞争群落，
为“竞争感知分桶”提供随机化单元。
""")
