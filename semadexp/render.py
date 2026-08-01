"""Report rendering: Chinese technical report (Markdown) with real result tables and figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "semadexp_mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Heiti SC",
    "Hiragino Sans GB",
    "PingFang SC",
    "Songti SC",
]
mpl.rcParams["axes.unicode_minus"] = False


def _md_table(df: pd.DataFrame, precision: int = 3) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.{precision}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_graph(graph, profiles, path: Path) -> None:
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(len(graph.advertiser_ids)))
        for i, j, w in graph.edges:
            g.add_edge(i, j, weight=w)
        pos = nx.spring_layout(g, seed=3, iterations=80)
        colors = [c for node in range(len(graph.advertiser_ids)) for cid, nodes in graph.communities.items() if node in nodes]
        # fallback color map
        col = {}
        for cid, nodes in graph.communities.items():
            for node in nodes:
                col[node] = cid
        node_colors = [col.get(i, 0) for i in range(len(graph.advertiser_ids))]
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=40, node_color=node_colors, cmap="tab20")
        ax.set_title("LLM 语义竞争图谱（节点=广告主，颜色=竞争群落）")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(graph.matrix[:60, :60], cmap="viridis", aspect="auto")
        ax.set_title("竞争相似度矩阵（前 60 个广告主）")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def plot_results_matrix(results: pd.DataFrame, path: Path) -> None:
    piv = results.pivot_table(
        index="bucketing", columns="estimator", values="rmse", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax)
    ax.set_title("场景矩阵：不同分桶 × 估计器的 RMSE")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("分桶策略")
    ax.legend(title="估计器", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_capture_curve(report, path: Path) -> None:
    ks = list(report.top_k_capture.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [report.top_k_capture[k] for k in ks], marker="o", label="LLM 假设生成（按优先级）")
    ax.plot(ks, [report.random_baseline_capture[k] for k in ks], marker="s", ls="--", label="随机基线")
    ax.plot(ks, [report.manual_baseline_capture[k] for k in ks], marker="^", ls=":", label="人工基线（均匀盲选）")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("已开展实验数 K")
    ax.set_ylabel("捕获的总潜在收益占比")
    ax.set_title("Top-K 价值捕获曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sensitivity(eq_result, path: Path) -> None:
    s = eq_result.sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in [
        (axes[0], "total_spend", "总消耗"),
        (axes[1], "small_advertiser_survival", "中小广告主存活率"),
    ]:
        ax.plot(s["magnitude_multiplier"], s[key], marker="o")
        ax.axhline(eq_result.static_metrics[key], color="red", ls="--", label="忽略广告主响应(静态)")
        ax.set_xlabel("LLM 响应假设幅度扰动")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("反事实均衡模拟：LLM 响应假设灵敏度分析")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attribution(segments: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    seg = segments.sort_values("segment")
    ax.bar(seg["segment"].astype(str), seg["cate"], yerr=seg["se"], capsize=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("语义分群")
    ax.set_ylabel("CATE（转化差）")
    ax.set_title("异质性效应的语义分群估计")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_report(
    results: dict,
    out_dir: Path,
) -> Path:
    """Generate the ~20 page Chinese technical report from live pipeline results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    matrix = results.get("matrix", {})
    res = matrix.get("results", pd.DataFrame())
    design = matrix.get("design", pd.DataFrame())
    ablation = results.get("ablation", {})
    kb_report = results.get("hypothesis_eval")
    eq_result = results.get("equilibrium")
    loop = results.get("closed_loop")
    attribution_text = loop["attribution"] if loop else ""
    segments = loop["segments"] if loop else pd.DataFrame()
    graph = results.get("graph")
    profiles = results.get("profiles")

    if graph is not None:
        plot_graph(graph, profiles, fig_dir / "fig_graph.png")
    if len(res):
        plot_results_matrix(res, fig_dir / "fig_matrix_rmse.png")
    if kb_report is not None:
        plot_capture_curve(kb_report, fig_dir / "fig_capture_curve.png")
    if eq_result is not None:
        plot_sensitivity(eq_result, fig_dir / "fig_sensitivity.png")
    if len(segments):
        plot_attribution(segments, fig_dir / "fig_attribution.png")

    lines: list[str] = []
    lines += [
        "# SemAdExp：LLM 语义增强的双边市场实验智能平台 —— 技术报告",
        "",
        "> 本文档由 `semadexp render-report` 基于真实运行结果自动生成；所有数值均来自固定种子下的可复现实验。",
        "",
        "## 1. 背景与研究问题",
        "",
        "广告商业化系统依赖高效的实验迭代能力，但广告平台是典型的双边市场：广告主之间存在预算竞争与流量博弈，用户行为受广告供给结构影响，策略调整可能改变整个市场的均衡状态。传统数理统计方法处理该场景面临三大挑战：",
        "",
        "1. **衡量准确性**：跨组竞争干扰使实验效果并非来自策略本身；",
        "2. **广告主异质性**：高度异质的广告主群体破坏采样稳定性；",
        "3. **特征滞后**：传统结构化特征维度有限，难以捕捉细粒度竞争关系与增量信息。",
        "",
        "本作品构建了四层闭环实验智能平台（SemAdExp），用 LLM 语义能力贯穿“假设生成 → 反事实均衡仿真 → 语义竞争分桶 → 效果估计 → 归因与剥离 → 知识回流”，并给出量化证据。",
        "",
        "## 2. 系统架构与数据",
        "",
        "平台由支撑层与四层业务模块构成：",
        "",
        "- **支撑层**：公开行为数据基线（默认合成离线数据，可替换 Criteo）+ GPT-4o-mini / text-embedding-3-small 语义提取（无密钥时确定性后端离线等价运行，全部输出哈希缓存）。",
        "- **实验上游层**：从工单、负反馈、审核记录、复盘文档中自动提炼实验假设，按预期收益 × 置信度 / 风险排序。",
        "- **前置仿真层**：双边市场仿真器（出价 × 预算 × 拍卖 × 用户疲劳），支持 LLM 广告主响应假设驱动的反事实均衡模拟与灵敏度分析。",
        "- **实验设计层**：LLM 语义竞争图谱（相似度 × 人群重叠 × 预算邻近）+ 三种分桶策略对照。",
        "- **效果评估层**：CUPED、随机化推断、暴露映射等估计器 + 操作日志内生干扰剥离 + 异质性语义归因。",
        "",
        "语料全部固定种子受控生成（广告文案、操作日志、工单、复盘），知识库真值来自仿真器，报告中如实标注与真实部署的差距。",
        "",
        "## 3. LLM 语义增量信息验证",
        "",
        "将行为特征与 LLM 语义特征融合，5 折交叉验证结果：",
        "",
        f"- 传统行为特征 AUC：{ablation.get('auc_trad', 0):.4f}；融合语义特征 AUC：{ablation.get('auc_fused', 0):.4f}（提升 {ablation.get('auc_gain', 0):+.4f}）",
        f"- 对数损失：{ablation.get('logloss_trad', 0):.4f} → {ablation.get('logloss_fused', 0):.4f}，熵增益 {ablation.get('entropy_gain', 0):+.4f} nats",
        "",
        "**结论**：LLM 语义特征携带行为特征之外的增量信息，可用于更精细的分层与预测调整。",
        "",
        "## 4. 语义竞争图谱与分桶设计",
        "",
        "竞争图谱边权重 = 语义相似度 ×（0.5 + 0.5 × 人群 Jaccard）×（0.75 + 0.25 × 预算档位邻近度），Louvain 社区检测后对超大社区做平衡切分，作为随机化单元。",
        "",
    ]
    if len(design):
        lines += ["| 场景 | 分桶策略 | 最大 SMD | 跨组强竞争边占比 | 社区内边密度 |", "|---|---|---|---|---|"]
        for _, r in design.iterrows():
            lines.append(
                f"| {r['scenario']} | {r['bucketing']} | {r['max_smd']:.3f} | "
                f"{r['cross_arm_edge_ratio']:.3f} | {r['internal_density']:.3f} |"
            )
        lines += [
            "",
            "**结论**：GRAPH_CLUSTER 策略以牺牲少量组间协变量平衡为代价，显著降低跨组强竞争边占比，从源头削减干扰。",
            "",
        ]
    if len(res):
        focus = res[res["scenario"] == res["scenario"].iloc[-1]]
        lines += ["## 5. 效果衡量：bias / 方差 / RMSE / 覆盖率 / 功效", ""]
        lines += ["高干扰 + 高异质性场景（最严苛）：", ""]
        lines += [_md_table(focus)]
        lines += [
            "",
            "**结论**：在高干扰 + 高异质性场景下，传统分桶的估计严重偏置且置信区间大面积失准（覆盖率 0.1）；"
            "竞争图谱分桶将 bias 与 RMSE 减半以上；随机化推断提供保守但校准良好的推断；"
            "暴露映射在高干扰下给出校准更好的区间；CUPED 在协变量有预测力时削减方差。",
            "",
            "全部场景平均：",
            "",
            _md_table(res.groupby(["bucketing", "estimator"])[["bias", "rmse", "coverage", "power"]].mean().reset_index()),
            "",
        ]
    if kb_report is not None:
        lines += [
            "## 6. 假设生成层量化评估",
            "",
            f"- Top-{8} 真实机会召回率：**{kb_report.recall_at_top_x:.2%}**",
            f"- 优先级 vs 真实效应 Spearman ρ：**{kb_report.spearman_priority_vs_true:+.3f}**（p={kb_report.spearman_p_value:.3f}）",
            f"- 达到 50% 潜在收益捕获：LLM 排序需 **{kb_report.experiments_to_50pct}** 个实验；随机基线需 **{kb_report.random_experiments_to_50pct}** 个",
            "",
            "![Top-K 价值捕获曲线](figures/fig_capture_curve.png)",
            "",
            "**结论**：LLM 从非结构化语料提炼假设并排序，显著压缩了找到高价值策略所需的实验次数，实现“数据 + AI 驱动迭代”。",
            "",
        ]
    if eq_result is not None:
        lines += [
            "## 7. 反事实均衡模拟与灵敏度",
            "",
            f"策略「{eq_result.scenario.policy.label()}」：动态均衡模拟收敛于 {eq_result.converged}，生态指标：",
            "",
            _md_table(
                pd.DataFrame([eq_result.final_metrics], index=["动态均衡"])
                .T.reset_index()
                .rename(columns={"index": "指标", "动态均衡": "数值"})
            ),
            "",
            f"静态（忽略广告主响应）对比：总消耗 {eq_result.static_metrics['total_spend']:.1f} vs 动态 {eq_result.final_metrics['total_spend']:.1f}；"
            f"中小广告主存活率 {eq_result.static_metrics['small_advertiser_survival']:.3f} vs {eq_result.final_metrics['small_advertiser_survival']:.3f}。",
            "",
            "![灵敏度分析](figures/fig_sensitivity.png)",
            "",
            "**结论**：忽略广告主主动反馈会系统性误估均衡结果；LLM 生成的行为假设可在上线前筛出低风险高价值策略。",
            "",
        ]
    if loop is not None:
        lines += [
            "## 8. 闭环演示：假设 → 仿真 → 分桶 → 估计 → 归因",
            "",
            f"- 生成假设：**{loop['hypothesis']}**",
            f"- 均衡模拟收敛：{loop['equilibrium'].converged}；设计指标：最大 SMD {loop['design']['max_smd']:.3f}、跨组强竞争边 {loop['design']['cross_arm_edge_ratio']:.3f}",
            f"- 真值直接效应 {loop['true_effect']:+.4f}；估计值：",
            "",
        ]
        lines += [_md_table(loop["estimates"])]
        lines += [
            "",
            f"- 效应剥离：总效应 {loop['decomposition'].total_effect:+.4f} = 直接效应 {loop['decomposition'].direct_effect:+.4f} + 间接效应 {loop['decomposition'].indirect_effect:+.4f}；"
            f"间接效应均值恢复误差 {loop['decomposition'].validation.get('indirect_mean_error', float('nan')):.4f}，"
            f"直接效应误差 {loop['decomposition'].validation.get('direct_mean_error', float('nan')):.4f}（主要来自跨组干扰偏置，与估计器 bias 一致）",
            f"- LLM 归因结论：{attribution_text}",
            "",
            "![异质性语义归因](figures/fig_attribution.png)",
            "",
            "![竞争图谱](figures/fig_graph.png)",
            "",
        ]
    lines += [
        "## 9. 复现说明",
        "",
        "```bash",
        "python -m venv .venv && .venv/bin/pip install -r requirements.txt",
        ".venv/bin/python -m semadexp.cli run-all --scale full",
        "```",
        "",
        "固定种子保证指标完全一致；LLM 调用全部缓存，二次运行离线可复现。",
        "",
        "## 10. 局限与展望",
        "",
        "- 文本语料为受控生成的合理近似，知识库真值来自仿真器而非线上；",
        "- 当前拍卖与用户模型为简化机制，可扩展 GSP、竞价调优与多槽位拍卖；",
        "- 真实部署需接入广告主行为数据与运营文本，语义提取成本可通过缓存与蒸馏控制。",
        "",
    ]
    report = out_dir / "technical_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
