"""Endogenous interference stripping: parse advertiser operation logs, model the
indirect feedback effect, and decompose total effect = direct + indirect."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BehaviorKind
from ..llm.client import LLMClient


@dataclass
class EffectDecomposition:
    total_effect: float
    direct_effect: float
    indirect_effect: float
    per_advertiser: pd.DataFrame
    validation: dict = field(default_factory=dict)


ACTION_PATTERNS: list[tuple[str, BehaviorKind, float]] = [
    (r"提升日预算|预算加了|加预算", BehaviorKind.RAISE_BUDGET, 0.15),
    (r"调高出价|出价往上调", BehaviorKind.RAISE_BID, 0.10),
    (r"更换素材|换了新文案|新素材", BehaviorKind.CHANGE_CREATIVE, 0.15),
    (r"扩定向|扩展了定向", BehaviorKind.EXPAND_TARGETING, 0.10),
    (r"暂停投放|先暂停", BehaviorKind.EXIT, 0.30),
]


def parse_operation_events(logs: list[dict], llm: LLMClient | None = None) -> pd.DataFrame:
    """Parse free-text operation logs into typed events. OpenAI backend returns JSON;
    deterministic backend uses pattern matching with magnitude extraction."""
    rows: list[dict] = []
    if llm is not None and llm.name == "OpenAILLM":
        system = (
            "从广告主操作日志中提取事件列表：advertiser_id、event_type("
            "raise_budget/raise_bid/change_creative/expand_targeting/exit)、magnitude(0-0.6)、day。"
            "返回 JSON: {events: [...]}"
        )
        try:
            raw = llm.extract_structured(system, "\n".join(str(l) for l in logs))
            for e in raw.get("events", []):
                rows.append(
                    {
                        "advertiser_id": int(e.get("advertiser_id", -1)),
                        "event_type": str(e.get("event_type", "")),
                        "magnitude": float(e.get("magnitude", 0.1)),
                        "day": int(e.get("day", 1)),
                        "text": "",
                    }
                )
            return pd.DataFrame(rows)
        except Exception:
            pass
    for log in logs:
        text = str(log.get("text", ""))
        for pattern, kind, default_mag in ACTION_PATTERNS:
            if re.search(pattern, text):
                m = re.search(r"幅度约([\d.]+)%", text)
                mag = float(m.group(1)) / 100.0 if m else default_mag
                rows.append(
                    {
                        "advertiser_id": int(log["advertiser_id"]),
                        "event_type": kind.value,
                        "magnitude": mag,
                        "day": int(log.get("day", 1)),
                        "text": text,
                    }
                )
                break
    return pd.DataFrame(rows)


def fit_response_model(
    ground_truth: pd.DataFrame, events: pd.DataFrame
) -> dict:
    """Regression: true_indirect ~ response intensity (calibrated on global counterfactual runs)."""
    intensity = events.groupby("advertiser_id")["magnitude"].sum()
    df = ground_truth.merge(intensity.rename("intensity"), on="advertiser_id", how="left")
    df["intensity"] = df["intensity"].fillna(0.0)
    x = df["intensity"].to_numpy(float)
    y = df["true_indirect"].to_numpy(float)
    beta, intercept = np.polyfit(x, y, 1)
    resid = y - (beta * x + intercept)
    r2 = 1 - resid.var() / max(y.var(), 1e-18)
    return {"beta": float(beta), "intercept": float(intercept), "r2": float(r2)}


def decompose_total(
    experiment_df: pd.DataFrame,
    events: pd.DataFrame,
    response_model: dict,
    ground_truth: pd.DataFrame | None = None,
) -> EffectDecomposition:
    """Decompose the observed total effect per advertiser into direct + indirect."""
    intensity = events.groupby("advertiser_id")["magnitude"].sum()
    df = experiment_df[["advertiser_id", "y"]].merge(intensity.rename("intensity"), on="advertiser_id", how="left")
    df["intensity"] = df["intensity"].fillna(0.0)
    indirect_hat = response_model["beta"] * df["intensity"].to_numpy() + response_model["intercept"]
    total_hat = df["y"].to_numpy()
    direct_hat = total_hat - indirect_hat
    df["indirect_hat"] = indirect_hat
    df["direct_hat"] = direct_hat
    validation: dict = {}
    if ground_truth is not None:
        v = df.merge(
            ground_truth[["advertiser_id", "true_direct", "true_indirect"]],
            on="advertiser_id",
            how="inner",
        )
        validation = {
            "direct_mean_error": float(
                abs(v["direct_hat"].mean() - v["true_direct"].mean())
            ),
            "indirect_mean_error": float(
                abs(v["indirect_hat"].mean() - v["true_indirect"].mean())
            ),
            "mae_direct": float(np.abs(v["direct_hat"] - v["true_direct"]).mean()),
            "mae_indirect": float(np.abs(v["indirect_hat"] - v["true_indirect"]).mean()),
            "rmse_per_advertiser": float(
                np.sqrt(np.mean((v["direct_hat"] + v["indirect_hat"] - (v["true_direct"] + v["true_indirect"])) ** 2))
            ),
        }
    return EffectDecomposition(
        total_effect=float(total_hat.mean()),
        direct_effect=float(direct_hat.mean()),
        indirect_effect=float(indirect_hat.mean()),
        per_advertiser=df,
        validation=validation,
    )
