"""Corpus generation: advertisers, ad creatives, operation logs, feedback texts.

All corpora are generated with fixed seeds. The creative templates embed the same
keywords the deterministic LLM backend parses, so the OpenAI backend and the
offline backend share the same contracts. Real public CTR data (Criteo) can be
plugged in via ``data/criteo/``; ``synthetic_behavior_dataset`` is the offline default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import BehaviorKind


CATEGORIES: list[dict[str, Any]] = [
    {
        "name": "美妆个护",
        "selling_points": ["价格优惠", "成分安全", "新品上市", "限量礼盒"],
        "audiences": ["女性", "白领", "学生"],
        "tones": ["促销", "情感"],
    },
    {
        "name": "3C数码",
        "selling_points": ["性能强悍", "性价比", "新品首发", "以旧换新"],
        "audiences": ["男性", "年轻人", "白领"],
        "tones": ["专业", "潮流"],
    },
    {
        "name": "服饰鞋包",
        "selling_points": ["换季折扣", "潮流新款", "限时优惠", "会员专享"],
        "audiences": ["女性", "年轻人", "学生"],
        "tones": ["潮流", "促销"],
    },
    {
        "name": "食品饮料",
        "selling_points": ["第二件半价", "产地直供", "健康低脂", "礼盒装"],
        "audiences": ["家庭", "女性", "年轻人"],
        "tones": ["亲切", "促销"],
    },
    {
        "name": "本地生活",
        "selling_points": ["到店优惠", "团购低价", "免费体验", "商圈热销"],
        "audiences": ["本地", "家庭", "白领"],
        "tones": ["促销", "亲切"],
    },
    {
        "name": "游戏娱乐",
        "selling_points": ["免费下载", "首充返利", "新服开测", "限时皮肤"],
        "audiences": ["年轻人", "男性", "游戏"],
        "tones": ["紧迫", "幽默"],
    },
    {
        "name": "金融信贷",
        "selling_points": ["低息快批", "额度高", "专业合规", "新人礼"],
        "audiences": ["白领", "年轻人", "男性"],
        "tones": ["专业", "权威"],
    },
    {
        "name": "教育培训",
        "selling_points": ["名师授课", "免费试听", "考纲精准", "小班教学"],
        "audiences": ["学生", "宝妈", "白领"],
        "tones": ["专业", "亲切"],
    },
]

TIER_BUDGET = {"S": 40.0, "M": 150.0, "L": 600.0}
TIER_VALUE = {"S": 35.0, "M": 55.0, "L": 80.0}


@dataclass
class AdvertiserSeed:
    id: int
    name: str
    category: str
    category_idx: int
    tier: str
    daily_budget: float
    initial_bid: float
    quality: float
    audience_tags: list[str]
    creatives: list[str]
    objective: str
    value_per_conversion: float = 5.0
    base_ctr: float = 0.05


def _creative(brand: str, cat: dict[str, Any], rng: np.random.Generator, tier: str) -> str:
    sp = rng.choice(cat["selling_points"])
    aud = rng.choice(cat["audiences"])
    tone = rng.choice(cat["tones"])
    obj = rng.choice(["转化", "拉新", "品牌", "复购", "到店"])
    price = rng.choice(["价格优惠", "折扣", "免费", "限量"]) if tier != "L" else rng.choice(["品质", "专业", "限量"])
    return (
        f"{brand}{cat['name']} {sp}，{aud}专享，{tone}调性，{obj}导向，"
        f"{price}，立即行动"
    )


def generate_advertisers(n: int = 120, seed: int = 42) -> list[AdvertiserSeed]:
    rng = np.random.default_rng(seed)
    out: list[AdvertiserSeed] = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        tier = rng.choice(["S", "S", "S", "M", "M", "L"], p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
        quality = float(np.clip(rng.normal(0.8, 0.12), 0.5, 1.0))
        brand = f"{cat['name'][:2]}{i:03d}"
        creatives = [_creative(brand, cat, rng, tier) for _ in range(2)]
        aud = list(rng.choice(cat["audiences"], size=2, replace=False))
        budget = TIER_BUDGET[tier] * float(rng.uniform(0.8, 1.2))
        bid = float(np.clip(rng.normal(0.9 if tier != "L" else 1.1, 0.12), 0.45, 1.4))
        out.append(
            AdvertiserSeed(
                id=i,
                name=brand,
                category=cat["name"],
                category_idx=i % len(CATEGORIES),
                tier=tier,
                daily_budget=budget,
                initial_bid=bid,
                quality=quality,
                audience_tags=list(aud),
                creatives=creatives,
                objective=str(rng.choice(["转化", "拉新", "品牌", "复购", "到店"])),
                value_per_conversion=TIER_VALUE[tier],
                base_ctr=float(np.clip(rng.normal(0.10, 0.02), 0.05, 0.18)),
            )
        )
    return out


def generate_operation_logs(
    advertisers: list[AdvertiserSeed],
    response_plan: list[tuple[BehaviorKind, float]] | None = None,
    seed: int = 5,
    advertiser_type: str | None = None,
    probability: float = 0.7,
) -> list[dict[str, Any]]:
    """Generate typed advertiser operation logs (with free-text notes) after a policy change."""
    rng = np.random.default_rng(seed)
    texts = {
        BehaviorKind.RAISE_BUDGET: ("提升日预算", "这周预算加了，多投一些"),
        BehaviorKind.RAISE_BID: ("调高出价", "竞争太激烈，把出价往上调了"),
        BehaviorKind.CHANGE_CREATIVE: ("更换素材", "旧素材点击率太低，换了新文案"),
        BehaviorKind.EXPAND_TARGETING: ("扩定向", "人群太窄，扩展了定向人群"),
        BehaviorKind.EXIT: ("暂停投放", "效果不好，先暂停投放"),
    }
    logs: list[dict[str, Any]] = []
    for ad in advertisers:
        if advertiser_type is not None and ad.tier != advertiser_type:
            continue
        if response_plan is None:
            continue
        for kind, mag in response_plan:
            if rng.random() > probability:
                continue
            head, note = texts[kind]
            logs.append(
                {
                    "advertiser_id": ad.id,
                    "event_type": kind.value,
                    "magnitude": round(mag, 3),
                    "day": int(rng.integers(1, 5)),
                    "text": f"[{head}] {note}，幅度约{mag:.0%}",
                }
            )
    return logs


OPPORTUNITY_TEXTS: dict[str, str] = {
    "lower_bid_floor": ("小广告主门槛太高", "出价门槛把中小商家挡在门外了，建议降低竞价门槛"),
    "raise_bid_floor": ("低质广告泛滥", "低价低质广告太多，建议提高竞价门槛"),
    "ad_load_cap": ("广告太多扰民", "一条信息流里广告太多，用户负反馈激增"),
    "auction_rule": ("竞价规则不透明", "一价机制导致恶意抬价，建议评估二价机制"),
    "budget_cap_multiplier": ("大客户预算受限", "大客户想加预算但受限，建议放宽预算上限"),
}


def generate_feedback_corpus(seed: int = 9, n_per_opportunity: int = 6) -> pd.DataFrame:
    """Multi-source unstructured corpus: complaints, tickets, audits, reviews, news."""
    rng = np.random.default_rng(seed)
    kinds = ["user_complaint", "advertiser_ticket", "audit_record", "experiment_review", "industry_news"]
    rows: list[dict[str, Any]] = []
    for opp_key, (title, body) in OPPORTUNITY_TEXTS.items():
        for j in range(n_per_opportunity):
            kind = kinds[j % len(kinds)]
            head = title if kind != "industry_news" else f"行业动态：{title}"
            rows.append(
                {
                    "id": f"{opp_key}_{j}",
                    "kind": kind,
                    "opportunity": opp_key,
                    "text": f"{head}：{body}（渠道：{kind}，编号 {rng.integers(1000, 9999)}）",
                }
            )
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def synthetic_behavior_dataset(n_advertisers: int = 120, seed: int = 3) -> pd.DataFrame:
    """Deterministic stand-in for the Criteo behavior baseline (offline default).

    Advertiser-level pre-experiment features with a known logistic CTR structure.
    When ``data/criteo/train.txt`` exists, ``load_criteo_sample`` can substitute it.
    """
    rng = np.random.default_rng(seed)
    n = n_advertisers
    ads = generate_advertisers(n, seed)
    spend = rng.lognormal(4.5, 0.9, n)
    conv = rng.poisson(spend / 8.0)
    ctr = np.clip(rng.normal(0.05, 0.02, n), 0.005, 0.2)
    budget = rng.choice([200.0, 800.0, 2500.0], n, p=[0.6, 0.3, 0.1])
    tier = pd.Categorical(pd.cut(budget, [0, 400, 1500, 1e9], labels=["S", "M", "L"]))
    cat = np.array([a.category_idx for a in ads])
    quality = np.clip(rng.normal(0.8, 0.12, n), 0.4, 1.0)
    x = np.column_stack(
        [
            np.log1p(spend),
            np.log1p(conv),
            ctr,
            np.log(budget),
            quality,
            (tier == "S").astype(float),
            (tier == "L").astype(float),
        ]
    )
    w = np.array([0.35, 0.5, 1.2, 0.25, 0.6, -0.4, 0.3])
    # semantic signal embedded in the creative corpus: learnable from LLM-extracted features
    cat_effect = np.array([0.8, 0.5, -0.3, 0.2, 1.2, 1.5, -1.2, -0.6])[cat]
    promo = np.array([1.2 if any(k in t for k in ("价格", "折扣", "优惠", "免费")) else 0.0 for t in [a.creatives[0] for a in ads]])
    premium = np.array([-0.9 if any(k in t for k in ("品质", "专业", "限量")) else 0.0 for t in [a.creatives[0] for a in ads]])
    sem_score = cat_effect + 0.8 * promo - 0.5 * premium
    logit = 0.7 * (x @ w) + sem_score - 3.4 + rng.normal(0, 0.25, n)
    y = (1.0 / (1.0 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)
    return pd.DataFrame(
        {
            "advertiser_id": np.arange(n),
            "pre_spend": spend,
            "pre_conversions": conv,
            "pre_ctr": ctr,
            "budget": budget,
            "quality": quality,
            "category": cat,
            "tier_S": (tier == "S").astype(int),
            "tier_L": (tier == "L").astype(int),
            "y_click_next": y,
        }
    )


def load_criteo_sample(path: str = "data/criteo/train.txt", n_rows: int = 50_000) -> pd.DataFrame:
    """Load a real Criteo sample if present (columns: label + 13 numeric + 26 categorical)."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Criteo data not found; place train.txt under data/criteo/ or use synthetic data.")
    df = pd.read_csv(p, sep="\t", header=None, nrows=n_rows)
    df.columns = ["label"] + [f"num_{i}" for i in range(13)] + [f"cat_{i}" for i in range(26)]
    return df
