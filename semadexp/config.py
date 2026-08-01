"""Pydantic configuration contracts for all SemAdExp modules."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AuctionType(str, Enum):
    FIRST_PRICE = "first_price"
    GSP = "gsp"


class PolicyKind(str, Enum):
    LOWER_BID_FLOOR = "lower_bid_floor"
    RAISE_BID_FLOOR = "raise_bid_floor"
    AD_LOAD_CAP = "ad_load_cap"
    AUCTION_RULE = "auction_rule"
    BUDGET_CAP_MULTIPLIER = "budget_cap_multiplier"


class BehaviorKind(str, Enum):
    RAISE_BUDGET = "raise_budget"
    RAISE_BID = "raise_bid"
    CHANGE_CREATIVE = "change_creative"
    EXPAND_TARGETING = "expand_targeting"
    EXIT = "exit"


class MarketConfig(BaseModel):
    """Parameters of the two-sided market simulation."""

    n_advertisers: int = 120
    n_categories: int = 8
    n_audience_tags: int = 12
    days: int = 12
    sessions_per_day: int = 800
    candidates_per_session: int = 8
    interference_strength: float = Field(default=1.0, ge=0.0, le=3.0)
    heterogeneity: float = Field(default=1.0, ge=0.0, le=3.0)
    auction_type: AuctionType = AuctionType.FIRST_PRICE
    bid_floor: float = 0.5
    ad_load_cap: int | None = None
    budget_cap_multiplier: float = 1.0
    seed: int = 42


class PolicyChange(BaseModel):
    """A platform policy change subject to experimentation."""

    kind: PolicyKind
    param: float = 1.0
    name: str | None = None

    def label(self) -> str:
        return self.name or self.kind.value


class ResponseHypothesis(BaseModel):
    """LLM-generated hypothesis about how one advertiser type responds to a policy."""

    advertiser_type: str = "*"
    behavior: BehaviorKind
    magnitude: float = 0.15
    probability: float = 1.0
    source: str = "llm"


class SimulationConfig(BaseModel):
    market: MarketConfig = Field(default_factory=MarketConfig)
    policy: PolicyChange | None = None
    responses: list[ResponseHypothesis] = Field(default_factory=list)
    experiment_days: int = 8


class BucketingPolicy(str, Enum):
    STRUCTURED_STRATIFIED = "structured_stratified"
    SEMANTIC_STRATIFIED = "semantic_stratified"
    GRAPH_CLUSTER = "graph_cluster"


class ExperimentConfig(BaseModel):
    bucketing: BucketingPolicy = BucketingPolicy.GRAPH_CLUSTER
    treatment_share: float = 0.5
    n_reps: int = 20
    interference: float = 1.0
    heterogeneity: float = 1.0
    seed: int = 7


class EquilibriumScenario(BaseModel):
    policy: PolicyChange
    response_hypotheses: list[ResponseHypothesis] = Field(default_factory=list)
    max_iters: int = 10
    tolerance: float = 0.08
    churn_threshold_roi: float = 0.35
    seed: int = 11


class LLMBackend(str, Enum):
    AUTO = "auto"
    OPENAI = "openai"
    DETERMINISTIC = "deterministic"


class LLMConfig(BaseModel):
    backend: LLMBackend = LLMBackend.AUTO
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    cache_dir: str = "data/cache"
    cost_log: str = "data/cache/llm_cost.csv"
    max_cost_usd: float = 20.0
    temperature: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)
