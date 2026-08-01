"""LLM abstraction with OpenAI backend, deterministic offline fallback, caching and cost tracking.

The deterministic backend makes the whole pipeline reproducible without an API key.
When ``OPENAI_API_KEY`` is set and backend is ``auto``, real LLM calls are used and
cached on disk (keyed by prompt hash), so re-runs are offline and cheap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import LLMBackend, LLMConfig


class LLMError(RuntimeError):
    pass


class CostExceeded(LLMError):
    pass


class LLMClient:
    """Minimal interface used by all SemAdExp modules."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def extract_structured(self, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError

    def generate_text(self, system: str, user: str) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _prompt_hash(system: str, user: str, model: str) -> str:
    return hashlib.sha256((system + "\x1e" + user + "\x1e" + model).encode("utf-8")).hexdigest()[:24]


class DeterministicLLM(LLMClient):
    """Rule-based offline backend.

    Mirrors the JSON contracts of the OpenAI backend so every layer works end-to-end
    without network access. Structured tags are extracted from template markers used
    by the corpus generator; embeddings are TF-IDF vectors.
    """

    KEYWORDS = {
        "selling_point": ["价格", "折扣", "优惠", "性价比", "品质", "专业", "限量", "新品", "免费", "高效"],
        "target_audience": ["学生", "白领", "宝妈", "男性", "女性", "年轻人", "家庭", "运动", "游戏", "旅行", "健康", "本地"],
        "tone": ["促销", "专业", "情感", "幽默", "紧迫", "权威", "亲切", "潮流"],
        "objective": ["转化", "拉新", "品牌", "复购", "下载", "到店", "咨询", "注册"],
    }
    CATEGORY_MARKERS = [
        "美妆", "数码", "服饰", "食品", "本地", "游戏", "金融", "教育",
    ]

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self._tfidf = None
        self._vocab: list[str] = []

    def _tags_from_text(self, text: str) -> dict[str, list[str]]:
        tags: dict[str, list[str]] = {}
        for key, words in self.KEYWORDS.items():
            tags[key] = [w for w in words if w in text]
        cat = next((c for c in self.CATEGORY_MARKERS if c in text), None)
        tags["category"] = [cat] if cat else []
        tags["price_tier"] = ["low"] if any(w in text for w in ["价格", "折扣", "优惠", "免费"]) else (
            ["high"] if any(w in text for w in ["品质", "专业", "限量"]) else ["mid"]
        )
        return tags

    def embed(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=512)
            self._tfidf.fit(texts)
        return self._tfidf.transform(texts).toarray().astype(np.float32)

    def extract_structured(self, system: str, user: str) -> dict[str, Any]:
        text = user
        tags = self._tags_from_text(text)
        return {
            "selling_points": tags["selling_point"],
            "target_audiences": tags["target_audience"],
            "categories": tags["category"],
            "tone": tags["tone"],
            "price_tier": tags["price_tier"][0] if tags["price_tier"] else "mid",
            "objective": tags["objective"],
            "summary": f"广告文案侧重{','.join(tags['selling_point'][:3]) or '综合卖点'}，面向{','.join(tags['target_audience'][:3]) or '广泛人群'}",
        }

    def generate_text(self, system: str, user: str) -> str:
        # Generic template response used for equilibrium / attribution narratives.
        if "假设" in system or "假设" in user:
            m = re.search(r"(中小|大型|品牌|长尾)[^，。]*?(提升|降低|调整|退出)", user)
            return f"根据历史规律，{m.group(0)}的假设成立概率较高。" if m else "根据行业常识，该类型广告主会进行有限幅度的策略调整。"
        if "归因" in system or "收益" in user:
            return "高收益广告主集中在强转化导向、促销类素材的广告主群体，低收益群体以品牌导向、高价类素材为主。"
        return "基于当前语料，未发现显著模式。"


class OpenAILLM(LLMClient):
    """Real OpenAI backend with disk caching, JSON schema prompting and cost budget."""

    INPUT_RATE = 0.15 / 1_000_000   # gpt-4o-mini input tokens
    OUTPUT_RATE = 0.60 / 1_000_000  # gpt-4o-mini output tokens
    EMBED_RATE = 0.02 / 1_000_000

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai package not installed; install with `pip install openai`") from exc
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._text_cache: dict[str, str] = {}
        self._embed_cache: dict[str, list[float]] = {}
        self._load_caches()

    def _load_caches(self) -> None:
        tp, ep = self.cache_dir / "text_cache.json", self.cache_dir / "embed_cache.json"
        if tp.exists():
            self._text_cache = json.loads(tp.read_text())
        if ep.exists():
            self._embed_cache = json.loads(ep.read_text())

    def _save_caches(self) -> None:
        (self.cache_dir / "text_cache.json").write_text(json.dumps(self._text_cache))
        (self.cache_dir / "embed_cache.json").write_text(json.dumps(self._embed_cache))

    def _log_cost(self, model: str, in_tokens: int, out_tokens: int, kind: str) -> None:
        cost = in_tokens * self.INPUT_RATE + out_tokens * self.OUTPUT_RATE
        if kind == "embed":
            cost = in_tokens * self.EMBED_RATE
        log = Path(self.config.cost_log)
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        with log.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["ts", "model", "kind", "in_tokens", "out_tokens", "cost_usd"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model, kind, in_tokens, out_tokens, round(cost, 6)])
        self._assert_budget()

    def _assert_budget(self) -> None:
        log = Path(self.config.cost_log)
        if not log.exists():
            return
        total = 0.0
        with log.open() as fh:
            for row in csv.DictReader(fh):
                total += float(row["cost_usd"])
        if total > self.config.max_cost_usd:
            raise CostExceeded(
                f"LLM cost ${total:.2f} exceeds configured budget ${self.config.max_cost_usd:.2f}; "
                "raise max_cost_usd or switch to deterministic backend."
            )

    @staticmethod
    def _token_est(text: str) -> int:
        return max(1, len(text) // 4)

    def extract_structured(self, system: str, user: str) -> dict[str, Any]:
        key = _prompt_hash(system, user, self.config.model + ":json")
        if key in self._text_cache:
            return json.loads(self._text_cache[key])
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system + " Respond with a single JSON object."},
                {"role": "user", "content": user},
            ],
        )
        out = resp.choices[0].message.content
        self._text_cache[key] = out
        self._save_caches()
        self._log_cost(
            self.config.model,
            self._token_est(system + user),
            self._token_est(out),
            "chat",
        )
        return json.loads(out)

    def generate_text(self, system: str, user: str) -> str:
        key = _prompt_hash(system, user, self.config.model)
        if key in self._text_cache:
            return self._text_cache[key]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        out = resp.choices[0].message.content or ""
        self._text_cache[key] = out
        self._save_caches()
        self._log_cost(
            self.config.model, self._token_est(system + user), self._token_est(out), "chat"
        )
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        fresh: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            key = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if key in self._embed_cache:
                vectors.append(self._embed_cache[key])
            else:
                vectors.append([])
                fresh.append((i, t))
        if fresh:
            resp = self._client.embeddings.create(
                model=self.config.embedding_model, input=[t for _, t in fresh]
            )
            for (i, t), item in zip(fresh, resp.data):
                key = hashlib.sha256(t.encode("utf-8")).hexdigest()
                self._embed_cache[key] = item.embedding
                vectors[i] = item.embedding
            self._save_caches()
            self._log_cost(self.config.embedding_model, sum(len(t) for _, t in fresh) // 4, 0, "embed")
        return np.asarray(vectors, dtype=np.float32)


def get_llm(config: LLMConfig | None = None) -> LLMClient:
    cfg = config or LLMConfig()
    backend = cfg.backend
    if backend == LLMBackend.AUTO:
        backend = LLMBackend.OPENAI if os.environ.get("OPENAI_API_KEY") else LLMBackend.DETERMINISTIC
    if backend == LLMBackend.OPENAI:
        try:
            return OpenAILLM(cfg)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            import warnings

            warnings.warn(f"OpenAI backend unavailable ({exc}); falling back to deterministic backend.")
            return DeterministicLLM(cfg)
    return DeterministicLLM(cfg)
