"""Export the committed sample data package under data/sample/.

Usage: python scripts/export_sample_data.py [--out data/sample] [--n 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from semadexp.data.corpora import (
    generate_advertisers,
    generate_feedback_corpus,
    generate_operation_logs,
    synthetic_behavior_dataset,
)
from semadexp.config import BehaviorKind


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ads = generate_advertisers(args.n, seed=42)
    advertisers = pd.DataFrame(
        [
            {
                "advertiser_id": a.id,
                "name": a.name,
                "category": a.category,
                "category_idx": a.category_idx,
                "tier": a.tier,
                "daily_budget": a.daily_budget,
                "initial_bid": a.initial_bid,
                "quality": a.quality,
                "audience_tags": "|".join(a.audience_tags),
                "creatives": "|".join(a.creatives),
                "objective": a.objective,
                "value_per_conversion": a.value_per_conversion,
                "base_ctr": a.base_ctr,
            }
            for a in ads
        ]
    )
    advertisers.to_csv(out / "advertisers_sample.csv", index=False)

    creative_rows = [
        {"advertiser_id": a.id, "creative_index": i, "text": text}
        for a in ads
        for i, text in enumerate(a.creatives)
    ]
    pd.DataFrame(creative_rows).to_csv(out / "creative_corpus_sample.csv", index=False)

    plan = [
        (BehaviorKind.RAISE_BUDGET, 0.2),
        (BehaviorKind.RAISE_BID, 0.1),
        (BehaviorKind.CHANGE_CREATIVE, 0.15),
    ]
    logs = generate_operation_logs(ads, plan, seed=5, probability=0.6)
    pd.DataFrame(logs).to_csv(out / "operation_logs_sample.csv", index=False)

    feedback = generate_feedback_corpus(seed=9)
    feedback.to_csv(out / "feedback_corpus_sample.csv", index=False)

    behavior = synthetic_behavior_dataset(args.n, seed=3)
    behavior.to_csv(out / "behavior_sample.csv", index=False)

    print(f"sample data exported to {out.resolve()}:")
    for path in sorted(out.glob("*.csv")):
        print(f"  {path.name}  ({len(pd.read_csv(path))} rows)")


if __name__ == "__main__":
    main()
