"""Loader for the committed sample data package (data/sample/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


def load_sample() -> dict[str, pd.DataFrame]:
    """Load all sample CSVs into a dict of DataFrames."""
    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(
            f"sample data directory not found at {SAMPLE_DIR}; run `python scripts/export_sample_data.py`"
        )
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        out[path.stem] = pd.read_csv(path)
    return out


def sample_path(name: str) -> Path:
    return SAMPLE_DIR / name

