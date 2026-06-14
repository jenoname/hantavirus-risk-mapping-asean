from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED


def load_dataset() -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / "dataset_ml.parquet"
    csv_path = PROCESSED_DIR / "dataset_ml.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Run scripts/04_preprocess.py first.")


def assign_asean_region(df: pd.DataFrame) -> pd.Series:
    """Assign seven project-specific geographic groups from grid centroids.

    These deterministic longitude/latitude partitions support region-aware
    splitting and LORO validation. They are broad analytical groups, not
    official administrative or biogeographic boundaries.
    """
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    region = np.full(len(df), "maritime_east", dtype=object)
    region[(lon < 103.0) & (lat >= 9.0)] = "mainland_west"
    region[(lon >= 103.0) & (lon < 110.0) & (lat >= 8.0)] = "mainland_mekong"
    region[(lon >= 110.0) & (lat >= 8.0)] = "philippines_north"
    region[(lon < 105.0) & (lat < 9.0)] = "sumatra_malay"
    region[(lon >= 105.0) & (lon < 116.0) & (lat < 2.5)] = "java_borneo"
    region[(lon >= 116.0) & (lat < 8.0)] = "sulawesi_maluku_papua"
    return pd.Series(region, index=df.index, name="asean_region")


def ensure_region_column(df: pd.DataFrame) -> pd.DataFrame:
    if "asean_region" not in df.columns:
        df = df.copy()
        df["asean_region"] = assign_asean_region(df)
    return df


def get_split_indices(df: pd.DataFrame, refresh: bool = False) -> dict[str, list[int]]:
    """Create one deterministic 70/15/15 split and reuse it across all stages."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    split_path = MODEL_DIR / "split_indices.json"
    if split_path.exists() and not refresh:
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        expected = set(range(len(df)))
        observed = {
            int(idx)
            for split_name in ("train", "validation", "test")
            for idx in splits.get(split_name, [])
        }
        split_total = sum(len(splits.get(name, [])) for name in ("train", "validation", "test"))
        if observed == expected and split_total == len(df):
            return splits
        print("Stored split_indices.json is incompatible with the current grid; regenerating it.")

    y = df["risk_label"].astype(int)
    strat = df["risk_label"].astype(str) + "_" + ensure_region_column(df)["asean_region"].astype(str)
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 3, y.astype(str))
    if (strat.value_counts() < 2).any():
        strat = y.astype(str)

    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=strat,
    )
    temp_strat = strat.iloc[temp_idx]
    temp_counts = temp_strat.value_counts()
    temp_strat = temp_strat.where(temp_strat.map(temp_counts) >= 2, y.iloc[temp_idx].astype(str))
    if (temp_strat.value_counts() < 2).any():
        temp_strat = y.iloc[temp_idx].astype(str)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_strat,
    )
    splits = {
        "train": sorted(map(int, train_idx)),
        "validation": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
    }
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    splits = get_split_indices(df)
    return (
        df.iloc[splits["train"]].copy(),
        df.iloc[splits["validation"]].copy(),
        df.iloc[splits["test"]].copy(),
        splits,
    )
