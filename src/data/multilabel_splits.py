"""Pair-level dataset splitting module for multi-label DDI dataset.

Assigns unique drug pairs to train/validation/test splits deterministically,
guaranteeing zero leakage between splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.config import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    MULTILABEL_PAIR_SPLITS_PATH,
)


class PairSplitError(Exception):
    """Raised when pair split validation fails."""


def validate_pair_splits(splits: pd.DataFrame | dict[str, str]) -> None:
    """Verify that pair keys belong to exactly one split and overlap is zero."""
    if isinstance(splits, dict):
        splits_df = pd.DataFrame(
            [{"pair_key": k, "split": v} for k, v in splits.items()]
        )
    else:
        splits_df = splits.copy()

    if not {"pair_key", "split"}.issubset(splits_df.columns):
        raise PairSplitError("Pair split table must contain `pair_key` and `split` columns.")

    if splits_df["pair_key"].duplicated().any():
        raise PairSplitError("Pair leakage: `pair_key` appears multiple times in split mapping.")

    by_split = {name: set(group["pair_key"]) for name, group in splits_df.groupby("split")}
    train_keys = by_split.get("train", set())
    val_keys = by_split.get("val", set())
    test_keys = by_split.get("test", set())

    overlap_tv = train_keys & val_keys
    overlap_tt = train_keys & test_keys
    overlap_vt = val_keys & test_keys

    if overlap_tv or overlap_tt or overlap_vt:
        raise PairSplitError(
            f"Pair split overlap detected! train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )


def generate_pair_level_splits(
    pair_keys: list[str] | pd.Series | set[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    output_path: Path | None = None,
) -> dict[str, str]:
    """Assign each unique drug pair to train/val/test deterministically.

    Returns
    -------
    dict[str, str]
        Mapping of pair_key -> split_name ('train', 'val', or 'test').
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise PairSplitError(f"Split ratios must sum to 1.0. Got {total}")

    unique_keys = sorted({str(k) for k in pair_keys})
    keys_arr = np.array(unique_keys)
    rng = np.random.default_rng(seed)
    shuffled = keys_arr.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = set(shuffled[:n_train])
    val_keys = set(shuffled[n_train : n_train + n_val])
    test_keys = set(shuffled[n_train + n_val :])

    split_map: dict[str, str] = {}
    for k in train_keys:
        split_map[k] = "train"
    for k in val_keys:
        split_map[k] = "val"
    for k in test_keys:
        split_map[k] = "test"

    validate_pair_splits(split_map)

    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "n_pairs": len(unique_keys),
            "counts": {"train": len(train_keys), "val": len(val_keys), "test": len(test_keys)},
            "splits": split_map,
        }
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return split_map


def load_pair_splits(path: Path | None = None) -> dict[str, str]:
    target_path = Path(path) if path is not None else MULTILABEL_PAIR_SPLITS_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Pair splits file not found at {target_path}")
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    validate_pair_splits(splits)
    return splits
