"""Analysis-only utilities for alternative DDI target formulations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import FINAL_MIMIC_TWOSIDES_ML_PATH, METRICS_DIR
from src.data.final_ml_dataset import load_final_ml_dataset


def analyze_target_formulations(
    frame: pd.DataFrame | None = None,
    threshold: int = 20,
) -> dict[str, Any]:
    """Analyze frequency-filtered multiclass and pair-level multi-label formulations."""
    data = frame if frame is not None else load_final_ml_dataset()
    train = data.loc[data["split"] == "train"]
    val = data.loc[data["split"] == "val"]
    test = data.loc[data["split"] == "test"]

    train_type_counts = train.groupby("type").size()
    frequent_types = {int(t) for t, c in train_type_counts.items() if int(c) >= threshold}
    rare_types = {int(t) for t, c in train_type_counts.items() if int(c) < threshold}
    missing_from_train = {int(t) for t in data["type"].unique()} - {int(t) for t in train_type_counts.index}

    def split_stats(subset: pd.DataFrame, type_set: set[int] | None = None) -> dict[str, int]:
        sub = subset if type_set is None else subset.loc[subset["type"].isin(type_set)]
        dedup = sub.drop_duplicates(["pair_key", "type"])
        return {
            "n_rows": int(len(sub)),
            "n_types": int(sub["type"].nunique()) if len(sub) else 0,
            "n_pair_types_dedup": int(len(dedup)),
            "n_pairs": int(sub["pair_key"].nunique()) if len(sub) else 0,
        }

    val_not_frequent = {int(t) for t in val["type"].unique()} - frequent_types
    test_not_frequent = {int(t) for t in test["type"].unique()} - frequent_types

    pair_type_splits = data.groupby(["pair_key", "type"])["split"].nunique()

    option1 = {
        "threshold": threshold,
        "n_frequent_types": len(frequent_types),
        "n_rare_types_in_train": len(rare_types),
        "n_types_never_in_train": len(missing_from_train),
        "types_never_in_train": sorted(missing_from_train),
        "train": {
            "frequent_only": split_stats(train, frequent_types),
            "rare_only": split_stats(train, rare_types),
            "all": split_stats(train),
        },
        "val": {
            "frequent_only": split_stats(val, frequent_types),
            "rare_only": split_stats(val, rare_types),
            "all": split_stats(val),
            "n_types_not_frequent_in_train": len(val_not_frequent),
            "rows_with_types_not_frequent_in_train": int(val["type"].isin(val_not_frequent).sum()),
        },
        "test": {
            "frequent_only": split_stats(test, frequent_types),
            "rare_only": split_stats(test, rare_types),
            "all": split_stats(test),
            "n_types_not_frequent_in_train": len(test_not_frequent),
            "rows_with_types_not_frequent_in_train": int(test["type"].isin(test_not_frequent).sum()),
        },
        "row_retention_pct": {
            "train": float(100 * train["type"].isin(frequent_types).mean()),
            "val": float(100 * val["type"].isin(frequent_types).mean()),
            "test": float(100 * test["type"].isin(frequent_types).mean()),
        },
        "other_class_projection": {
            "n_classes_with_other": len(frequent_types) + 1,
            "n_other_train_rows": int(train["type"].isin(rare_types).sum()),
            "n_other_val_rows": int(val["type"].isin(val_not_frequent).sum()),
            "n_other_test_rows": int(test["type"].isin(test_not_frequent).sum()),
        },
        "pair_type_split_leakage": int((pair_type_splits > 1).sum()),
    }

    labels_per_pair = data.groupby("pair_key")["type"].apply(lambda s: len(set(s.astype(int))))
    pair_split = data.groupby("pair_key")["split"].first()
    type_pair_counts = data.drop_duplicates(["pair_key", "type"]).groupby("type").size()
    train_type_pair_counts = train.drop_duplicates(["pair_key", "type"]).groupby("type").size()
    n_pairs = int(data["pair_key"].nunique())
    n_types = int(data["type"].nunique())
    n_positive = int(len(data.drop_duplicates(["pair_key", "type"])))

    option2: dict[str, Any] = {
        "n_unique_pairs": n_pairs,
        "labels_per_pair": {
            "min": int(labels_per_pair.min()),
            "max": int(labels_per_pair.max()),
            "median": float(labels_per_pair.median()),
            "mean": float(labels_per_pair.mean()),
            "quantiles": {
                "25%": float(labels_per_pair.quantile(0.25)),
                "75%": float(labels_per_pair.quantile(0.75)),
                "90%": float(labels_per_pair.quantile(0.90)),
            },
        },
        "distribution_labels_per_pair": {
            str(k): int(v) for k, v in sorted(Counter(labels_per_pair).items())
        },
        "splits": {},
        "type_support_pairs": {
            "types_with_ge_20_pairs": int((type_pair_counts >= 20).sum()),
            "types_with_ge_10_pairs": int((type_pair_counts >= 10).sum()),
            "types_with_1_pair": int((type_pair_counts == 1).sum()),
            "median_pairs_per_type": float(type_pair_counts.median()),
        },
        "train_type_pair_support": {
            "types_with_ge_20_train_pairs": int((train_type_pair_counts >= 20).sum()),
            "types_with_ge_10_train_pairs": int((train_type_pair_counts >= 10).sum()),
            "types_with_1_train_pair": int((train_type_pair_counts == 1).sum()),
        },
        "label_matrix": {
            "n_pairs": n_pairs,
            "n_types": n_types,
            "n_positive_labels": n_positive,
            "density": float(n_positive / (n_pairs * n_types)),
            "avg_labels_per_pair": float(n_positive / n_pairs),
        },
        "pairs_with_multiple_types": int((labels_per_pair > 1).sum()),
        "unseen_types": {
            "val_types_not_in_train": sorted(int(t) for t in set(val["type"].unique()) - set(train["type"].unique())),
            "test_types_not_in_train": sorted(int(t) for t in set(test["type"].unique()) - set(train["type"].unique())),
        },
    }
    for split in ("train", "val", "test"):
        pairs = pair_split[pair_split == split].index
        lpp = labels_per_pair.loc[pairs]
        option2["splits"][split] = {
            "n_pairs": int(len(pairs)),
            "labels_per_pair_min": int(lpp.min()),
            "labels_per_pair_max": int(lpp.max()),
            "labels_per_pair_median": float(lpp.median()),
            "labels_per_pair_mean": float(lpp.mean()),
            "total_positive_labels": int(lpp.sum()),
        }

    return {
        "dataset_path": str(FINAL_MIMIC_TWOSIDES_ML_PATH),
        "dataset_rows": int(len(data)),
        "option1_frequency_filtered_multiclass": option1,
        "option2_pair_level_multilabel": option2,
        "formulation_comparison": {
            "current_955_class": {
                "n_classes": 955,
                "train_rows": int(len(train)),
                "test_pair_types_dedup": split_stats(test)["n_pair_types_dedup"],
            },
            "option1_frequent_filtered_exclude_rare": {
                "n_classes": len(frequent_types),
                "train_rows": option1["train"]["frequent_only"]["n_rows"],
                "test_rows": option1["test"]["frequent_only"]["n_rows"],
                "test_pair_types_dedup": option1["test"]["frequent_only"]["n_pair_types_dedup"],
            },
            "option1_with_other_class": {
                "n_classes": len(frequent_types) + 1,
                "other_train_rows": option1["other_class_projection"]["n_other_train_rows"],
            },
            "option2_pair_multilabel": {
                "n_samples": n_pairs,
                "train_pairs": option2["splits"]["train"]["n_pairs"],
                "val_pairs": option2["splits"]["val"]["n_pairs"],
                "test_pairs": option2["splits"]["test"]["n_pairs"],
                "label_space": n_types,
                "avg_positive_labels_per_pair": option2["label_matrix"]["avg_labels_per_pair"],
            },
        },
    }


def write_target_formulation_analysis(
    output_path: Path | None = None,
    threshold: int = 20,
) -> dict[str, Any]:
    report = analyze_target_formulations(threshold=threshold)
    path = output_path or METRICS_DIR / "target_formulation_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
