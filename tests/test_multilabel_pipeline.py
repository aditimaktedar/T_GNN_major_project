"""Tests for the multi-label DDI dataset pipeline.

Covers Parts 1–4 of engineering readiness:
  - Part 1: Tests match real project APIs
  - Part 2: Pipeline aggregation, multi-hot target, label mapping
  - Part 3: Split overlap/determinism
  - Part 4: Age preprocessing and leakage prevention
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Real project imports ────────────────────────────────────────────────
from src.data.multilabel_pipeline import (
    build_multilabel_dataset,
    load_versioned_label_mapping,
    save_versioned_label_mapping,
)
from src.data.multilabel_validation import validate_multilabel_target
from src.data.multilabel_splits import (
    generate_pair_level_splits,
    validate_pair_splits,
    PairSplitError,
)
from src.data.multilabel_age import AgeStandardizer, prepare_age_feature
from src.data.multilabel_audit import generate_multilabel_dataset_audit


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_synthetic_source(n_pairs: int = 10, types_per_pair: int = 3,
                           type_pool: list[int] | None = None,
                           seed: int = 0) -> pd.DataFrame:
    """Build a small synthetic source dataset that mirrors the real CSV schema."""
    rng = np.random.default_rng(seed)
    if type_pool is None:
        type_pool = list(range(1, 21))  # 20 possible interaction types

    rows = []
    for i in range(n_pairs):
        da = f"DRUG_{i}_A"
        db = f"DRUG_{i}_B"
        pk = f"{da}|{db}"
        sa = f"CCO{'C' * i}"  # dummy distinct SMILES
        sb = f"CCN{'C' * i}"
        age = float(rng.integers(20, 90))
        chosen_types = rng.choice(type_pool, size=types_per_pair, replace=False)
        for t in chosen_types:
            rows.append({
                "pair_key": pk,
                "drug_a": da,
                "drug_b": db,
                "smiles_a": sa,
                "smiles_b": sb,
                "anchor_age": age,
                "type": int(t),
            })
    return pd.DataFrame(rows)


def _save_source_csv(df: pd.DataFrame, tmp: Path) -> Path:
    p = tmp / "source.csv"
    df.to_csv(p, index=False)
    return p


# =====================================================================
# PART 2: DATASET PIPELINE TESTS
# =====================================================================

class TestPipelineAggregation:
    """Verify the multi-label pipeline aggregation logic."""

    def test_aggregates_multiple_types_for_one_pair(self, tmp_path):
        """Part 2-1: Multiple interaction types for one pair → single row."""
        source = pd.DataFrame({
            "pair_key": ["A|B", "A|B", "A|B"],
            "drug_a": ["A", "A", "A"],
            "drug_b": ["B", "B", "B"],
            "smiles_a": ["CCO", "CCO", "CCO"],
            "smiles_b": ["CCN", "CCN", "CCN"],
            "anchor_age": [50.0, 50.0, 50.0],
            "type": [1, 2, 3],
        })
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")

        src_path = _save_source_csv(source, tmp_path)
        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        assert len(df) == 1, "One unique pair should produce exactly one row"

    def test_one_row_per_unique_pair(self, tmp_path):
        """Part 2-2: Exactly one row per unique drug pair."""
        source = _make_synthetic_source(n_pairs=5, types_per_pair=4, seed=42)
        type_pool = sorted(source["type"].unique())
        mapping = {int(t): i for i, t in enumerate(type_pool)}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")

        src_path = _save_source_csv(source, tmp_path)
        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        assert df["pair_key"].nunique() == len(df), "Each row must be a unique pair"
        assert len(df) == 5

    def test_produces_multi_hot_target(self, tmp_path):
        """Part 2-3: Target vector is a valid multi-hot encoding."""
        source = pd.DataFrame({
            "pair_key": ["X|Y", "X|Y", "P|Q"],
            "drug_a": ["X", "X", "P"],
            "drug_b": ["Y", "Y", "Q"],
            "smiles_a": ["C", "C", "CC"],
            "smiles_b": ["N", "N", "NN"],
            "anchor_age": [40.0, 40.0, 60.0],
            "type": [10, 20, 10],
        })
        mapping = {10: 0, 20: 1, 30: 2}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        row_xy = df.loc[df["pair_key"] == "X|Y"].iloc[0]
        target = json.loads(row_xy["target"]) if isinstance(row_xy["target"], str) else row_xy["target"]
        target_arr = np.array(target)
        # Types 10 and 20 → indices 0 and 1 set
        assert target_arr[0] == 1
        assert target_arr[1] == 1
        assert target_arr[2] == 0
        # All values binary
        assert set(np.unique(target_arr)).issubset({0, 1})

    def test_preserves_pair_label_associations(self, tmp_path):
        """Part 2-4: No active label association is silently dropped."""
        source = _make_synthetic_source(n_pairs=8, types_per_pair=3, seed=7)
        type_pool = sorted(source["type"].unique())
        mapping = {int(t): i for i, t in enumerate(type_pool)}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        # Verify via validate_multilabel_target (would raise on loss)
        valid_source = source[source["type"].astype(int).isin(mapping.keys())]
        validate_multilabel_target(valid_source, df, mapping)

    def test_no_duplicate_pairs(self, tmp_path):
        """Part 2-5: No duplicate pair keys in derived dataset."""
        source = _make_synthetic_source(n_pairs=6, types_per_pair=2, seed=3)
        type_pool = sorted(source["type"].unique())
        mapping = {int(t): i for i, t in enumerate(type_pool)}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        assert not df["pair_key"].duplicated().any()

    def test_rejects_missing_required_columns(self, tmp_path):
        """Part 2-6: Pipeline raises on missing required columns."""
        bad_source = pd.DataFrame({"col_a": [1], "col_b": [2]})
        src_path = _save_source_csv(bad_source, tmp_path)
        mapping = {1: 0}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")

        with pytest.raises((KeyError, ValueError, Exception)):
            build_multilabel_dataset(
                source_path=src_path,
                label_mapping_path=tmp_path / "mapping.json",
                output_parquet=tmp_path / "out.parquet",
                output_csv=tmp_path / "out.csv",
                audit_path=tmp_path / "audit.json",
                splits_path=tmp_path / "splits.json",
            )

    def test_uses_configured_label_mapping(self, tmp_path):
        """Part 2-7: Pipeline uses the provided label mapping, not ad-hoc derivation."""
        source = pd.DataFrame({
            "pair_key": ["A|B", "A|B"],
            "drug_a": ["A", "A"],
            "drug_b": ["B", "B"],
            "smiles_a": ["C", "C"],
            "smiles_b": ["N", "N"],
            "anchor_age": [50.0, 50.0],
            "type": [100, 200],
        })
        # Mapping has 5 labels but source only uses 2 of them
        mapping = {100: 0, 200: 1, 300: 2, 400: 3, 500: 4}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        target = json.loads(df.iloc[0]["target"])
        assert len(target) == 5, "Target dimension must equal label mapping size"

    def test_target_dimension_equals_label_space(self, tmp_path):
        """Part 2-8: Target vector dim == active label-space size."""
        n_labels = 7
        source = pd.DataFrame({
            "pair_key": ["A|B"],
            "drug_a": ["A"],
            "drug_b": ["B"],
            "smiles_a": ["C"],
            "smiles_b": ["N"],
            "anchor_age": [45.0],
            "type": [3],
        })
        mapping = {i: i for i in range(n_labels)}
        mapping[3] = 3  # ensure type 3 is in the mapping
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        target = json.loads(df.iloc[0]["target"])
        assert len(target) == n_labels

    def test_does_not_silently_lose_in_vocabulary_labels(self, tmp_path):
        """Part 2-9: No label in the active mapping is silently dropped."""
        source = pd.DataFrame({
            "pair_key": ["A|B", "A|B", "A|B"],
            "drug_a": ["A", "A", "A"],
            "drug_b": ["B", "B", "B"],
            "smiles_a": ["C", "C", "C"],
            "smiles_b": ["N", "N", "N"],
            "anchor_age": [50.0, 50.0, 50.0],
            "type": [1, 2, 3],
        })
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        target = np.array(json.loads(df.iloc[0]["target"]))
        assert target[0] == 1  # type 1
        assert target[1] == 1  # type 2
        assert target[2] == 1  # type 3
        assert target[3] == 0  # type 4 not in source

    def test_handles_out_of_vocabulary_labels(self, tmp_path):
        """Part 2-10: Labels outside mapping are excluded, not error."""
        source = pd.DataFrame({
            "pair_key": ["A|B", "A|B"],
            "drug_a": ["A", "A"],
            "drug_b": ["B", "B"],
            "smiles_a": ["C", "C"],
            "smiles_b": ["N", "N"],
            "anchor_age": [50.0, 50.0],
            "type": [1, 999],  # 999 is out-of-vocabulary
        })
        mapping = {1: 0, 2: 1, 3: 2}
        save_versioned_label_mapping(mapping, tmp_path / "mapping.json")
        src_path = _save_source_csv(source, tmp_path)

        df, _ = build_multilabel_dataset(
            source_path=src_path,
            label_mapping_path=tmp_path / "mapping.json",
            output_parquet=tmp_path / "out.parquet",
            output_csv=tmp_path / "out.csv",
            audit_path=tmp_path / "audit.json",
            splits_path=tmp_path / "splits.json",
        )
        target = np.array(json.loads(df.iloc[0]["target"]))
        assert len(target) == 3
        assert target[0] == 1   # type 1 is in mapping
        assert target[1] == 0   # type 2 not in source
        assert target[2] == 0   # type 3 not in source


# =====================================================================
# PART 3: SPLIT TESTS
# =====================================================================

class TestSplits:
    """Verify deterministic, leakage-free pair-level splitting."""

    def _make_pair_keys(self, n: int = 50) -> list[str]:
        return [f"DRUG_{i}_A|DRUG_{i}_B" for i in range(n)]

    def test_train_val_no_overlap(self):
        keys = self._make_pair_keys(50)
        split_map = generate_pair_level_splits(keys, seed=42)
        train = {k for k, v in split_map.items() if v == "train"}
        val = {k for k, v in split_map.items() if v == "val"}
        assert len(train & val) == 0

    def test_train_test_no_overlap(self):
        keys = self._make_pair_keys(50)
        split_map = generate_pair_level_splits(keys, seed=42)
        train = {k for k, v in split_map.items() if v == "train"}
        test = {k for k, v in split_map.items() if v == "test"}
        assert len(train & test) == 0

    def test_val_test_no_overlap(self):
        keys = self._make_pair_keys(50)
        split_map = generate_pair_level_splits(keys, seed=42)
        val = {k for k, v in split_map.items() if v == "val"}
        test = {k for k, v in split_map.items() if v == "test"}
        assert len(val & test) == 0

    def test_every_pair_in_exactly_one_split(self):
        keys = self._make_pair_keys(50)
        split_map = generate_pair_level_splits(keys, seed=42)
        assert set(split_map.keys()) == set(keys)
        for k in keys:
            assert split_map[k] in {"train", "val", "test"}

    def test_deterministic_with_same_seed(self):
        keys = self._make_pair_keys(50)
        split_1 = generate_pair_level_splits(keys, seed=123)
        split_2 = generate_pair_level_splits(keys, seed=123)
        assert split_1 == split_2

    def test_different_seed_different_result(self):
        keys = self._make_pair_keys(50)
        split_1 = generate_pair_level_splits(keys, seed=1)
        split_2 = generate_pair_level_splits(keys, seed=2)
        # Extremely unlikely to be identical with different seeds
        assert split_1 != split_2

    def test_validate_pair_splits_catches_duplicate(self):
        """validate_pair_splits raises on duplicate pair_key entries."""
        with pytest.raises(PairSplitError):
            validate_pair_splits(pd.DataFrame({
                "pair_key": ["A", "A", "B"],
                "split": ["train", "val", "test"],
            }))

    def test_validate_pair_splits_catches_overlap(self):
        """validate_pair_splits raises when same pair appears in two splits."""
        bad_map = {"A": "train", "B": "val"}
        # Trick: create a second entry for A in test via DataFrame
        with pytest.raises(PairSplitError):
            validate_pair_splits(pd.DataFrame({
                "pair_key": ["A", "B", "A"],
                "split": ["train", "val", "test"],
            }))

    def test_bad_ratio_sum_raises(self):
        keys = self._make_pair_keys(10)
        with pytest.raises(PairSplitError):
            generate_pair_level_splits(keys, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


# =====================================================================
# PART 4: AGE TESTS
# =====================================================================

class TestAgePreprocessing:
    """Verify age feature preprocessing and leakage prevention."""

    def test_age_read_from_anchor_age(self):
        """Age is read from `anchor_age` column."""
        df = pd.DataFrame({
            "anchor_age": [30.0, 50.0, 70.0],
            "split": ["train", "train", "test"],
        })
        scaled, std = prepare_age_feature(df, age_col="anchor_age")
        assert len(scaled) == 3

    def test_age_is_numeric(self):
        """Age values after transformation are numeric float32."""
        std = AgeStandardizer()
        std.fit([25.0, 45.0, 65.0])
        result = std.transform([30.0, 50.0])
        assert result.dtype == np.float32

    def test_missing_age_handling_explicit(self):
        """Missing ages are filled with training mean, not silently dropped."""
        std = AgeStandardizer()
        std.fit([20.0, 40.0, 60.0])  # mean = 40
        result = std.transform([np.nan, 50.0])
        # NaN should be replaced with mean (40), then scaled: (40-40)/std = 0
        assert np.isfinite(result[0])
        assert np.isclose(result[0], 0.0, atol=0.01)

    def test_age_preprocessing_deterministic(self):
        """Same input → same output."""
        std = AgeStandardizer()
        std.fit([20.0, 40.0, 60.0])
        r1 = std.transform([30.0, 50.0])
        r2 = std.transform([30.0, 50.0])
        np.testing.assert_array_equal(r1, r2)

    def test_age_scaler_fitted_on_training_only(self):
        """AgeStandardizer is fitted only on training split data."""
        df = pd.DataFrame({
            "anchor_age": [20.0, 40.0, 60.0, 100.0],
            "split": ["train", "train", "train", "test"],
        })
        scaled, std = prepare_age_feature(df, split_col="split", age_col="anchor_age")
        # Train mean = (20+40+60)/3 = 40, std ≈ 16.33
        assert np.isclose(std.mean, 40.0)
        # The test sample (100) should NOT influence the mean
        assert not np.isclose(std.mean, 55.0)  # Would be 55 if test included

    def test_age_not_fitted_on_test(self):
        """Test data is only transformed, never used for fitting."""
        train_ages = [25.0, 35.0, 45.0]
        test_ages = [200.0]  # Extreme value

        std = AgeStandardizer()
        std.fit(train_ages)
        train_mean = std.mean

        # Verify the extreme test value didn't affect fit
        _ = std.transform(test_ages)
        assert std.mean == train_mean, "Transform must not change fitted statistics"

    def test_age_standardizer_serialization(self):
        """AgeStandardizer can be serialized and deserialized."""
        std = AgeStandardizer()
        std.fit([10.0, 20.0, 30.0])
        d = std.to_dict()
        std2 = AgeStandardizer.from_dict(d)
        assert std2.mean == std.mean
        assert std2.std == std.std
        np.testing.assert_array_equal(
            std.transform([15.0]),
            std2.transform([15.0]),
        )

    def test_age_standardizer_unfitted_raises(self):
        """Calling transform on unfitted standardizer raises."""
        std = AgeStandardizer()
        with pytest.raises(RuntimeError):
            std.transform([30.0])

    def test_prepare_age_feature_missing_column_raises(self):
        """Missing age column raises KeyError."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(KeyError):
            prepare_age_feature(df, age_col="anchor_age")


# =====================================================================
# LABEL MAPPING TESTS
# =====================================================================

class TestLabelMapping:
    """Verify label mapping load/save round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        mapping = {10: 0, 20: 1, 30: 2}
        path = tmp_path / "mapping.json"
        save_versioned_label_mapping(mapping, path)
        loaded = load_versioned_label_mapping(path)
        assert loaded == mapping

    def test_mapping_has_correct_n_classes(self, tmp_path):
        mapping = {i: i for i in range(15)}
        path = tmp_path / "mapping.json"
        save_versioned_label_mapping(mapping, path)
        raw = json.loads(path.read_text())
        assert raw["n_classes"] == 15


# =====================================================================
# AUDIT TESTS
# =====================================================================

class TestAudit:
    """Verify audit report generation."""

    def test_audit_report_structure(self):
        source = pd.DataFrame({
            "pair_key": ["A|B", "A|B", "C|D"],
            "type": [1, 2, 1],
        })
        derived = pd.DataFrame({
            "pair_key": ["A|B", "C|D"],
            "target": [json.dumps([1, 1, 0]), json.dumps([1, 0, 0])],
        })
        mapping = {1: 0, 2: 1, 3: 2}
        report = generate_multilabel_dataset_audit(source, derived, mapping)
        assert "source_rows" in report
        assert "unique_drug_pairs" in report
        assert "labels_per_pair" in report
        assert report["source_rows"] == 3
        assert report["unique_drug_pairs"] == 2
