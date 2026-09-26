"""Tests for MIMIC ↔ TWOSIDES CID-based label join."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.mimic_twosides_join import join_mimic_twosides_labels


def test_join_preserves_interaction_types(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "mimic_drug_name_norm": ["drug_a", "drug_b"],
            "twosides_drug_id": ["CID000000001", "CID000000002"],
            "mapping_status": ["matched", "matched"],
            "mapping_method": ["local_pubchem_cache", "local_pubchem_cache"],
        }
    ).to_parquet(mapping, index=False)

    interactions = tmp_path / "interactions.parquet"
    pd.DataFrame(
        {
            "drug_a": ["CID000000001", "CID000000001"],
            "drug_b": ["CID000000002", "CID000000002"],
            "pair_key": ["CID000000001||CID000000002", "CID000000001||CID000000002"],
            "interaction_type": [0, 1],
            "neg_sample_drug": ["CID000000003", "CID000000004"],
        }
    ).to_parquet(interactions, index=False)

    pairs = tmp_path / "pairs.parquet"
    pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "admission_id": [10, 20, 30],
            "drug_a": ["drug_a", "drug_a", "drug_a"],
            "drug_b": ["drug_b", "drug_x", "drug_b"],
            "pair_key": ["drug_a||drug_b", "drug_a||drug_x", "drug_b||drug_a"],
            "pair_rule": ["same_admission", "same_admission", "same_admission"],
        }
    ).to_parquet(pairs, index=False)

    labeled, stats = join_mimic_twosides_labels(
        mimic_pairs_path=pairs,
        mapping_path=mapping,
        interactions_path=interactions,
        output_path=tmp_path / "labeled.parquet",
        stats_path=tmp_path / "stats.json",
        batch_size=10,
    )

    assert stats["mimic_pairs_considered"] == 3
    assert stats["pairs_with_both_drugs_mapped"] == 2
    assert stats["pairs_successfully_joined_to_twosides"] == 2
    assert stats["pairs_with_unmapped_drug"] == 1
    assert stats["positive_interaction_records"] == 4
    assert stats["n_interaction_types"] == 2
    assert stats["negative_labels_created"] == 0

    assert len(labeled) == 4
    assert set(labeled["interaction_type"]) == {0, 1}
    assert (labeled["label"] == "positive").all()
    assert labeled["twosides_pair_key"].iloc[0] == "CID000000001||CID000000002"
