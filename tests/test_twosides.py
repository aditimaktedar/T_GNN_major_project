"""Tests for TWOSIDES loading using small temporary fixtures only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.exceptions import MissingInputError, SchemaError
from src.data.twosides import (
    load_twosides,
    load_twosides_drugs,
    load_twosides_interactions,
    validate_ddis_columns,
    validate_drug_smiles_columns,
)


def test_validate_real_twosides_column_names() -> None:
    resolved = validate_ddis_columns(["d1", "d2", "type", "Neg samples"])
    assert resolved["drug_a"] == "d1"
    assert resolved["neg_sample_drug"] == "Neg samples"
    drug_cols = validate_drug_smiles_columns(["Unnamed: 0", "drug_id", "smiles"])
    assert drug_cols["drug_id"] == "drug_id"


def test_missing_ddis_columns_raise_schema_error() -> None:
    with pytest.raises(SchemaError):
        validate_ddis_columns(["d1", "type"])


def test_twosides_loader_with_cid_identifiers(tmp_path: Path) -> None:
    tw_dir = tmp_path / "twosides"
    tw_dir.mkdir()
    pd.DataFrame(
        {
            "d1": ["CID000000001", "CID000000001", "CID000000002"],
            "d2": ["CID000000002", "CID000000002", "CID000000002"],
            "type": [0, 1, 0],
            "Neg samples": ["CID000000003", "CID000000004", "CID000000003"],
        }
    ).to_csv(tw_dir / "ddis.csv", index=False)
    pd.DataFrame(
        {
            "drug_id": ["CID000000001", "CID000000002", "CID000000003", "CID000000004"],
            "smiles": ["C", "CC", "CCC", "CCCC"],
        }
    ).to_csv(tw_dir / "drug_smiles.csv", index=False)

    interactions, stats = load_twosides(
        twosides_dir=tw_dir,
        output_path=tmp_path / "interactions.parquet",
        stats_path=tmp_path / "stats.json",
        streaming=False,
    )
    assert stats["n_interaction_records"] == 2
    assert stats["skipped_self_or_empty_pairs"] == 1
    assert stats["n_unique_pairs"] == 1
    assert stats["n_unique_drugs"] == 2
    assert "PubChem Compound ID" in stats["identifier_scheme"]
    assert (tmp_path / "interactions.parquet").exists()


def test_drug_missing_from_catalog_raises(tmp_path: Path) -> None:
    tw_dir = tmp_path / "twosides"
    tw_dir.mkdir()
    pd.DataFrame({"d1": ["CID000000001"], "d2": ["CID000000002"], "type": [0]}).to_csv(
        tw_dir / "ddis.csv", index=False
    )
    pd.DataFrame({"drug_id": ["CID000000001"], "smiles": ["C"]}).to_csv(
        tw_dir / "drug_smiles.csv", index=False
    )
    with pytest.raises(SchemaError):
        load_twosides(twosides_dir=tw_dir, streaming=False)


def test_missing_twosides_directory(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError):
        load_twosides_interactions(twosides_dir=tmp_path / "missing")
