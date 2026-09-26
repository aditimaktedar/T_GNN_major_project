"""Tests for local-only PubChem mapping candidate preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.pubchem import format_twosides_cid
from src.features.pubchem_candidates import prepare_mimic_pubchem_mapping_candidates


def test_local_candidates_exact_cache_and_unresolved(tmp_path: Path) -> None:
    exposures = tmp_path / "exposures.parquet"
    pd.DataFrame(
        {
            "drug_name_norm": ["acetaminophen", "CID000000001", "unknown drug", "acetaminophen"],
            "drug_name_raw": ["Acetaminophen", "CID000000001", "Unknown Drug", "Acetaminophen"],
        }
    ).to_parquet(exposures, index=False)

    twosides = tmp_path / "twosides_drugs.parquet"
    pd.DataFrame(
        {
            "drug_id": ["CID000000001", "CID000001983"],
            "smiles": ["C", "CC"],
        }
    ).to_parquet(twosides, index=False)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    success = cache_dir / "cid_success.jsonl"
    success.write_text(
        json.dumps(
            {
                "mimic_drug_name": "acetaminophen",
                "lookup_status": "resolved",
                "pubchem_cid": 1983,
                "twosides_drug_id": format_twosides_cid(1983),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    table, stats = prepare_mimic_pubchem_mapping_candidates(
        exposures_path=exposures,
        twosides_drugs_path=twosides,
        cache_dir=cache_dir,
        output_path=tmp_path / "candidates.parquet",
        stats_path=tmp_path / "stats.json",
    )

    assert stats["api_calls"] == 0
    assert stats["unique_mimic_drugs"] == 3
    assert stats["twosides_drugs"] == 2
    assert stats["directly_matched_drugs"] == 2
    assert stats["matched_exact_cid_string"] == 1
    assert stats["matched_local_pubchem_cache"] == 1
    assert stats["unresolved_drugs"] == 1

    by_name = table.set_index("mimic_drug_name_norm")
    assert by_name.loc["CID000000001", "mapping_method"] == "exact_cid_string"
    assert by_name.loc["acetaminophen", "mapping_method"] == "local_pubchem_cache"
    assert by_name.loc["unknown drug", "mapping_status"] == "unresolved"
