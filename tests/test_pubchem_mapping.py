"""Tests for MIMIC → PubChem CID mapping (mocked; no live API)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.pubchem import (
    format_twosides_cid,
    lookup_cid_by_name,
    map_mimic_drugs_to_pubchem,
    normalize_pubchem_query,
)


class CidListResponse:
    def __init__(self, status_code: int, cids: list[int] | None = None):
        self.status_code = status_code
        self.headers = {}
        self.cids = cids or []

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"IdentifierList": {"CID": self.cids}}


@pytest.fixture
def mimic_norm_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "mimic_norm.parquet"
    pd.DataFrame(
        {
            "drug_name_norm": ["aspirin", "ambiguous drug", "unknown drug", "aspirin"],
            "drug_name_raw": ["Aspirin", "Ambiguous Drug", "Unknown Drug", "Aspirin"],
        }
    ).to_parquet(path, index=False)
    return path


def test_normalize_pubchem_query_conservative() -> None:
    assert normalize_pubchem_query("  Furosemide  ") == "Furosemide"
    assert normalize_pubchem_query("  multi   space ") == "multi space"
    assert normalize_pubchem_query(None) is None


def test_format_twosides_cid() -> None:
    assert format_twosides_cid(2173) == "CID000002173"


def test_lookup_cid_resolved_and_ambiguous() -> None:
    def fetch(url: str, timeout: int):
        if "aspirin" in url:
            return CidListResponse(200, [2244])
        if "ambiguous" in url:
            return CidListResponse(200, [1, 2, 3])
        return CidListResponse(404, [])

    ok = lookup_cid_by_name("aspirin", fetch_fn=fetch, pause_seconds=0)
    assert ok["lookup_status"] == "resolved"
    assert ok["pubchem_cid"] == 2244
    assert ok["twosides_drug_id"] == format_twosides_cid(2244)

    amb = lookup_cid_by_name("ambiguous", fetch_fn=fetch, pause_seconds=0)
    assert amb["lookup_status"] == "ambiguous"
    assert amb["pubchem_cid"] is None

    missing = lookup_cid_by_name("missing", fetch_fn=fetch, pause_seconds=0)
    assert missing["lookup_status"] == "not_found"


def test_map_mimic_drugs_uses_cache(mimic_norm_fixture: Path, tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str, timeout: int):
        calls.append(url)
        if "aspirin" in url:
            return CidListResponse(200, [2244])
        if "ambiguous" in url:
            return CidListResponse(200, [10, 11])
        return CidListResponse(404, [])

    cache = tmp_path / "cid_cache"
    out = tmp_path / "mapping.parquet"
    _, stats1 = map_mimic_drugs_to_pubchem(
        input_path=mimic_norm_fixture,
        cache_dir=cache,
        output_path=out,
        stats_path=tmp_path / "stats1.json",
        fetch_fn=fetch,
        pause_seconds=0,
    )
    assert stats1["total_unique_mimic_drug_names"] == 3
    assert stats1["resolved_count"] == 1
    assert stats1["api_requests"] == 3

    _, stats2 = map_mimic_drugs_to_pubchem(
        input_path=mimic_norm_fixture,
        cache_dir=cache,
        output_path=out,
        stats_path=tmp_path / "stats2.json",
        fetch_fn=fetch,
        pause_seconds=0,
    )
    assert stats2["cached_lookups"] == 3
    assert stats2["api_requests"] == 0
    assert len(calls) == 3

    table = pd.read_parquet(out)
    aspirin = table.loc[table["mimic_drug_name"] == "aspirin"].iloc[0]
    assert aspirin["lookup_status"] == "resolved"
    assert aspirin["twosides_drug_id"] == format_twosides_cid(2244)
