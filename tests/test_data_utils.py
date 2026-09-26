"""Tests for Member B path config, inventory, and inspection utilities.

These tests use tiny temporary files only. They do not use MIMIC, TWOSIDES,
OFFSIDES, DrugBank, or any credentials.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from src.data.config import (
    DRUGBANK_DIR,
    INTERIM_DATA_DIR,
    MIMIC_DIR,
    OFFSIDES_DIR,
    PHARMGKB_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TWOSIDES_DIR,
)
from src.data.inspect_data import inspect_file
from src.data.inventory import build_inventory


SRC_DATA_DIR = PROJECT_ROOT / "src" / "data"


def test_project_paths_resolve_relative_to_repository() -> None:
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "data").is_dir()
    assert RAW_DATA_DIR == PROJECT_ROOT / "data" / "raw"
    assert INTERIM_DATA_DIR == PROJECT_ROOT / "data" / "interim"
    assert PROCESSED_DATA_DIR == PROJECT_ROOT / "data" / "processed"
    assert RESULTS_DIR == PROJECT_ROOT / "results"
    assert MIMIC_DIR == RAW_DATA_DIR / "mimic"
    assert TWOSIDES_DIR == RAW_DATA_DIR / "twosides"
    assert OFFSIDES_DIR == RAW_DATA_DIR / "offsides"
    assert PHARMGKB_DIR == RAW_DATA_DIR / "pharmgkb"
    assert DRUGBANK_DIR == RAW_DATA_DIR / "drugbank"


def test_no_hardcoded_machine_specific_paths_in_data_package() -> None:
    forbidden = ("/Users/", "/home/", "C:\\", "C:/")
    for path in SRC_DATA_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains a machine-specific path token: {token}"


def test_inventory_detects_temporary_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("demo_id,demo_value\n1,apple\n2,banana\n", encoding="utf-8")

    records = build_inventory(directories=[tmp_path], project_root=tmp_path)

    assert len(records) == 1
    assert records[0]["empty"] is False
    assert records[0]["file_type"] == "csv"
    assert records[0]["path"] == "tiny.csv"
    assert records[0]["size_bytes"] > 0
    assert "demo_id" in (records[0]["header_preview"] or "")


def test_inspection_reads_temporary_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "demo_id,demo_value\n1,apple\n2,banana\n3,cherry\n",
        encoding="utf-8",
    )

    report = inspect_file(csv_path)

    assert report["n_rows"] == 3
    assert report["n_columns"] == 2
    assert report["columns"] == ["demo_id", "demo_value"]
    assert report["empty"] is False
    assert len(report["sample"]) == 3


def test_inspection_identifies_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "cols.csv"
    csv_path.write_text("alpha,beta,gamma\n1,2,3\n", encoding="utf-8")

    report = inspect_file(csv_path)

    assert report["columns"] == ["alpha", "beta", "gamma"]
    assert report["n_columns"] == 3


def test_inspection_reports_missing_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text("demo_id,demo_value\n1,apple\n2,\n,pear\n", encoding="utf-8")

    report = inspect_file(csv_path)

    assert report["missing_value_counts"]["demo_id"] == 1
    assert report["missing_value_counts"]["demo_value"] == 1
    assert report["identifier_unique_counts"]["demo_id"]["count"] == 2


def test_inspection_handles_compressed_csv_and_tsv(tmp_path: Path) -> None:
    csv_gz = tmp_path / "tiny.csv.gz"
    with gzip.open(csv_gz, "wt", encoding="utf-8") as handle:
        handle.write("demo_id,demo_value\n10,red\n20,blue\n")

    tsv_gz = tmp_path / "tiny.tsv.gz"
    with gzip.open(tsv_gz, "wt", encoding="utf-8") as handle:
        handle.write("demo_id\tdemo_value\n30\tgreen\n")

    csv_report = inspect_file(csv_gz)
    tsv_report = inspect_file(tsv_gz)

    assert csv_report["file_type"] == "csv.gz"
    assert csv_report["n_rows"] == 2
    assert csv_report["columns"] == ["demo_id", "demo_value"]
    assert tsv_report["file_type"] == "tsv.gz"
    assert tsv_report["n_rows"] == 1
    assert tsv_report["columns"] == ["demo_id", "demo_value"]


def test_inspection_handles_parquet(tmp_path: Path) -> None:
    parquet_path = tmp_path / "tiny.parquet"
    frame = pd.DataFrame({"demo_id": [1, 2], "demo_value": ["x", None]})
    frame.to_parquet(parquet_path, index=False)

    report = inspect_file(parquet_path)

    assert report["file_type"] == "parquet"
    assert report["n_rows"] == 2
    assert report["columns"] == ["demo_id", "demo_value"]
    assert "demo_id" in report["identifier_unique_counts"]
