"""RxNorm normalization interface.

The official RxNorm mapping source is not chosen yet. This module never invents
RxNorm identifiers. A mapping table must be supplied as a local CSV/TSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import METRICS_DIR, MIMIC_NORMALIZED_PATH, RXNORM_MAPPING_DIR
from src.data.exceptions import MissingInputError, SchemaError
from src.data.io_utils import (
    load_table_chunked,
    normalize_text,
    read_header,
    resolve_column,
    write_json,
    write_table,
)

# Placeholder until the team chooses the official RxNorm mapping procedure.
RXNORM_MAPPING_PLACEHOLDER = {
    "status": "pending",
    "required_target": "RxNorm",
    "official_source": None,
    "expected_local_path": str(RXNORM_MAPPING_DIR),
    "note": (
        "Do not invent an RxNorm source. When the team finalizes the mapping method, "
        "place a CSV/TSV mapping table under data/raw/rxnorm/ or pass --mapping."
    ),
}

SOURCE_DRUG_ALIASES = ("source_drug", "drug", "drug_name", "drug_name_norm", "mimic_drug", "name")
RXNORM_ALIASES = ("rxnorm_id", "rxcui", "rxnorm", "rxnorm_cui")


def load_rxnorm_mapping(mapping_path: Path) -> tuple[pd.DataFrame, dict]:
    if not mapping_path.exists():
        raise MissingInputError(
            f"RxNorm mapping file not found: {mapping_path}. "
            "Normalization cannot invent RxNorm IDs. Provide a CSV/TSV with source drug "
            f"names and RxNorm identifiers, or place one under {RXNORM_MAPPING_DIR}. "
            f"Official mapping source is still pending: {RXNORM_MAPPING_PLACEHOLDER}"
        )
    columns = read_header(mapping_path)
    source_col = resolve_column(columns, SOURCE_DRUG_ALIASES)
    rxnorm_col = resolve_column(columns, RXNORM_ALIASES)
    if source_col is None or rxnorm_col is None:
        raise SchemaError(
            "The mapping file must contain a source-drug column "
            f"{SOURCE_DRUG_ALIASES} and an RxNorm column {RXNORM_ALIASES}. "
            f"Actual columns: {columns}"
        )
    raw = load_table_chunked(mapping_path)
    mapping = pd.DataFrame(
        {
            "source_drug_norm": raw[source_col].map(normalize_text).map(
                lambda value: value.lower() if isinstance(value, str) else value
            ),
            "rxnorm_id": raw[rxnorm_col].map(normalize_text),
        }
    )
    mapping = mapping.dropna(subset=["source_drug_norm", "rxnorm_id"])
    mapping = mapping[mapping["rxnorm_id"].astype(str).str.strip() != ""]
    mapping = mapping.drop_duplicates(subset=["source_drug_norm"], keep="first")
    report = {
        "mapping_path": str(mapping_path),
        "source_drug_column": source_col,
        "rxnorm_column": rxnorm_col,
        "n_mapping_rows_used": int(len(mapping)),
        "rxnorm_source_status": RXNORM_MAPPING_PLACEHOLDER,
    }
    return mapping, report


def normalize_drugs(
    prescriptions: pd.DataFrame | Path,
    mapping_path: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    drug_column: str = "drug_name_norm",
) -> tuple[pd.DataFrame, dict]:
    """Attach RxNorm IDs from a supplied mapping. Unmapped drugs stay unmapped."""
    if mapping_path is None:
        return prepare_drugs_pending_rxnorm(
            prescriptions,
            output_path=output_path,
            stats_path=stats_path,
            drug_column=drug_column,
        )
    if isinstance(prescriptions, Path):
        if not prescriptions.exists():
            raise MissingInputError(
                f"Normalize step needs cleaned prescriptions at {prescriptions}."
            )
        frame = load_table_chunked(prescriptions)
    else:
        frame = prescriptions.copy()

    if drug_column not in frame.columns:
        if "drug_name_raw" in frame.columns:
            frame["drug_name_norm"] = frame["drug_name_raw"].map(normalize_text).map(
                lambda value: value.lower() if isinstance(value, str) else value
            )
            drug_column = "drug_name_norm"
        else:
            raise SchemaError(
                f"No drug column '{drug_column}' or drug_name_raw in {list(frame.columns)}"
            )

    mapping, mapping_report = load_rxnorm_mapping(mapping_path)
    merged = frame.merge(mapping, how="left", left_on=drug_column, right_on="source_drug_norm")
    if "source_drug_norm" in merged.columns:
        merged = merged.drop(columns=["source_drug_norm"])
    merged["rxnorm_mapped"] = merged["rxnorm_id"].notna()

    unique_drugs = merged[drug_column].dropna().astype(str).unique()
    mapped_mask = merged["rxnorm_mapped"]
    mapped_drugs = set(merged.loc[mapped_mask, drug_column].astype(str))
    unmapped_drugs = sorted(set(unique_drugs) - mapped_drugs)

    stats = {
        **mapping_report,
        "n_rows": int(len(merged)),
        "unique_drugs": int(len(unique_drugs)),
        "mapped_rows": int(mapped_mask.sum()),
        "unmapped_rows": int((~mapped_mask).sum()),
        "mapped_unique_drugs": int(len(mapped_drugs)),
        "unmapped_unique_drugs": int(len(unmapped_drugs)),
        "unmapped_drugs": unmapped_drugs,
        "note": "Unmapped drugs were not assigned invented RxNorm identifiers.",
    }
    out = output_path if output_path is not None else MIMIC_NORMALIZED_PATH
    write_table(merged, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "rxnorm_normalization_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return merged, stats


def prepare_drugs_pending_rxnorm(
    prescriptions: pd.DataFrame | Path,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    drug_column: str = "drug_name_norm",
) -> tuple[pd.DataFrame, dict]:
    """Preserve cleaned MIMIC drug names while RxNorm mapping remains pending."""
    if isinstance(prescriptions, Path):
        if not prescriptions.exists():
            raise MissingInputError(
                f"Prepare step needs cleaned prescriptions at {prescriptions}. "
                "Run ingest-mimic and clean first."
            )
        frame = load_table_chunked(prescriptions)
    else:
        frame = prescriptions.copy()

    if drug_column not in frame.columns:
        if "drug_name_raw" in frame.columns:
            frame["drug_name_norm"] = frame["drug_name_raw"].map(normalize_text).map(
                lambda value: value.lower() if isinstance(value, str) else value
            )
            drug_column = "drug_name_norm"
        else:
            raise SchemaError(
                f"No drug column '{drug_column}' or drug_name_raw in {list(frame.columns)}"
            )

    prepared = frame.copy()
    prepared["rxnorm_id"] = pd.NA
    prepared["rxnorm_mapped"] = False
    prepared["rxnorm_status"] = "pending"

    unique_drugs = prepared[drug_column].dropna().astype(str).unique()
    stats = {
        "rxnorm_source_status": RXNORM_MAPPING_PLACEHOLDER,
        "n_rows": int(len(prepared)),
        "unique_drugs": int(len(unique_drugs)),
        "mapped_rows": 0,
        "unmapped_rows": int(len(prepared)),
        "mapped_unique_drugs": 0,
        "unmapped_unique_drugs": int(len(unique_drugs)),
        "note": (
            "RxNorm mapping is pending. Original MIMIC drug names/identifiers were preserved "
            "for downstream pair construction."
        ),
    }
    out = output_path if output_path is not None else MIMIC_NORMALIZED_PATH
    write_table(prepared, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "rxnorm_normalization_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return prepared, stats
