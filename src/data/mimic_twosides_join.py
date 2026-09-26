"""Join MIMIC drug pairs to TWOSIDES interaction labels via verified local CID mappings.

Uses only directly supported mappings from mimic_pubchem_mapping_candidates.parquet.
Preserves TWOSIDES interaction_type. Does not invent labels or call external APIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import (
    DEFAULT_MIMIC_CHUNKSIZE,
    DRUG_PAIRS_PATH,
    METRICS_DIR,
    MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH,
    MIMIC_TWOSIDES_LABELED_PATH,
    TWOSIDES_INTERACTIONS_PATH,
)
from src.data.exceptions import MissingInputError, SchemaError
from src.data.io_utils import write_json, write_table

SUPPORTED_MAPPING_METHODS = ("exact_cid_string", "local_pubchem_cache")
DEFAULT_BATCH_SIZE = 200_000


def load_verified_mimic_cid_mapping(
    mapping_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load MIMIC drug_name_norm → TWOSIDES drug_id for directly supported mappings."""
    path = mapping_path or MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH
    if not path.exists():
        raise MissingInputError(
            f"Verified mapping candidates not found at {path}. "
            "Run: python -m src.data.pipeline pubchem-candidates"
        )
    frame = pd.read_parquet(path)
    required = {
        "mimic_drug_name_norm",
        "twosides_drug_id",
        "mapping_status",
        "mapping_method",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SchemaError(f"Mapping file missing columns: {sorted(missing)}")

    verified = frame[
        (frame["mapping_status"] == "matched")
        & (frame["mapping_method"].isin(SUPPORTED_MAPPING_METHODS))
        & frame["twosides_drug_id"].notna()
    ].copy()
    verified["mimic_drug_name_norm"] = verified["mimic_drug_name_norm"].astype(str).str.strip()
    verified["twosides_drug_id"] = verified["twosides_drug_id"].astype(str).str.strip()
    verified = verified[(verified["mimic_drug_name_norm"] != "") & (verified["twosides_drug_id"] != "")]
    verified = verified.drop_duplicates(subset=["mimic_drug_name_norm"], keep="first")
    lookup = dict(zip(verified["mimic_drug_name_norm"], verified["twosides_drug_id"]))
    return verified, lookup


def load_twosides_interactions_table(
    interactions_path: Path | None = None,
    mapped_cids: set[str] | None = None,
) -> pd.DataFrame:
    """Load TWOSIDES interactions, optionally restricted to mapped CID pairs."""
    path = interactions_path or TWOSIDES_INTERACTIONS_PATH
    if not path.exists():
        raise MissingInputError(
            f"TWOSIDES interactions not found at {path}. Run: python -m src.data.pipeline twosides"
        )

    pf = pq.ParquetFile(path)
    required = {"drug_a", "drug_b", "pair_key", "interaction_type"}
    if not required.issubset(set(pf.schema.names)):
        raise SchemaError(
            f"TWOSIDES interactions missing columns. Need {sorted(required)}, got {pf.schema.names}"
        )

    columns = ["drug_a", "drug_b", "pair_key", "interaction_type"]
    if "neg_sample_drug" in pf.schema.names:
        columns.append("neg_sample_drug")

    chunks: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=DEFAULT_BATCH_SIZE, columns=columns):
        chunk = batch.to_pandas()
        if mapped_cids is not None:
            chunk = chunk[chunk["drug_a"].isin(mapped_cids) & chunk["drug_b"].isin(mapped_cids)]
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def _add_twosides_pair_key(frame: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    out = frame.copy()
    a = out[col_a].astype(str)
    b = out[col_b].astype(str)
    swap = a > b
    out["twosides_drug_a"] = np.where(swap, b, a)
    out["twosides_drug_b"] = np.where(swap, a, b)
    out["twosides_pair_key"] = out["twosides_drug_a"] + "||" + out["twosides_drug_b"]
    return out


def _join_batch(
    batch: pd.DataFrame,
    mimic_to_cid: dict[str, str],
    twosides_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    counters = {
        "considered": int(len(batch)),
        "both_mapped": 0,
        "joined": 0,
        "both_mapped_no_twosides": 0,
        "unmapped_drug": 0,
        "interaction_records": 0,
    }
    if not len(batch):
        return pd.DataFrame(), counters

    if "drug_a" not in batch.columns or "drug_b" not in batch.columns:
        raise SchemaError(f"MIMIC pairs need drug_a and drug_b. Columns: {list(batch.columns)}")

    working = batch.copy()
    working["cid_a"] = working["drug_a"].astype(str).str.strip().map(mimic_to_cid)
    working["cid_b"] = working["drug_b"].astype(str).str.strip().map(mimic_to_cid)
    both = working[working["cid_a"].notna() & working["cid_b"].notna()].copy()
    counters["both_mapped"] = int(len(both))
    counters["unmapped_drug"] = counters["considered"] - counters["both_mapped"]

    if not len(both):
        return pd.DataFrame(), counters

    both = _add_twosides_pair_key(both, "cid_a", "cid_b")
    if "pair_key" not in both.columns:
        both["mimic_pair_key"] = both["drug_a"].astype(str) + "||" + both["drug_b"].astype(str)
    else:
        both["mimic_pair_key"] = both["pair_key"].astype(str)

    merged = both.merge(
        twosides_table,
        left_on="twosides_pair_key",
        right_on="pair_key",
        how="inner",
        suffixes=("", "_tw"),
    )
    if not len(merged):
        counters["both_mapped_no_twosides"] = counters["both_mapped"]
        return pd.DataFrame(), counters

    counters["interaction_records"] = int(len(merged))
    counters["joined"] = int(merged[["patient_id", "admission_id", "mimic_pair_key"]].drop_duplicates().shape[0])
    counters["both_mapped_no_twosides"] = counters["both_mapped"] - counters["joined"]

    labeled = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "admission_id": merged.get("admission_id"),
            "mimic_drug_a": merged["drug_a"],
            "mimic_drug_b": merged["drug_b"],
            "mimic_pair_key": merged["mimic_pair_key"],
            "twosides_drug_a": merged["twosides_drug_a"],
            "twosides_drug_b": merged["twosides_drug_b"],
            "twosides_pair_key": merged["twosides_pair_key"],
            "interaction_type": merged["interaction_type"].astype(int),
            "neg_sample_drug": merged["neg_sample_drug"] if "neg_sample_drug" in merged.columns else None,
            "label": "positive",
            "label_source": "twosides",
            "pair_rule": merged["pair_rule"] if "pair_rule" in merged.columns else None,
        }
    )
    return labeled, counters


def join_mimic_twosides_labels(
    mimic_pairs_path: Path | None = None,
    mapping_path: Path | None = None,
    interactions_path: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Join MIMIC pairs to TWOSIDES interaction records for verified CID mappings."""
    pairs_path = mimic_pairs_path or DRUG_PAIRS_PATH
    if not pairs_path.exists():
        raise MissingInputError(f"MIMIC drug pairs not found at {pairs_path}.")

    verified, mimic_to_cid = load_verified_mimic_cid_mapping(mapping_path)
    mapped_cids = set(verified["twosides_drug_id"].astype(str))
    twosides_table = load_twosides_interactions_table(
        interactions_path=interactions_path,
        mapped_cids=mapped_cids,
    )

    pf = pq.ParquetFile(pairs_path)
    writer: pq.ParquetWriter | None = None
    out = output_path or MIMIC_TWOSIDES_LABELED_PATH

    interaction_types: set[int] = set()
    patients: set[str] = set()
    stats = {
        "mimic_pairs_considered": 0,
        "pairs_with_both_drugs_mapped": 0,
        "pairs_successfully_joined_to_twosides": 0,
        "pairs_both_mapped_no_twosides_interaction": 0,
        "pairs_with_unmapped_drug": 0,
        "positive_interaction_records": 0,
        "verified_mimic_drugs_mapped": int(len(mimic_to_cid)),
        "twosides_interaction_rows_for_mapped_cids": int(len(twosides_table)),
        "twosides_unique_pair_keys_for_mapped_cids": int(twosides_table["pair_key"].nunique())
        if len(twosides_table)
        else 0,
        "negative_labels_created": 0,
        "note": (
            "Each output row is one MIMIC patient/admission pair joined to one TWOSIDES "
            "interaction_type. TWOSIDES interaction types are preserved. No negative labels "
            "were invented."
        ),
    }

    for batch in pf.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        joined, counters = _join_batch(frame, mimic_to_cid, twosides_table)
        stats["mimic_pairs_considered"] += counters["considered"]
        stats["pairs_with_both_drugs_mapped"] += counters["both_mapped"]
        stats["pairs_successfully_joined_to_twosides"] += counters["joined"]
        stats["pairs_both_mapped_no_twosides_interaction"] += counters["both_mapped_no_twosides"]
        stats["pairs_with_unmapped_drug"] += counters["unmapped_drug"]
        stats["positive_interaction_records"] += counters["interaction_records"]

        if len(joined):
            interaction_types.update(joined["interaction_type"].astype(int).tolist())
            patients.update(joined["patient_id"].dropna().astype(str).tolist())
            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()
    elif not out.exists():
        write_table(pd.DataFrame(), out)

    stats["unmatched_pairs"] = (
        stats["pairs_with_unmapped_drug"] + stats["pairs_both_mapped_no_twosides_interaction"]
    )
    stats["n_interaction_types"] = len(interaction_types)
    stats["unique_patients_represented"] = len(patients)
    stats["interaction_type_min"] = min(interaction_types) if interaction_types else None
    stats["interaction_type_max"] = max(interaction_types) if interaction_types else None

    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "mimic_twosides_label_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    stats["stats_path"] = str(metrics_out)

    result = pd.read_parquet(out) if out.exists() and out.stat().st_size > 0 else pd.DataFrame()
    return result, stats
