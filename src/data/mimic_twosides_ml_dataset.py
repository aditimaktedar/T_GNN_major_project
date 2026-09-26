"""Streaming ML dataset preparation for MIMIC ↔ TWOSIDES labeled pairs.

Builds a compact, deduplicated dataset without loading the full 849M-row join
into memory. Negatives are sampled only from real MIMIC drug pairs whose verified
TWOSIDES CIDs do not appear as an interacting pair in TWOSIDES.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import (
    DEFAULT_MIMIC_CHUNKSIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DRUG_PAIRS_PATH,
    METRICS_DIR,
    MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH,
    MIMIC_TWOSIDES_LABELED_PATH,
    MIMIC_TWOSIDES_ML_DATASET_PATH,
    MIMIC_TWOSIDES_ML_NEGATIVES_PATH,
    MIMIC_TWOSIDES_ML_POSITIVES_PATH,
    MIMIC_TWOSIDES_ML_SPLITS_PATH,
    TWOSIDES_PAIRS_PATH,
)
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.io_utils import write_json
from src.data.mimic_twosides_join import (
    SUPPORTED_MAPPING_METHODS,
    _add_twosides_pair_key,
    load_verified_mimic_cid_mapping,
)
from src.data.splits import validate_patient_splits

DEFAULT_NEGATIVE_RATIO = 1.0
DEFAULT_DEDUP_BUCKETS = 512
DEFAULT_BATCH_SIZE = DEFAULT_MIMIC_CHUNKSIZE

# Sanity thresholds for real-data runs (see validate_real_data_inputs).
MIN_TWOSIDES_UNIQUE_PAIRS_FOR_REAL_RUN = 60_000
MIN_LABELED_ROWS_FOR_REAL_RUN = 800_000_000
EXPECTED_VERIFIED_MAPPINGS = 489

# Documented ML column semantics (see SCHEMA.md).
ML_FIELD_DEFINITIONS = {
    "patient_id": (
        "MIMIC subject identifier copied from labeled-pairs `patient_id` "
        "(MIMIC-IV `subject_id`). Used for patient-level train/val/test assignment."
    ),
    "drug_a": (
        "Canonical MIMIC formulary name (`drug_name_norm`) with lexicographic ordering "
        "so unordered pairs are stable: drug_a <= drug_b."
    ),
    "drug_b": (
        "Second canonical MIMIC formulary name in the ordered pair (drug_a <= drug_b)."
    ),
    "interaction_type": (
        "TWOSIDES interaction code (integer 0–962) preserved for positive rows. "
        "Null for negative rows because no TWOSIDES interaction was observed."
    ),
    "label": (
        "Binary task label: `positive` when the row comes from a verified TWOSIDES "
        "interaction join; `negative` when the row is a sampled MIMIC pair with both "
        "drugs mapped to TWOSIDES CIDs but no TWOSIDES interaction for that CID pair."
    ),
    "label_source": (
        "`twosides` for positives; `mimic_unmatched_twosides_pair` for negatives."
    ),
}

POSITIVE_OUTPUT_COLUMNS = [
    "patient_id",
    "drug_a",
    "drug_b",
    "mimic_pair_key",
    "twosides_drug_a",
    "twosides_drug_b",
    "twosides_pair_key",
    "interaction_type",
    "label",
    "label_source",
    "pair_rule",
]

NEGATIVE_OUTPUT_COLUMNS = [
    "patient_id",
    "admission_id",
    "drug_a",
    "drug_b",
    "mimic_pair_key",
    "twosides_drug_a",
    "twosides_drug_b",
    "twosides_pair_key",
    "interaction_type",
    "label",
    "label_source",
    "pair_rule",
]

DATASET_OUTPUT_COLUMNS = POSITIVE_OUTPUT_COLUMNS + ["split"]


def validate_real_data_inputs(
    labeled_path: Path | None = None,
    mapping_path: Path | None = None,
    unique_pairs_path: Path | None = None,
    *,
    require_real_scale: bool = True,
) -> dict:
    """Preflight checks before expensive streaming. Read-only; never modifies sources."""
    labeled = labeled_path or MIMIC_TWOSIDES_LABELED_PATH
    mapping = mapping_path or MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH
    unique_pairs = unique_pairs_path or TWOSIDES_PAIRS_PATH

    if not labeled.exists():
        raise MissingInputError(
            f"Labeled pairs missing at {labeled}. Run: python -m src.data.pipeline labels-twosides"
        )
    labeled_pf = pq.ParquetFile(labeled)
    labeled_rows = int(labeled_pf.metadata.num_rows)

    if not mapping.exists():
        raise MissingInputError(
            f"Mapping candidates missing at {mapping}. Run: python -m src.data.pipeline pubchem-candidates"
        )
    mapping_frame = pd.read_parquet(
        mapping,
        columns=["mapping_status", "mapping_method", "twosides_drug_id"],
    )
    verified = mapping_frame[
        (mapping_frame["mapping_status"] == "matched")
        & (mapping_frame["mapping_method"].isin(SUPPORTED_MAPPING_METHODS))
        & mapping_frame["twosides_drug_id"].notna()
    ]
    n_verified = int(verified["twosides_drug_id"].astype(str).nunique())

    if not unique_pairs.exists():
        raise MissingInputError(
            f"TWOSIDES unique pairs missing at {unique_pairs}. "
            "Reload from raw (read-only): python -m src.data.pipeline twosides"
        )
    unique_pf = pq.ParquetFile(unique_pairs)
    n_unique_pairs = int(unique_pf.metadata.num_rows)

    checks = {
        "labeled_rows": labeled_rows,
        "labeled_columns": list(labeled_pf.schema.names),
        "verified_mimic_mappings": n_verified,
        "twosides_unique_pairs": n_unique_pairs,
    }

    if require_real_scale:
        if labeled_rows < MIN_LABELED_ROWS_FOR_REAL_RUN:
            raise ConfigurationError(
                f"Labeled pairs has {labeled_rows:,} rows; expected real-scale "
                f"(>={MIN_LABELED_ROWS_FOR_REAL_RUN:,}). Refusing expensive run on a truncated file."
            )
        if n_unique_pairs < MIN_TWOSIDES_UNIQUE_PAIRS_FOR_REAL_RUN:
            raise ConfigurationError(
                f"TWOSIDES unique pairs has {n_unique_pairs:,} rows; expected real-scale "
                f"(>={MIN_TWOSIDES_UNIQUE_PAIRS_FOR_REAL_RUN:,}). "
                "Interim TWOSIDES files may have been overwritten by tests. "
                "Reload from raw CSV (read-only on raw): python -m src.data.pipeline twosides"
            )
        if n_verified < EXPECTED_VERIFIED_MAPPINGS:
            raise ConfigurationError(
                f"Only {n_verified} verified mappings found; expected {EXPECTED_VERIFIED_MAPPINGS}. "
                "Do not invent mappings. Regenerate candidates from local cache only."
            )

    return checks


def check_patient_leakage(splits: pd.DataFrame) -> None:
    """Explicit patient-leakage guard used before writing ML splits/datasets."""
    validate_patient_splits(splits)


def describe_prepare_outputs() -> dict:
    """Document expected inputs/outputs for manual real-data runs."""
    return {
        "command": "python -m src.data.pipeline prepare-ml-twosides",
        "reads_only": [
            str(MIMIC_TWOSIDES_LABELED_PATH),
            str(DRUG_PAIRS_PATH),
            str(MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH),
            str(TWOSIDES_PAIRS_PATH),
        ],
        "writes": [
            str(MIMIC_TWOSIDES_ML_POSITIVES_PATH),
            str(MIMIC_TWOSIDES_ML_NEGATIVES_PATH),
            str(MIMIC_TWOSIDES_ML_SPLITS_PATH),
            str(MIMIC_TWOSIDES_ML_DATASET_PATH),
            str(METRICS_DIR / "mimic_twosides_ml_dataset_statistics.json"),
        ],
        "does_not_modify": [
            "data/raw/**",
            "data/interim/mimic_twosides_labeled_pairs.parquet",
            "data/interim/twosides_*.parquet",
            "data/interim/mimic_pubchem_mapping_candidates.parquet",
        ],
        "expected_input_scale": {
            "labeled_rows": "849,453,614",
            "labeled_size_gb": "~2.1",
            "drug_pairs_rows": "130,753,703",
            "verified_mappings": 489,
            "twosides_unique_pairs": "63,472",
            "eligible_negative_pool": "~343,058 patient-admission pairs",
        },
        "expected_output_scale": {
            "positives_after_dedup": (
                "<= 849M; dedup at (patient_id, drug_a, drug_b, interaction_type)"
            ),
            "negatives_sampled": (
                "min(round(n_positives_after_dedup * negative_ratio), eligible_pool); "
                "default ratio 1.0 caps at eligible pool (~343K)"
            ),
            "unique_patients": "~148,258",
        },
    }


def inspect_labeled_pairs_columns(labeled_path: Path | None = None) -> dict:
    """Return actual parquet schema/columns for the joined labeled dataset."""
    path = labeled_path or MIMIC_TWOSIDES_LABELED_PATH
    if not path.exists():
        raise MissingInputError(
            f"Labeled pairs not found at {path}. Run: python -m src.data.pipeline labels-twosides"
        )
    pf = pq.ParquetFile(path)
    columns = list(pf.schema.names)
    return {
        "path": str(path),
        "num_rows": int(pf.metadata.num_rows),
        "num_row_groups": int(pf.metadata.num_row_groups),
        "columns": columns,
        "field_definitions": ML_FIELD_DEFINITIONS,
    }


def canonicalize_mimic_pair(drug_a: str, drug_b: str) -> tuple[str, str, str]:
    """Return lexicographically ordered (drug_a, drug_b, mimic_pair_key)."""
    a = str(drug_a).strip()
    b = str(drug_b).strip()
    if a <= b:
        return a, b, f"{a}||{b}"
    return b, a, f"{b}||{a}"


def _bucket_index(values: pd.Series, n_buckets: int) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
    return (hashed % n_buckets).astype(np.int32)


def _positives_from_labeled_batch(batch: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "mimic_drug_a", "mimic_drug_b", "interaction_type", "label"}
    missing = required - set(batch.columns)
    if missing:
        raise SchemaError(f"Labeled batch missing columns: {sorted(missing)}")

    positive = batch[batch["label"].astype(str) == "positive"].copy()
    if not len(positive):
        return pd.DataFrame(columns=POSITIVE_OUTPUT_COLUMNS)

    drugs = positive.apply(
        lambda row: canonicalize_mimic_pair(row["mimic_drug_a"], row["mimic_drug_b"]),
        axis=1,
        result_type="expand",
    )
    positive["drug_a"] = drugs[0]
    positive["drug_b"] = drugs[1]
    positive["mimic_pair_key"] = drugs[2]

    tw = _add_twosides_pair_key(positive, "twosides_drug_a", "twosides_drug_b")
    positive["twosides_drug_a"] = tw["twosides_drug_a"]
    positive["twosides_drug_b"] = tw["twosides_drug_b"]
    positive["twosides_pair_key"] = tw["twosides_pair_key"]

    out = pd.DataFrame(
        {
            "patient_id": positive["patient_id"],
            "drug_a": positive["drug_a"],
            "drug_b": positive["drug_b"],
            "mimic_pair_key": positive["mimic_pair_key"],
            "twosides_drug_a": positive["twosides_drug_a"],
            "twosides_drug_b": positive["twosides_drug_b"],
            "twosides_pair_key": positive["twosides_pair_key"],
            "interaction_type": positive["interaction_type"].astype("Int64"),
            "label": "positive",
            "label_source": positive["label_source"] if "label_source" in positive.columns else "twosides",
            "pair_rule": positive["pair_rule"] if "pair_rule" in positive.columns else None,
        }
    )
    return out


def _dedup_key_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["patient_id"].astype(str)
        + "||"
        + frame["drug_a"].astype(str)
        + "||"
        + frame["drug_b"].astype(str)
        + "||"
        + frame["interaction_type"].astype(str)
    )


def _filter_new_keys(conn: sqlite3.Connection, keys: list[str]) -> list[bool]:
    """Return a mask indicating which keys are newly inserted (not seen before)."""
    if not keys:
        return []
    keep = [False] * len(keys)
    cursor = conn.cursor()
    for start in range(0, len(keys), 5000):
        chunk = keys[start : start + 5000]
        placeholders = ",".join("?" for _ in chunk)
        existing = {
            row[0]
            for row in cursor.execute(
                f"SELECT k FROM seen WHERE k IN ({placeholders})",
                chunk,
            )
        }
        new_keys = [k for k in chunk if k not in existing]
        if new_keys:
            cursor.executemany("INSERT INTO seen(k) VALUES (?)", [(k,) for k in new_keys])
        idx = 0
        for offset, key in enumerate(chunk):
            if key not in existing:
                keep[start + offset] = True
    conn.commit()
    return keep


def _summarize_positive_output(
    positives_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    include_interaction_type_counts: bool = False,
) -> dict[str, int | dict[int, int]]:
    """Stream summarized counts from deduped positives without loading all rows."""
    patients: set[str] = set()
    drugs: set[str] = set()
    unordered_pairs: set[str] = set()
    interaction_type_counts: dict[int, int] = {}
    rows = 0

    pf = pq.ParquetFile(positives_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        rows += int(len(frame))
        patients.update(frame["patient_id"].astype(str).tolist())
        drugs.update(frame["drug_a"].astype(str).tolist())
        drugs.update(frame["drug_b"].astype(str).tolist())
        unordered_pairs.update(frame["mimic_pair_key"].astype(str).tolist())
        if include_interaction_type_counts:
            for itype, count in frame["interaction_type"].value_counts().items():
                interaction_type_counts[int(itype)] = interaction_type_counts.get(int(itype), 0) + int(count)

    result: dict[str, int | dict[int, int]] = {
        "positive_rows_after_dedup": rows,
        "unique_patients": len(patients),
        "unique_drugs": len(drugs),
        "unique_unordered_pairs": len(unordered_pairs),
        "n_interaction_types": 0,
        "interaction_type_counts": {},
    }
    if include_interaction_type_counts:
        result["n_interaction_types"] = len(interaction_type_counts)
        result["interaction_type_counts"] = dict(sorted(interaction_type_counts.items()))
    return result


def stream_dedupe_positives(
    labeled_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_buckets: int = DEFAULT_DEDUP_BUCKETS,
    temp_dir: Path | None = None,
    resume_buckets: bool = False,
) -> tuple[Path, dict]:
    """Stream labeled positives, dedupe at patient/pair/interaction_type, write parquet."""
    del n_buckets, temp_dir, resume_buckets  # legacy args retained for CLI compatibility

    path = labeled_path or MIMIC_TWOSIDES_LABELED_PATH
    out = output_path or MIMIC_TWOSIDES_ML_POSITIVES_PATH
    inspect = inspect_labeled_pairs_columns(path)

    db_path = out.parent / f".{out.stem}_dedup.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE seen (k TEXT PRIMARY KEY)")

    pf = pq.ParquetFile(path)
    raw_rows = 0
    deduped_rows = 0
    duplicates_removed = 0
    interaction_type_counts: dict[int, int] = {}
    patients: set[str] = set()
    drugs: set[str] = set()
    writer: pq.ParquetWriter | None = None

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            raw_rows += int(len(frame))
            positives = _positives_from_labeled_batch(frame)
            if not len(positives):
                continue

            positives = positives.drop_duplicates(
                subset=["patient_id", "drug_a", "drug_b", "interaction_type"],
                keep="first",
            )
            keys = _dedup_key_series(positives).tolist()
            keep_mask = _filter_new_keys(conn, keys)
            kept = positives.loc[keep_mask]
            duplicates_removed += int(len(positives) - len(kept))
            if not len(kept):
                continue

            deduped_rows += int(len(kept))
            patients.update(kept["patient_id"].astype(str).tolist())
            drugs.update(kept["drug_a"].astype(str).tolist())
            drugs.update(kept["drug_b"].astype(str).tolist())
            for itype, count in kept["interaction_type"].value_counts().items():
                interaction_type_counts[int(itype)] = interaction_type_counts.get(int(itype), 0) + int(count)

            table = pa.Table.from_pandas(kept, preserve_index=False)
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        conn.close()
        db_path.unlink(missing_ok=True)

    if not out.exists():
        pd.DataFrame(columns=POSITIVE_OUTPUT_COLUMNS).to_parquet(out, index=False)

    stats = {
        "labeled_rows_streamed": raw_rows,
        "positive_rows_before_dedup": raw_rows,
        "positive_rows_after_dedup": deduped_rows,
        "duplicate_rows_removed": duplicates_removed,
        "unique_patients": len(patients),
        "unique_drugs": len(drugs),
        "unique_unordered_pairs": None,
        "n_interaction_types": len(interaction_type_counts),
        "interaction_type_counts": dict(sorted(interaction_type_counts.items())),
        "input_columns": inspect["columns"],
        "output_path": str(out),
        "dedup_key": ["patient_id", "drug_a", "drug_b", "interaction_type"],
        "dedup_method": "sqlite_primary_key streaming dedup (one batch in memory)",
        "resumed_from_buckets": False,
    }
    return out, stats


def _load_twosides_pair_keys(unique_pairs_path: Path | None = None) -> set[str]:
    path = unique_pairs_path or TWOSIDES_PAIRS_PATH
    if not path.exists():
        raise MissingInputError(
            f"TWOSIDES unique pairs not found at {path}. Run: python -m src.data.pipeline twosides"
        )
    frame = pd.read_parquet(path, columns=["pair_key"])
    return set(frame["pair_key"].astype(str))


def collect_negative_candidate_pairs(
    mimic_pairs_path: Path | None = None,
    mapping_path: Path | None = None,
    unique_pairs_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temp_pool_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Stream MIMIC pairs and collect real pairs eligible for negative sampling."""
    pairs_path = mimic_pairs_path or DRUG_PAIRS_PATH
    if not pairs_path.exists():
        raise MissingInputError(f"MIMIC drug pairs not found at {pairs_path}.")

    _, mimic_to_cid = load_verified_mimic_cid_mapping(mapping_path)
    twosides_keys = _load_twosides_pair_keys(unique_pairs_path)

    pf = pq.ParquetFile(pairs_path)
    pool_writer: pq.ParquetWriter | None = None
    pool_out = temp_pool_path
    in_memory_chunks: list[pd.DataFrame] = []
    seen_patient_pairs: set[str] = set()
    considered = 0
    both_mapped = 0
    eligible = 0
    matched_twosides = 0
    written = 0

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            considered += int(len(frame))
            frame["cid_a"] = frame["drug_a"].astype(str).str.strip().map(mimic_to_cid)
            frame["cid_b"] = frame["drug_b"].astype(str).str.strip().map(mimic_to_cid)
            mapped = frame[frame["cid_a"].notna() & frame["cid_b"].notna()].copy()
            both_mapped += int(len(mapped))
            if not len(mapped):
                continue

            tw = _add_twosides_pair_key(mapped, "cid_a", "cid_b")
            mapped["twosides_pair_key"] = tw["twosides_pair_key"]
            mapped["twosides_drug_a"] = tw["twosides_drug_a"]
            mapped["twosides_drug_b"] = tw["twosides_drug_b"]

            in_tw = mapped["twosides_pair_key"].isin(twosides_keys)
            matched_twosides += int(in_tw.sum())
            unmatched = mapped[~in_tw].copy()
            if not len(unmatched):
                continue

            canon = unmatched.apply(
                lambda row: canonicalize_mimic_pair(row["drug_a"], row["drug_b"]),
                axis=1,
                result_type="expand",
            )
            unmatched["drug_a"] = canon[0]
            unmatched["drug_b"] = canon[1]
            unmatched["mimic_pair_key"] = canon[2]
            eligible += int(len(unmatched))

            subset = unmatched[
                [
                    "patient_id",
                    "admission_id",
                    "drug_a",
                    "drug_b",
                    "mimic_pair_key",
                    "twosides_drug_a",
                    "twosides_drug_b",
                    "twosides_pair_key",
                    "pair_rule",
                ]
            ].copy()
            subset["_patient_pair_key"] = (
                subset["patient_id"].astype(str)
                + "||"
                + subset["mimic_pair_key"].astype(str)
            )
            subset = subset[~subset["_patient_pair_key"].isin(seen_patient_pairs)]
            if not len(subset):
                continue
            seen_patient_pairs.update(subset["_patient_pair_key"].tolist())
            subset = subset.drop(columns=["_patient_pair_key"])
            written += int(len(subset))

            if pool_out is not None:
                table = pa.Table.from_pandas(subset, preserve_index=False)
                if pool_writer is None:
                    pool_out.parent.mkdir(parents=True, exist_ok=True)
                    pool_writer = pq.ParquetWriter(pool_out, table.schema)
                pool_writer.write_table(table)
            else:
                in_memory_chunks.append(subset)
    finally:
        if pool_writer is not None:
            pool_writer.close()

    if pool_out is not None and pool_out.exists() and pool_out.stat().st_size > 0:
        pool = pd.read_parquet(pool_out)
    elif in_memory_chunks:
        pool = pd.concat(in_memory_chunks, ignore_index=True)
    else:
        pool = pd.DataFrame(columns=NEGATIVE_OUTPUT_COLUMNS)

    stats = {
        "mimic_pairs_considered": considered,
        "pairs_with_both_drugs_mapped": both_mapped,
        "pairs_with_twosides_interaction": matched_twosides,
        "eligible_negative_pairs_before_patient_pair_dedup": eligible,
        "eligible_negative_pairs_after_patient_pair_dedup": int(len(pool)),
        "verified_mapping_methods": list(SUPPORTED_MAPPING_METHODS),
        "note": (
            "Eligible negatives are real MIMIC same-admission pairs where both drugs map "
            "to verified TWOSIDES CIDs but the canonical TWOSIDES pair_key is absent "
            "from twosides_unique_pairs.parquet. No drug pairs were fabricated."
        ),
    }
    return pool, stats


def sample_negative_pairs(
    candidate_pool: pd.DataFrame,
    n_positives: int,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Deterministically sample negatives from the eligible real MIMIC pair pool."""
    if negative_ratio < 0:
        raise ConfigurationError(f"negative_ratio must be >= 0. Got {negative_ratio}.")

    target = int(round(n_positives * negative_ratio)) if n_positives > 0 else 0
    pool = candidate_pool.copy()
    if not len(pool) or target <= 0:
        empty = pd.DataFrame(columns=NEGATIVE_OUTPUT_COLUMNS)
        return empty, {
            "negative_ratio": negative_ratio,
            "target_negative_rows": target,
            "sampled_negative_rows": 0,
            "seed": seed,
            "negative_sampling_method": "deterministic_without_replacement_from_eligible_mimic_pairs",
        }

    n_sample = min(target, len(pool))
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(pool.index.to_numpy(), size=n_sample, replace=False)
    sampled = pool.loc[chosen_idx].copy()
    sampled["interaction_type"] = pd.Series([pd.NA] * len(sampled), dtype="Int64")
    sampled["label"] = "negative"
    sampled["label_source"] = "mimic_unmatched_twosides_pair"

    stats = {
        "negative_ratio": negative_ratio,
        "target_negative_rows": target,
        "eligible_negative_pool_size": int(len(pool)),
        "sampled_negative_rows": int(len(sampled)),
        "seed": seed,
        "negative_sampling_method": (
            "Sample min(round(n_positives * negative_ratio), eligible_pool) indices "
            "with numpy.random.Generator(seed).choice(..., replace=False) from real "
            "MIMIC pairs whose verified TWOSIDES CID pair is absent from TWOSIDES."
        ),
    }
    return sampled[NEGATIVE_OUTPUT_COLUMNS], stats


def build_patient_splits_table(
    patient_ids: set[str] | list[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assign each patient to exactly one split."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ConfigurationError(f"Split ratios must sum to 1. Got {total}")

    patients = np.array(sorted({str(pid) for pid in patient_ids}))
    if len(patients) == 0:
        raise ConfigurationError("No patients available for split assignment.")

    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    splits = pd.DataFrame(
        [{"patient_id": pid, "split": "train"} for pid in train_ids]
        + [{"patient_id": pid, "split": "val"} for pid in val_ids]
        + [{"patient_id": pid, "split": "test"} for pid in test_ids]
    )
    check_patient_leakage(splits)

    out = output_path or MIMIC_TWOSIDES_ML_SPLITS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out, index=False)

    stats = {
        "n_patients": int(len(patients)),
        "n_train_patients": int((splits["split"] == "train").sum()),
        "n_val_patients": int((splits["split"] == "val").sum()),
        "n_test_patients": int((splits["split"] == "test").sum()),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "output_path": str(out),
    }
    return splits, stats


def _attach_split(frame: pd.DataFrame, split_lookup: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = out["patient_id"].astype(str).map(split_lookup)
    missing = int(out["split"].isna().sum())
    if missing:
        raise ConfigurationError(f"{missing} rows have no patient split assignment.")
    return out


def _write_dataset_part(
    frame: pd.DataFrame,
    split_lookup: dict[str, str],
    writer: pq.ParquetWriter | None,
    output_path: Path,
) -> pq.ParquetWriter:
    part = _attach_split(frame, split_lookup)
    columns = [col for col in DATASET_OUTPUT_COLUMNS if col in part.columns]
    table = pa.Table.from_pandas(part[columns], preserve_index=False)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)
    return writer


def assemble_ml_dataset(
    positives_path: Path,
    negatives_path: Path,
    splits_path: Path,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Path, dict]:
    """Join deduped positives and sampled negatives with patient splits (streaming)."""
    out = output_path or MIMIC_TWOSIDES_ML_DATASET_PATH
    splits = pd.read_parquet(splits_path)
    validate_patient_splits(splits)
    split_lookup = dict(zip(splits["patient_id"].astype(str), splits["split"].astype(str)))

    writer: pq.ParquetWriter | None = None
    row_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"positive": 0, "negative": 0}

    for part_path in (positives_path, negatives_path):
        if not part_path.exists() or part_path.stat().st_size == 0:
            continue
        pf = pq.ParquetFile(part_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            writer = _write_dataset_part(frame, split_lookup, writer, out)
            attached = _attach_split(frame, split_lookup)
            for split_name, count in attached["split"].value_counts().items():
                row_counts[str(split_name)] = row_counts.get(str(split_name), 0) + int(count)
            for label_name, count in attached["label"].value_counts().items():
                label_counts[str(label_name)] = label_counts.get(str(label_name), 0) + int(count)

    if writer is not None:
        writer.close()
    elif not out.exists():
        pd.DataFrame(columns=DATASET_OUTPUT_COLUMNS).to_parquet(out, index=False)

    stats = {
        "output_path": str(out),
        "train_rows": row_counts.get("train", 0),
        "val_rows": row_counts.get("val", 0),
        "test_rows": row_counts.get("test", 0),
        "positive_rows": label_counts.get("positive", 0),
        "negative_rows": label_counts.get("negative", 0),
    }
    return out, stats


def prepare_mimic_twosides_ml_dataset(
    labeled_path: Path | None = None,
    mimic_pairs_path: Path | None = None,
    mapping_path: Path | None = None,
    unique_pairs_path: Path | None = None,
    positives_path: Path | None = None,
    negatives_path: Path | None = None,
    splits_path: Path | None = None,
    dataset_path: Path | None = None,
    stats_path: Path | None = None,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_buckets: int = DEFAULT_DEDUP_BUCKETS,
    require_real_scale: bool | None = None,
    resume_buckets: bool = False,
) -> dict:
    """End-to-end streaming ML dataset preparation."""
    using_defaults = all(
        path is None
        for path in (
            labeled_path,
            mimic_pairs_path,
            mapping_path,
            unique_pairs_path,
            positives_path,
            negatives_path,
            splits_path,
            dataset_path,
        )
    )
    if require_real_scale is None:
        require_real_scale = using_defaults

    preflight = validate_real_data_inputs(
        labeled_path=labeled_path,
        mapping_path=mapping_path,
        unique_pairs_path=unique_pairs_path,
        require_real_scale=require_real_scale,
    )
    schema_info = inspect_labeled_pairs_columns(labeled_path)

    pos_out = positives_path or MIMIC_TWOSIDES_ML_POSITIVES_PATH
    neg_out = negatives_path or MIMIC_TWOSIDES_ML_NEGATIVES_PATH
    split_out = splits_path or MIMIC_TWOSIDES_ML_SPLITS_PATH
    ds_out = dataset_path or MIMIC_TWOSIDES_ML_DATASET_PATH
    neg_pool_temp = pos_out.parent / f".{neg_out.stem}_candidate_pool.parquet"

    _, pos_stats = stream_dedupe_positives(
        labeled_path=labeled_path,
        output_path=pos_out,
        batch_size=batch_size,
        n_buckets=n_buckets,
        resume_buckets=resume_buckets,
    )

    try:
        neg_pool, neg_pool_stats = collect_negative_candidate_pairs(
            mimic_pairs_path=mimic_pairs_path,
            mapping_path=mapping_path,
            unique_pairs_path=unique_pairs_path,
            batch_size=batch_size,
            temp_pool_path=neg_pool_temp if require_real_scale else None,
        )
    finally:
        neg_pool_temp.unlink(missing_ok=True)
    negatives, neg_sample_stats = sample_negative_pairs(
        neg_pool,
        n_positives=pos_stats["positive_rows_after_dedup"],
        negative_ratio=negative_ratio,
        seed=seed,
    )
    neg_out.parent.mkdir(parents=True, exist_ok=True)
    negatives.to_parquet(neg_out, index=False)

    pos_patients = set()
    if pos_out.exists() and pos_out.stat().st_size > 0:
        pf = pq.ParquetFile(pos_out)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["patient_id"]):
            pos_patients.update(batch.to_pandas()["patient_id"].astype(str).tolist())

    neg_patients = set(negatives["patient_id"].astype(str).tolist()) if len(negatives) else set()
    _, split_stats = build_patient_splits_table(
        patient_ids=pos_patients | neg_patients,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_path=split_out,
    )

    _, assemble_stats = assemble_ml_dataset(
        positives_path=pos_out,
        negatives_path=neg_out,
        splits_path=split_out,
        output_path=ds_out,
        batch_size=batch_size,
    )

    stats = {
        **preflight,
        **schema_info,
        **pos_stats,
        **neg_pool_stats,
        **neg_sample_stats,
        **split_stats,
        **assemble_stats,
        "positive_count": pos_stats["positive_rows_after_dedup"],
        "negative_count": int(len(negatives)),
        "duplicate_rows_removed_during_dedup": pos_stats["duplicate_rows_removed"],
        "random_seed": seed,
        "paths": {
            "positives": str(pos_out),
            "negatives": str(neg_out),
            "splits": str(split_out),
            "dataset": str(ds_out),
        },
        "field_definitions": ML_FIELD_DEFINITIONS,
        "prepare_outputs": describe_prepare_outputs(),
    }
    metrics_out = stats_path or METRICS_DIR / "mimic_twosides_ml_dataset_statistics.json"
    write_json(metrics_out, stats)
    stats["stats_path"] = str(metrics_out)
    return stats
