"""Construct patient-level concurrent drug pairs.

Temporal / concurrency rules (choose one; do not invent another):

- interval_overlap: two distinct drugs for the same patient form a pair when
  their [start_time, end_time] intervals overlap (start <= other_end and
  other_start <= end). Requires start_time and end_time.
- same_admission: two distinct drugs for the same patient and admission_id.
  Requires admission_id. Does not use clock times.
- same_calendar_day: two distinct drugs for the same patient whose start_time
  dates are the same. Requires start_time.

If the requested rule needs columns that are absent, this module raises
ConfigurationError instead of inventing timestamps.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data.config import DEFAULT_MIMIC_CHUNKSIZE, DRUG_PAIRS_PATH, METRICS_DIR, MIMIC_EXPOSURES_PATH
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.io_utils import load_table_chunked, write_json, write_table

VALID_RULES = ("interval_overlap", "same_admission", "same_calendar_day")


def canonicalize_pair(drug_a: object, drug_b: object) -> tuple[str, str] | None:
    a = str(drug_a).strip() if drug_a is not None and not pd.isna(drug_a) else ""
    b = str(drug_b).strip() if drug_b is not None and not pd.isna(drug_b) else ""
    if not a or not b or a == b:
        return None
    return (a, b) if a < b else (b, a)


def _drug_key_column(frame: pd.DataFrame) -> str:
    if "rxnorm_id" in frame.columns and frame["rxnorm_id"].notna().any():
        return "rxnorm_id"
    if "drug_name_norm" in frame.columns:
        return "drug_name_norm"
    if "drug_name_raw" in frame.columns:
        return "drug_name_raw"
    raise SchemaError("No drug identifier column available for pair construction.")


def _intervals_overlap(start_a, end_a, start_b, end_b) -> bool:
    if pd.isna(start_a) or pd.isna(end_a) or pd.isna(start_b) or pd.isna(end_b):
        return False
    return start_a <= end_b and start_b <= end_a


def aggregate_drug_exposures(
    prescriptions: pd.DataFrame | Path,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    chunksize: int = DEFAULT_MIMIC_CHUNKSIZE,
) -> tuple[pd.DataFrame, dict]:
    """Collapse repeated prescription rows to unique patient/admission/drug exposures."""
    input_rows = 0
    aggregates: dict[tuple, dict] = {}

    def _consume_frame(frame: pd.DataFrame) -> None:
        nonlocal input_rows
        if "patient_id" not in frame.columns:
            raise SchemaError("Exposure aggregation requires patient_id.")
        input_rows += len(frame)
        drug_col = _drug_key_column(frame)
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            patient_id = row_dict["patient_id"]
            admission_id = row_dict.get("admission_id")
            drug_value = row_dict.get(drug_col)
            if drug_value is None or pd.isna(drug_value) or str(drug_value).strip() == "":
                continue
            drug_key = str(drug_value).strip()
            key = (patient_id, admission_id, drug_key) if admission_id is not None else (patient_id, drug_key)
            record = aggregates.get(key)
            if record is None:
                record = {
                    "patient_id": patient_id,
                    "admission_id": admission_id,
                    drug_col: drug_key,
                    "drug_name_raw": row_dict.get("drug_name_raw"),
                    "drug_name_norm": row_dict.get("drug_name_norm"),
                    "route": row_dict.get("route"),
                    "ndc": row_dict.get("ndc"),
                    "gsn": row_dict.get("gsn"),
                    "formulary_drug_cd": row_dict.get("formulary_drug_cd"),
                    "start_time": row_dict.get("start_time"),
                    "end_time": row_dict.get("end_time"),
                }
                aggregates[key] = record
                continue
            start = row_dict.get("start_time")
            end = row_dict.get("end_time")
            if start is not None and not pd.isna(start):
                if record["start_time"] is None or pd.isna(record["start_time"]) or start < record["start_time"]:
                    record["start_time"] = start
            if end is not None and not pd.isna(end):
                if record["end_time"] is None or pd.isna(record["end_time"]) or end > record["end_time"]:
                    record["end_time"] = end

    if isinstance(prescriptions, Path):
        if not prescriptions.exists():
            raise MissingInputError(f"Exposure aggregation needs prescriptions at {prescriptions}.")
        pf = pq.ParquetFile(prescriptions)
        drug_col = None
        for batch in pf.iter_batches(batch_size=chunksize):
            frame = batch.to_pandas()
            if drug_col is None:
                drug_col = _drug_key_column(frame)
            _consume_frame(frame)
    else:
        drug_col = _drug_key_column(prescriptions)
        _consume_frame(prescriptions.copy())

    exposures = pd.DataFrame.from_records(list(aggregates.values()))
    stats = {
        "input_rows": int(input_rows),
        "output_rows": int(len(exposures)),
        "drug_id_field": drug_col or "drug_name_norm",
        "group_columns": ["patient_id", "admission_id", drug_col or "drug_name_norm"],
        "note": "Repeated prescription rows were collapsed before pair construction.",
    }
    out = output_path if output_path is not None else MIMIC_EXPOSURES_PATH
    write_table(exposures, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "mimic_exposure_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return exposures, stats


def _construct_same_admission_pairs_fast(frame: pd.DataFrame, drug_col: str) -> pd.DataFrame:
    """Vectorized same-admission pairing on deduplicated exposures."""
    cols = ["patient_id", "admission_id", drug_col]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise SchemaError(f"same_admission pairing missing columns: {missing}")
    unique = frame[cols].dropna(subset=[drug_col, "patient_id", "admission_id"]).drop_duplicates()
    unique = unique.rename(columns={drug_col: "drug_a"})
    unique["drug_a"] = unique["drug_a"].astype(str).str.strip()
    unique = unique[unique["drug_a"] != ""]
    right = unique.rename(columns={"drug_a": "drug_b"})
    pairs = unique.merge(right, on=["patient_id", "admission_id"], how="inner")
    pairs = pairs[pairs["drug_a"] < pairs["drug_b"]].copy()
    pairs["pair_key"] = pairs["drug_a"] + "||" + pairs["drug_b"]
    pairs["pair_rule"] = "same_admission"
    pairs["drug_id_field"] = drug_col
    return pairs.drop_duplicates(subset=["patient_id", "admission_id", "pair_key"])


def _construct_pairs_from_frame(
    frame: pd.DataFrame,
    pair_rule: str,
    drug_col: str,
) -> tuple[pd.DataFrame, dict]:
    if pair_rule == "same_admission":
        pairs = _construct_same_admission_pairs_fast(frame, drug_col)
        return pairs, {"duplicate_pair_records_removed": 0, "skipped_incomplete_intervals": 0}

    working = frame.copy()
    working["_drug_key"] = working[drug_col].map(lambda value: None if pd.isna(value) else str(value).strip())
    working = working[working["_drug_key"].notna() & (working["_drug_key"] != "")]

    records: list[dict] = []
    skipped_incomplete_intervals = 0

    if pair_rule == "same_admission":
        group_cols = ["patient_id", "admission_id"]
    elif pair_rule == "same_calendar_day":
        working["_day"] = pd.to_datetime(working["start_time"], errors="coerce").dt.date
        group_cols = ["patient_id", "_day"]
    else:
        group_cols = ["patient_id"]

    for _, group in working.groupby(group_cols, dropna=True):
        if pair_rule == "interval_overlap":
            rows = group[["_drug_key", "start_time", "end_time"]].to_dict("records")
            seen_pairs = set()
            for left, right in combinations(rows, 2):
                pair = canonicalize_pair(left["_drug_key"], right["_drug_key"])
                if pair is None:
                    continue
                if (
                    pd.isna(left["start_time"])
                    or pd.isna(left["end_time"])
                    or pd.isna(right["start_time"])
                    or pd.isna(right["end_time"])
                ):
                    skipped_incomplete_intervals += 1
                    continue
                if not _intervals_overlap(
                    left["start_time"], left["end_time"], right["start_time"], right["end_time"]
                ):
                    continue
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                overlap_start = max(left["start_time"], right["start_time"])
                overlap_end = min(left["end_time"], right["end_time"])
                record = {
                    "patient_id": group["patient_id"].iloc[0],
                    "drug_a": pair[0],
                    "drug_b": pair[1],
                    "pair_key": f"{pair[0]}||{pair[1]}",
                    "pair_rule": pair_rule,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "drug_id_field": drug_col,
                }
                if "admission_id" in group.columns:
                    record["admission_id"] = group["admission_id"].iloc[0]
                records.append(record)
        else:
            drugs = sorted({drug for drug in group["_drug_key"].tolist() if drug})
            patient_id = group["patient_id"].iloc[0]
            extra = {}
            if pair_rule == "same_admission":
                extra["admission_id"] = group["admission_id"].iloc[0]
            if pair_rule == "same_calendar_day":
                extra["event_date"] = str(group["_day"].iloc[0])
            for drug_a, drug_b in combinations(drugs, 2):
                pair = canonicalize_pair(drug_a, drug_b)
                if pair is None:
                    continue
                records.append(
                    {
                        "patient_id": patient_id,
                        "drug_a": pair[0],
                        "drug_b": pair[1],
                        "pair_key": f"{pair[0]}||{pair[1]}",
                        "pair_rule": pair_rule,
                        "drug_id_field": drug_col,
                        **extra,
                    }
                )

    pairs = pd.DataFrame.from_records(records)
    if len(pairs):
        before = int(len(pairs))
        pairs = pairs.drop_duplicates(subset=["patient_id", "pair_key"] + (["admission_id"] if "admission_id" in pairs.columns else []))
        duplicates_removed = before - int(len(pairs))
    else:
        duplicates_removed = 0
        pairs = pd.DataFrame(
            columns=["patient_id", "drug_a", "drug_b", "pair_key", "pair_rule", "drug_id_field"]
        )

    stats = {
        "duplicate_pair_records_removed": int(duplicates_removed),
        "skipped_incomplete_intervals": int(skipped_incomplete_intervals),
    }
    return pairs, stats


def construct_drug_pairs(
    prescriptions: pd.DataFrame | Path,
    pair_rule: str = "interval_overlap",
    output_path: Path | None = None,
    stats_path: Path | None = None,
    aggregate_exposures_first: bool = True,
    exposures_path: Path | None = None,
    chunksize: int = DEFAULT_MIMIC_CHUNKSIZE,
) -> tuple[pd.DataFrame, dict]:
    if pair_rule not in VALID_RULES:
        raise ConfigurationError(f"Unknown pair_rule {pair_rule!r}. Valid: {VALID_RULES}")

    if isinstance(prescriptions, Path):
        if not prescriptions.exists():
            raise MissingInputError(f"Pair construction needs prescriptions at {prescriptions}.")
        if aggregate_exposures_first:
            exposures, exposure_stats = aggregate_drug_exposures(
                prescriptions,
                output_path=exposures_path or MIMIC_EXPOSURES_PATH,
                chunksize=chunksize,
            )
            frame = exposures
        else:
            frame = load_table_chunked(prescriptions, chunksize=chunksize)
            exposure_stats = {"aggregate_exposures_first": False}
    else:
        frame = prescriptions.copy()
        exposure_stats = {"aggregate_exposures_first": False}

    if "patient_id" not in frame.columns:
        raise SchemaError("Pair construction requires patient_id.")

    if pair_rule == "interval_overlap" and not {"start_time", "end_time"}.issubset(frame.columns):
        raise ConfigurationError(
            "pair_rule='interval_overlap' needs start_time and end_time. "
            f"Available columns: {list(frame.columns)}"
        )
    if pair_rule == "same_admission" and "admission_id" not in frame.columns:
        raise ConfigurationError(
            "pair_rule='same_admission' needs admission_id. "
            f"Available columns: {list(frame.columns)}"
        )
    if pair_rule == "same_calendar_day" and "start_time" not in frame.columns:
        raise ConfigurationError(
            "pair_rule='same_calendar_day' needs start_time. "
            f"Available columns: {list(frame.columns)}"
        )

    drug_col = _drug_key_column(frame)
    pairs, pair_stats = _construct_pairs_from_frame(frame, pair_rule, drug_col)

    stats = {
        "pair_rule": pair_rule,
        "pair_rule_description": {
            "interval_overlap": "Distinct drugs for the same patient whose start/end intervals overlap.",
            "same_admission": "Distinct drugs sharing patient_id and admission_id.",
            "same_calendar_day": "Distinct drugs for the same patient with the same start_time date.",
        }[pair_rule],
        "n_pairs": int(len(pairs)),
        "n_patients": int(pairs["patient_id"].nunique()) if len(pairs) else 0,
        "n_unique_drugs": int(len(set(pairs["drug_a"]).union(set(pairs["drug_b"])))) if len(pairs) else 0,
        "n_unique_pairs": int(pairs["pair_key"].nunique()) if len(pairs) else 0,
        "drug_id_field": drug_col,
        "self_pairs": 0,
        "aggregate_exposures_first": aggregate_exposures_first,
        **exposure_stats,
        **pair_stats,
        "note": "Unordered pairs (drug_a < drug_b). A-A self pairs are not created.",
    }
    out = output_path if output_path is not None else DRUG_PAIRS_PATH
    write_table(pairs, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "drug_pair_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return pairs, stats
