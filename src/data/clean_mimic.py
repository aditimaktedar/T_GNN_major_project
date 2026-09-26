"""Clean standardized MIMIC prescription tables without touching raw files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import DEFAULT_MIMIC_CHUNKSIZE, METRICS_DIR, MIMIC_CLEAN_PATH
from src.data.exceptions import MissingInputError, SchemaError
from src.data.io_utils import load_table_chunked, normalize_text, write_json, write_table


def _parse_time(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    invalid = int(((series.notna()) & (series.astype(str).str.strip() != "") & parsed.isna()).sum())
    return parsed, invalid


def _clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if "patient_id" not in frame.columns or "drug_name_raw" not in frame.columns:
        raise SchemaError(
            "Cleaning requires patient_id and drug_name_raw columns produced by mimic loading. "
            f"Available columns: {list(frame.columns)}"
        )

    input_rows = int(len(frame))
    working = frame.copy()
    working["_drop_reason"] = pd.Series([None] * len(working), dtype="object")

    missing_patient = working["patient_id"].isna() | (
        working["patient_id"].astype(str).str.strip() == ""
    )
    working.loc[missing_patient & working["_drop_reason"].isna(), "_drop_reason"] = "missing_patient_id"

    missing_drug = working["drug_name_raw"].isna() | (
        working["drug_name_raw"].astype(str).str.strip() == ""
    )
    working.loc[missing_drug & working["_drop_reason"].isna(), "_drop_reason"] = "missing_drug_name"

    invalid_start = 0
    invalid_end = 0
    if "start_time" in working.columns:
        parsed_start, invalid_start = _parse_time(working["start_time"])
        working["start_time"] = parsed_start
        bad_start = working["start_time"].isna() & frame["start_time"].notna()
        working.loc[bad_start & working["_drop_reason"].isna(), "_drop_reason"] = "invalid_start_time"
    if "end_time" in working.columns:
        parsed_end, invalid_end = _parse_time(working["end_time"])
        working["end_time"] = parsed_end
        bad_end = working["end_time"].isna() & frame["end_time"].notna()
        working.loc[bad_end & working["_drop_reason"].isna(), "_drop_reason"] = "invalid_end_time"

    if {"start_time", "end_time"}.issubset(working.columns):
        inverted = (
            working["start_time"].notna()
            & working["end_time"].notna()
            & (working["start_time"] > working["end_time"])
        )
        working.loc[inverted & working["_drop_reason"].isna(), "_drop_reason"] = "inverted_interval"

    working["drug_name_norm"] = working["drug_name_raw"].map(normalize_text)
    working["drug_name_norm"] = working["drug_name_norm"].map(
        lambda value: value.lower() if isinstance(value, str) else value
    )

    dropped = working[working["_drop_reason"].notna()].copy()
    kept = working[working["_drop_reason"].isna()].drop(columns=["_drop_reason"]).copy()
    chunk_stats = {
        "input_rows": input_rows,
        "output_rows": int(len(kept)),
        "missing_patient_values": int(missing_patient.sum()),
        "missing_drug_values": int(missing_drug.sum()),
        "invalid_start_times": invalid_start,
        "invalid_end_times": invalid_end,
        "invalid_dates": invalid_start + invalid_end,
        "rows_flagged_unusable": int(len(dropped)),
    }
    return kept, dropped, chunk_stats


def _merge_stats(total: dict, chunk: dict) -> None:
    for key, value in chunk.items():
        if key not in total:
            total[key] = value
        elif isinstance(value, int):
            total[key] += value


def _write_batches(frames: list[pd.DataFrame], output_path: Path) -> int:
    writer: pq.ParquetWriter | None = None
    total = 0
    for frame in frames:
        if not len(frame):
            continue
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total += len(frame)
    if writer is not None:
        writer.close()
    elif not output_path.exists():
        write_table(pd.DataFrame(), output_path)
    return total


def clean_prescriptions(
    prescriptions: pd.DataFrame | Path,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    chunksize: int = DEFAULT_MIMIC_CHUNKSIZE,
    streaming: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Clean prescriptions and write dropped-row reasons instead of dropping silently."""
    out = output_path if output_path is not None else MIMIC_CLEAN_PATH
    stats: dict = {
        "duplicate_rows_removed": 0,
        "drop_reason_counts": {},
        "note": (
            "Unusable rows were not deleted silently: they are counted here and written to "
            "data/interim/mimic_prescriptions_dropped.parquet when any exist. Exact duplicate "
            "removal is deferred to exposure aggregation for large streaming runs."
        ),
    }
    dropped_frames: list[pd.DataFrame] = []
    kept_frames: list[pd.DataFrame] = []
    stream_write = False

    if isinstance(prescriptions, Path):
        if not prescriptions.exists():
            raise MissingInputError(
                f"Clean step needs a standardized prescription table at {prescriptions}. "
                "Run: python -m src.data.pipeline ingest-mimic"
            )
        if streaming and prescriptions.suffix.lower() == ".parquet":
            stream_write = True
            writer: pq.ParquetWriter | None = None
            dropped_writer: pq.ParquetWriter | None = None
            pf = pq.ParquetFile(prescriptions)
            for batch in pf.iter_batches(batch_size=chunksize):
                kept, dropped, chunk_stats = _clean_frame(batch.to_pandas())
                if len(kept):
                    table = pa.Table.from_pandas(kept, preserve_index=False)
                    if writer is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(out, table.schema)
                    writer.write_table(table)
                if len(dropped):
                    drop_table = pa.Table.from_pandas(dropped, preserve_index=False)
                    dropped_path = out.with_name("mimic_prescriptions_dropped.parquet")
                    if dropped_writer is None:
                        dropped_path.parent.mkdir(parents=True, exist_ok=True)
                        dropped_writer = pq.ParquetWriter(dropped_path, drop_table.schema)
                    dropped_writer.write_table(drop_table)
                _merge_stats(stats, chunk_stats)
            if writer is not None:
                writer.close()
            elif not out.exists():
                write_table(pd.DataFrame(), out)
            if dropped_writer is not None:
                dropped_writer.close()
                stats["dropped_rows_path"] = str(out.with_name("mimic_prescriptions_dropped.parquet"))
        else:
            frame = load_table_chunked(prescriptions, chunksize=chunksize)
            kept, dropped, chunk_stats = _clean_frame(frame)
            kept_frames.append(kept)
            if len(dropped):
                dropped_frames.append(dropped)
            _merge_stats(stats, chunk_stats)
    else:
        kept, dropped, chunk_stats = _clean_frame(prescriptions.copy())
        kept_frames.append(kept)
        if len(dropped):
            dropped_frames.append(dropped)
        _merge_stats(stats, chunk_stats)

    if not stream_write:
        kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
        before_dedup = int(len(kept))
        kept = kept.drop_duplicates()
        stats["duplicate_rows_removed"] = before_dedup - int(len(kept))
        if dropped_frames:
            dropped_all = pd.concat(dropped_frames, ignore_index=True)
            stats["drop_reason_counts"] = dropped_all["_drop_reason"].value_counts().astype(int).to_dict()
            dropped_path = out.with_name("mimic_prescriptions_dropped.parquet")
            write_table(dropped_all, dropped_path)
            stats["dropped_rows_path"] = str(dropped_path)
        write_table(kept, out)
        stats["records_retained"] = int(len(kept))
        result_frame = kept
    else:
        stats["records_retained"] = stats.get("output_rows", 0)
        stats["streaming"] = True
        if stats.get("rows_flagged_unusable", 0):
            stats["drop_reason_counts"] = {"see_dropped_file": stats["rows_flagged_unusable"]}
        result_frame = pd.DataFrame()

    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "mimic_cleaning_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    stats["stats_path"] = str(metrics_out)
    return result_frame, stats
