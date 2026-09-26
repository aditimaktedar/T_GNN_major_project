"""Inspect a local dataset file without assuming a medical schema.

This tool reports file metadata and a small sample. It does not download data,
modify files, or invent column names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

# Identifier-like names are detected from the file's own headers, not from a
# guessed medical schema. A column is treated as an identifier candidate when
# its name looks like an ID field (for example drug_id or patient_key).
_IDENTIFIER_EXACT = {
    "id",
    "key",
    "code",
    "uuid",
    "cui",
    "rxcui",
    "rxnorm",
}
_IDENTIFIER_SUFFIXES = ("_id", "_key", "_code", "_uuid", "_cui")


def _looks_like_identifier(column_name: str) -> bool:
    lowered = str(column_name).strip().lower()
    if lowered in _IDENTIFIER_EXACT:
        return True
    return any(lowered.endswith(suffix) for suffix in _IDENTIFIER_SUFFIXES)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def detect_file_type(path: Path) -> str:
    """Return a simple type label from the filename suffix only."""
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".tsv.gz"):
        return "tsv.gz"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".tsv"):
        return "tsv"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if path.suffix:
        return path.suffix.lstrip(".").lower()
    return "unknown"


def _csv_read_kwargs(file_type: str) -> dict:
    kwargs: dict = {"low_memory": False}
    if file_type in {"tsv", "tsv.gz"}:
        kwargs["sep"] = "\t"
    if file_type.endswith(".gz"):
        kwargs["compression"] = "gzip"
    return kwargs


def _chunked_csv_stats(
    path: Path,
    file_type: str,
    chunk_size: int,
    unique_cap: int,
) -> dict:
    read_kwargs = _csv_read_kwargs(file_type)
    header = pd.read_csv(path, nrows=0, **read_kwargs)
    columns = [str(col) for col in header.columns]
    dtypes: dict[str, str] = {}
    missing = {col: 0 for col in columns}
    unique_sets = {col: set() for col in columns if _looks_like_identifier(col)}
    unique_capped = {col: False for col in unique_sets}
    n_rows = 0
    first_chunk = None

    reader: Iterator[pd.DataFrame] = pd.read_csv(
        path,
        chunksize=chunk_size,
        **read_kwargs,
    )
    for chunk in reader:
        if first_chunk is None:
            first_chunk = chunk
            dtypes = {str(col): str(dtype) for col, dtype in chunk.dtypes.items()}
        n_rows += len(chunk)
        for col in columns:
            if col in chunk.columns:
                missing[col] += int(chunk[col].isna().sum())
        for col, values in unique_sets.items():
            if unique_capped[col] or col not in chunk.columns:
                continue
            for value in chunk[col].dropna().unique().tolist():
                values.add(value)
                if len(values) >= unique_cap:
                    unique_capped[col] = True
                    break

    unique_counts = {}
    for col, values in unique_sets.items():
        unique_counts[col] = {
            "count": len(values),
            "capped": unique_capped[col],
        }

    return {
        "columns": columns,
        "n_columns": len(columns),
        "n_rows": n_rows,
        "dtypes": dtypes,
        "missing_value_counts": missing,
        "identifier_unique_counts": unique_counts,
        "first_chunk": first_chunk if first_chunk is not None else pd.DataFrame(),
        "row_count_method": f"chunked pandas read (chunk_size={chunk_size})",
    }


def _inspect_parquet(path: Path, sample_rows: int) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet inspection requires pyarrow. Install it in the project environment."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = list(schema.names)
    dtypes = {name: str(schema.field(name).type) for name in columns}
    n_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None

    if parquet_file.num_row_groups == 0:
        sample = pd.DataFrame(columns=columns)
    elif n_rows is not None and n_rows <= max(sample_rows, 0):
        sample = parquet_file.read().to_pandas()
    else:
        sample = parquet_file.read_row_group(0).to_pandas().head(sample_rows)

    missing = {col: int(sample[col].isna().sum()) for col in sample.columns}
    unique_counts = {}
    for col in columns:
        if _looks_like_identifier(col) and col in sample.columns:
            unique_counts[col] = {
                "count": int(sample[col].nunique(dropna=True)),
                "capped": False,
                "note": "unique counts are from the sampled rows only; full-file unique counts were not computed",
            }

    return {
        "columns": columns,
        "n_columns": len(columns),
        "n_rows": n_rows,
        "dtypes": dtypes,
        "missing_value_counts": missing,
        "missing_value_scope": "sample only; full-file missing counts were not loaded",
        "identifier_unique_counts": unique_counts,
        "sample": sample.head(sample_rows),
        "row_count_method": "parquet footer metadata (file not fully loaded)",
    }


def _inspect_jsonl(path: Path, sample_rows: int, max_unique_scan_rows: int) -> dict:
    n_rows = 0
    sample_records: list[dict] = []
    columns: list[str] = []
    missing = {}
    unique_sets: dict[str, set] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError("JSONL inspection currently supports object-per-line files only.")
            n_rows += 1
            if n_rows <= sample_rows:
                sample_records.append(record)
            if n_rows <= max_unique_scan_rows:
                for key in record:
                    if key not in columns:
                        columns.append(key)
                        missing[key] = 0
                        if _looks_like_identifier(key):
                            unique_sets[key] = set()
                for key in columns:
                    value = record.get(key)
                    if value is None:
                        missing[key] += 1
                    elif key in unique_sets:
                        unique_sets[key].add(value)

    unique_counts = {
        col: {"count": len(values), "capped": n_rows > max_unique_scan_rows}
        for col, values in unique_sets.items()
    }
    sample = pd.DataFrame(sample_records)
    dtypes = {str(col): str(dtype) for col, dtype in sample.dtypes.items()} if not sample.empty else {}
    note = None
    if n_rows > max_unique_scan_rows:
        note = (
            f"Missing-value and unique-value scans used the first {max_unique_scan_rows} rows only."
        )

    return {
        "columns": columns,
        "n_columns": len(columns),
        "n_rows": n_rows,
        "dtypes": dtypes,
        "missing_value_counts": missing,
        "identifier_unique_counts": unique_counts,
        "sample": sample,
        "row_count_method": "line-by-line JSONL parse",
        "scan_note": note,
    }


def _inspect_json(path: Path, sample_rows: int, max_json_bytes: int) -> dict:
    size = path.stat().st_size
    if size > max_json_bytes:
        return {
            "error": (
                f"JSON file is { _human_size(size) }, which is larger than the "
                f"safe in-memory limit ({_human_size(max_json_bytes)}). "
                "Use JSONL for large files, or inspect after converting to CSV."
            ),
            "columns": [],
            "n_columns": None,
            "n_rows": None,
            "dtypes": {},
            "missing_value_counts": {},
            "identifier_unique_counts": {},
            "sample": pd.DataFrame(),
            "row_count_method": "not loaded",
        }

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        frame = pd.json_normalize(payload)
    else:
        raise ValueError("JSON inspection supports an object or a list of objects.")

    columns = [str(col) for col in frame.columns]
    unique_counts = {}
    for col in columns:
        if _looks_like_identifier(col):
            unique_counts[col] = {
                "count": int(frame[col].nunique(dropna=True)),
                "capped": False,
            }

    return {
        "columns": columns,
        "n_columns": len(columns),
        "n_rows": int(len(frame)),
        "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        "missing_value_counts": {col: int(frame[col].isna().sum()) for col in columns},
        "identifier_unique_counts": unique_counts,
        "sample": frame.head(sample_rows),
        "row_count_method": "full JSON load (file was under the size limit)",
    }


def inspect_file(
    path: str | Path,
    sample_rows: int = 5,
    chunk_size: int = 100_000,
    unique_cap: int = 10_000,
    max_json_bytes: int = 50 * 1024 * 1024,
) -> dict:
    """Inspect one local file and return a structured report dictionary."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    file_type = detect_file_type(file_path)
    size = file_path.stat().st_size
    report = {
        "path": str(file_path),
        "size_bytes": size,
        "size_human": _human_size(size),
        "file_type": file_type,
        "empty": size == 0,
    }

    if size == 0:
        report.update(
            {
                "n_rows": 0,
                "n_columns": 0,
                "columns": [],
                "dtypes": {},
                "missing_value_counts": {},
                "identifier_unique_counts": {},
                "sample": [],
                "notes": ["File is empty."],
            }
        )
        return report

    notes: list[str] = []
    try:
        if file_type in {"csv", "tsv", "csv.gz", "tsv.gz"}:
            stats = _chunked_csv_stats(file_path, file_type, chunk_size, unique_cap)
            sample = stats.pop("first_chunk").head(sample_rows)
            report.update(stats)
            report["sample"] = sample
        elif file_type == "parquet":
            stats = _inspect_parquet(file_path, sample_rows)
            report.update(stats)
        elif file_type == "jsonl":
            stats = _inspect_jsonl(file_path, sample_rows, unique_cap)
            report.update(stats)
            if stats.get("scan_note"):
                notes.append(stats["scan_note"])
        elif file_type == "json":
            stats = _inspect_json(file_path, sample_rows, max_json_bytes)
            report.update(stats)
            if stats.get("error"):
                notes.append(stats["error"])
        else:
            notes.append(
                f"No tabular inspector is implemented for type '{file_type}'. "
                "Only path, size, and type are reported."
            )
            report.update(
                {
                    "n_rows": None,
                    "n_columns": None,
                    "columns": [],
                    "dtypes": {},
                    "missing_value_counts": {},
                    "identifier_unique_counts": {},
                    "sample": pd.DataFrame(),
                    "row_count_method": "not inspected",
                }
            )
    except Exception as exc:
        notes.append(f"Inspection failed: {exc}")
        report.update(
            {
                "n_rows": None,
                "n_columns": None,
                "columns": [],
                "dtypes": {},
                "missing_value_counts": {},
                "identifier_unique_counts": {},
                "sample": pd.DataFrame(),
            }
        )

    if report.get("identifier_unique_counts"):
        for col, info in report["identifier_unique_counts"].items():
            if info.get("capped"):
                notes.append(
                    f"Unique-value count for '{col}' stopped at {unique_cap} distinct values."
                )

    report["notes"] = notes
    sample = report.get("sample")
    if isinstance(sample, pd.DataFrame):
        cleaned = sample.astype(object)
        cleaned = cleaned.where(pd.notna(sample), None)
        report["sample"] = cleaned.to_dict(orient="records")
    return report


def format_report(report: dict) -> str:
    """Turn an inspect_file() dictionary into beginner-friendly text."""
    lines = [
        "Dataset inspection report",
        "=========================",
        f"Path: {report['path']}",
        f"Size: {report['size_human']} ({report['size_bytes']} bytes)",
        f"Type: {report['file_type']}",
        f"Empty: {report['empty']}",
    ]
    if report.get("n_rows") is not None:
        lines.append(f"Rows: {report['n_rows']}")
    if report.get("row_count_method"):
        lines.append(f"Row count method: {report['row_count_method']}")
    if report.get("n_columns") is not None:
        lines.append(f"Columns: {report['n_columns']}")
    if report.get("columns"):
        lines.append("Column names:")
        for name in report["columns"]:
            dtype = report.get("dtypes", {}).get(name, "unknown")
            missing = report.get("missing_value_counts", {}).get(name, "n/a")
            lines.append(f"  - {name}  (dtype={dtype}, missing={missing})")
    if report.get("missing_value_scope"):
        lines.append(f"Missing-value scope: {report['missing_value_scope']}")
    if report.get("identifier_unique_counts"):
        lines.append("Unique values for identifier-like columns:")
        for name, info in report["identifier_unique_counts"].items():
            suffix = "+" if info.get("capped") else ""
            extra = f"  note: {info['note']}" if info.get("note") else ""
            lines.append(f"  - {name}: {info['count']}{suffix}{extra}")
    sample = report.get("sample") or []
    if sample:
        lines.append("Sample records:")
        lines.append(json.dumps(sample, indent=2, default=str, allow_nan=False))
    notes = report.get("notes") or []
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a local dataset file and print a safe summary."
    )
    parser.add_argument("file", help="Path to a local file (CSV, TSV, Parquet, JSON, JSONL).")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="How many records to show in the sample (default: 5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for CSV/TSV reading (default: 100000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_file(args.file, sample_rows=args.sample_rows, chunk_size=args.chunk_size)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
