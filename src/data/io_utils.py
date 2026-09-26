"""Shared table I/O for the Member B pipeline.

Raw files are only read. Intermediate and processed files may be written.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd

from src.data.inspect_data import detect_file_type

TABULAR_TYPES = {"csv", "tsv", "csv.gz", "tsv.gz", "parquet"}


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> Path:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    suffix = path.name.lower()
    if suffix.endswith(".parquet"):
        frame.to_parquet(path, index=False)
    elif suffix.endswith(".tsv") or suffix.endswith(".tsv.gz"):
        frame.to_csv(path, index=False, sep="\t")
    else:
        frame.to_csv(path, index=False)
    return path


def read_table(path: Path, chunksize: int | None = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
    file_type = detect_file_type(path)
    if file_type not in TABULAR_TYPES:
        raise ValueError(f"Unsupported tabular type for {path}: {file_type}")
    if file_type == "parquet":
        if chunksize is not None:
            raise ValueError("Chunked parquet reads are not used; parquet metadata/columns are loaded via pandas.")
        return pd.read_parquet(path)
    kwargs: dict = {}
    if file_type in {"tsv", "tsv.gz"}:
        kwargs["sep"] = "\t"
    if file_type.endswith(".gz"):
        kwargs["compression"] = "gzip"
    if chunksize is not None:
        return pd.read_csv(path, chunksize=chunksize, **kwargs)
    return pd.read_csv(path, **kwargs)


def read_header(path: Path) -> list[str]:
    file_type = detect_file_type(path)
    if file_type == "parquet":
        frame = pd.read_parquet(path)
        return [str(col) for col in frame.columns]
    kwargs: dict = {"nrows": 0}
    if file_type in {"tsv", "tsv.gz"}:
        kwargs["sep"] = "\t"
    if file_type.endswith(".gz"):
        kwargs["compression"] = "gzip"
    header = pd.read_csv(path, **kwargs)
    return [str(col) for col in header.columns]


def load_table_chunked(path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    file_type = detect_file_type(path)
    if file_type == "parquet":
        return pd.read_parquet(path)
    chunks = list(read_table(path, chunksize=chunksize))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def iter_tabular_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if detect_file_type(path) in TABULAR_TYPES:
            files.append(path)
    return files


def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    """Return the actual column whose name matches an alias (case-insensitive).

    No fuzzy medical matching: only exact name matches after case fold.
    """
    lowered = {str(col).strip().lower(): str(col) for col in columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def resolve_columns(columns: Sequence[str], mapping: dict[str, Sequence[str]]) -> dict[str, str]:
    resolved = {}
    for canonical, aliases in mapping.items():
        actual = resolve_column(columns, aliases)
        if actual is not None:
            resolved[canonical] = actual
    return resolved


def normalize_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = " ".join(str(value).strip().split())
    return text or None
