"""Discover and load MIMIC-IV prescription tables without modifying raw files.

Supports the MIMIC-IV 2.1 layout (hosp/prescriptions.csv) via streaming reads for
large hospital prescription tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import (
    DEFAULT_MIMIC_CHUNKSIZE,
    MIMIC_DIR,
    MIMIC_HOSP_PRESCRIPTIONS,
    MIMIC_IV_ROOT,
    MIMIC_PRESCRIPTIONS_PATH,
)
from src.data.exceptions import MissingInputError
from src.data.inspect_data import detect_file_type
from src.data.io_utils import (
    iter_tabular_files,
    read_header,
    resolve_columns,
    write_json,
    write_table,
)

# Public MIMIC-IV hosp.prescriptions column names plus generic aliases.
PRESCRIPTION_COLUMN_ALIASES = {
    "patient_id": ("patient_id", "subject_id"),
    "admission_id": ("admission_id", "hadm_id"),
    "drug_name_raw": ("drug_name_raw", "drug", "drug_name", "medication"),
    "ndc": ("ndc",),
    "gsn": ("gsn",),
    "formulary_drug_cd": ("formulary_drug_cd",),
    "drug_type": ("drug_type",),
    "start_time": ("start_time", "starttime", "startdate", "start_date"),
    "end_time": ("end_time", "stoptime", "endtime", "end_date", "stop_date", "stopdate"),
    "dose": ("dose", "dose_val_rx"),
    "route": ("route",),
}

FILENAME_HINTS = ("prescription", "pharmacy", "pharm", "medication", "med")
EXCLUDED_FILENAME_FRAGMENTS = (
    "emar_detail",
    "emar.csv",
    "poe_detail",
    "labevents",
    "chartevents",
    "inputevents",
    "outputevents",
    "diagnoses",
    "procedures",
    "microbiology",
    "transfers",
    "admissions.csv",
    "patients.csv",
    "d_icd",
    "d_hcpcs",
    "d_labitems",
    "d_items",
    "services.csv",
    "drgcodes",
    "hcpcsevents",
    "omr.csv",
    "ingredientevents",
    "datetimeevents",
    "procedureevents",
    "icustays",
)

PREFERRED_RELATIVE_PATHS = (
    "hosp/prescriptions.csv",
    "prescriptions.csv",
)


def resolve_mimic_root(mimic_dir: Path | None = None) -> Path:
    """Return the directory that contains MIMIC tabular files."""
    if mimic_dir is not None:
        return mimic_dir
    if MIMIC_IV_ROOT.exists():
        return MIMIC_IV_ROOT
    return MIMIC_DIR


def _filename_looks_like_prescription(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in FILENAME_HINTS)


def _filename_is_excluded(path: Path) -> bool:
    lowered = str(path).lower()
    return any(fragment in lowered for fragment in EXCLUDED_FILENAME_FRAGMENTS)


def _score_candidate(path: Path, resolved: dict[str, str]) -> int:
    score = 0
    rel = path.as_posix()
    for preferred in PREFERRED_RELATIVE_PATHS:
        if rel.endswith(preferred):
            score += 10
            break
    if path.name.lower() == "prescriptions.csv":
        score += 8
    if _filename_looks_like_prescription(path):
        score += 2
    if "patient_id" in resolved:
        score += 2
    if "drug_name_raw" in resolved:
        score += 2
    if "admission_id" in resolved:
        score += 1
    if "start_time" in resolved or "end_time" in resolved:
        score += 1
    if path.name.lower() == "pharmacy.csv":
        score -= 1
    return score


def discover_prescription_files(mimic_dir: Path | None = None) -> list[dict]:
    """Find candidate prescription files from filenames and actual headers."""
    root = resolve_mimic_root(mimic_dir)
    if not root.exists():
        raise MissingInputError(
            "No MIMIC-IV folder was found. Place the approved MIMIC-IV 2.1 files under "
            f"{MIMIC_DIR} or {MIMIC_IV_ROOT} without modifying them. Source: "
            "https://www.kaggle.com/datasets/mangeshwagle/mimic-iv-2-1"
        )

    discovered = []
    for path in iter_tabular_files(root):
        if _filename_is_excluded(path):
            continue
        if detect_file_type(path) not in {"csv", "tsv", "csv.gz", "tsv.gz", "parquet"}:
            continue
        columns = read_header(path)
        resolved = resolve_columns(columns, PRESCRIPTION_COLUMN_ALIASES)
        discovered.append(
            {
                "path": path,
                "columns": columns,
                "resolved": resolved,
                "score": _score_candidate(path, resolved),
                "filename_hint": _filename_looks_like_prescription(path),
            }
        )

    usable = [
        item
        for item in discovered
        if "patient_id" in item["resolved"] and "drug_name_raw" in item["resolved"]
    ]
    usable.sort(key=lambda item: (-item["score"], str(item["path"])))
    return usable


def select_prescription_file(mimic_dir: Path | None = None) -> dict:
    """Return the single best prescription source file for ingestion."""
    if mimic_dir is not None:
        candidates = discover_prescription_files(mimic_dir)
        if not candidates:
            raise MissingInputError(
                f"No usable MIMIC-IV prescription table was found under {mimic_dir}."
            )
        best = candidates[0]
        best["selection_reason"] = "highest-scoring discovered prescription file"
        return best

    if MIMIC_HOSP_PRESCRIPTIONS.exists():
        columns = read_header(MIMIC_HOSP_PRESCRIPTIONS)
        resolved = resolve_columns(columns, PRESCRIPTION_COLUMN_ALIASES)
        if "patient_id" in resolved and "drug_name_raw" in resolved:
            return {
                "path": MIMIC_HOSP_PRESCRIPTIONS,
                "columns": columns,
                "resolved": resolved,
                "score": 100,
                "filename_hint": True,
                "selection_reason": "preferred MIMIC-IV 2.1 hosp/prescriptions.csv",
            }

    candidates = discover_prescription_files(None)
    if not candidates:
        root = resolve_mimic_root(None)
        compressed = list(root.rglob("*.7z")) + list(root.rglob("*.zip"))
        extra = ""
        if compressed:
            extra = (
                " Compressed archives were found but are not read directly. "
                "Extract CSV/Parquet files first, without changing the originals' content."
            )
        raise MissingInputError(
            "No usable MIMIC-IV prescription table was found under "
            f"{root}. Need a tabular file whose actual headers include a patient "
            "identifier (subject_id or patient_id) and a drug name (drug, drug_name, "
            f"medication, or drug_name_raw).{extra}"
        )
    best = candidates[0]
    best["selection_reason"] = "highest-scoring discovered prescription file"
    return best


def _coerce_column_types(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    for col in coerced.columns:
        if col in {"patient_id", "admission_id"}:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        elif col in {"start_time", "end_time"}:
            continue
        else:
            coerced[col] = coerced[col].astype("string")
    return coerced


def _standardize_chunk(raw: pd.DataFrame, resolved: dict[str, str], source_file: str) -> pd.DataFrame:
    standardized = pd.DataFrame()
    for canonical, actual in resolved.items():
        standardized[canonical] = raw[actual]
    standardized["source_file"] = source_file
    return _coerce_column_types(standardized)


def _load_prescriptions_streaming(
    path: Path,
    resolved: dict[str, str],
    output_path: Path,
    chunksize: int,
) -> tuple[int, dict]:
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    file_type = detect_file_type(path)
    if file_type == "parquet":
        frame = pd.read_parquet(path)
        standardized = _standardize_chunk(frame, resolved, str(path))
        write_table(standardized, output_path)
        return int(len(standardized)), {"streaming": False, "chunks_processed": 1}

    kwargs: dict = {"chunksize": chunksize, "low_memory": False}
    if file_type in {"tsv", "tsv.gz"}:
        kwargs["sep"] = "\t"
    if file_type.endswith(".gz"):
        kwargs["compression"] = "gzip"

    for chunk in pd.read_csv(path, **kwargs):
        standardized = _standardize_chunk(chunk, resolved, str(path))
        table = pa.Table.from_pandas(standardized, preserve_index=False)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total_rows += len(standardized)

    if writer is not None:
        writer.close()
    else:
        write_table(pd.DataFrame(columns=list(resolved.keys()) + ["source_file"]), output_path)
    return total_rows, {"streaming": True, "chunksize": chunksize}


def load_prescriptions(
    mimic_dir: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = DEFAULT_MIMIC_CHUNKSIZE,
    streaming: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load the primary MIMIC prescription table into a standardized interim file."""
    selected = select_prescription_file(mimic_dir)
    path: Path = selected["path"]
    resolved: dict[str, str] = selected["resolved"]
    out = output_path if output_path is not None else MIMIC_PRESCRIPTIONS_PATH

    if streaming and path.suffix.lower() == ".csv":
        n_rows, stream_meta = _load_prescriptions_streaming(path, resolved, out, chunksize)
        table = pd.DataFrame(columns=list(resolved.keys()) + ["source_file"])
    else:
        from src.data.io_utils import load_table_chunked

        raw = load_table_chunked(path, chunksize=chunksize)
        table = _standardize_chunk(raw, resolved, str(path))
        write_table(table, out)
        n_rows = int(len(table))
        stream_meta = {"streaming": False}

    metadata = {
        "n_rows": n_rows,
        "n_source_files": 1,
        "source_files": [
            {
                "path": str(path),
                "n_rows": n_rows,
                "actual_columns": selected["columns"],
                "source_column_map": resolved,
                "selection_reason": selected.get("selection_reason"),
            }
        ],
        "canonical_columns": list(resolved.keys()) + ["source_file"],
        "mimic_root": str(resolve_mimic_root(mimic_dir)),
        "streaming": stream_meta,
        "note": (
            "Only the primary prescription table is ingested. pharmacy.csv and eMAR tables "
            "are excluded unless no prescriptions table exists. Canonical names are aliases "
            "applied only to columns that existed in the discovered file."
        ),
    }
    write_json(out.with_name("mimic_prescriptions_metadata.json"), metadata)
    metadata["output_path"] = str(out)
    return table, metadata
