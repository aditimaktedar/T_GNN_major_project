"""Prepare local-only MIMIC → TWOSIDES PubChem CID mapping candidates.

This stage does NOT call PubChem or any external API. It uses only local files:
- MIMIC drug exposures
- TWOSIDES drug catalog
- Optional existing on-disk PubChem CID lookup cache from a prior run

No fuzzy matching. No invented mappings.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data.config import (
    METRICS_DIR,
    MIMIC_EXPOSURES_PATH,
    MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH,
    PUBCHEM_CACHE_DIR,
    TWOSIDES_DRUGS_PATH,
)
from src.data.exceptions import MissingInputError
from src.data.io_utils import write_json, write_table


def normalize_pubchem_query(name: object) -> str | None:
    """Conservative normalization shared with PubChem mapping utilities."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = " ".join(str(name).strip().split())
    return text or None

DEFAULT_BATCH_SIZE = 200_000


def _load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_mimic_drug_frequencies(
    exposures_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Count unique MIMIC drug names from exposures without scanning drug_pairs."""
    path = exposures_path or MIMIC_EXPOSURES_PATH
    if not path.exists():
        raise MissingInputError(
            f"MIMIC exposures not found at {path}. Run MIMIC prepare-mimic first."
        )

    pf = pq.ParquetFile(path)
    if "drug_name_norm" not in pf.schema.names:
        raise MissingInputError(
            f"Expected drug_name_norm in {path}. Columns: {pf.schema.names}"
        )

    columns = ["drug_name_norm"]
    if "drug_name_raw" in pf.schema.names:
        columns.append("drug_name_raw")

    counts: Counter[str] = Counter()
    raw_example: dict[str, str] = {}
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        frame = frame[frame["drug_name_norm"].notna()]
        frame["drug_name_norm"] = frame["drug_name_norm"].astype(str).str.strip()
        frame = frame[frame["drug_name_norm"] != ""]
        batch_counts = frame["drug_name_norm"].value_counts()
        for name, freq in batch_counts.items():
            counts[name] += int(freq)
            if name not in raw_example and "drug_name_raw" in frame.columns:
                match = frame.loc[frame["drug_name_norm"] == name, "drug_name_raw"]
                if len(match):
                    raw_example[name] = str(match.iloc[0])

    rows = [
        {
            "mimic_drug_name_norm": name,
            "mimic_drug_name_raw": raw_example.get(name),
            "exposure_count": int(counts[name]),
        }
        for name in sorted(counts.keys())
    ]
    return pd.DataFrame(rows)


def load_twosides_cid_catalog(drugs_path: Path | None = None) -> pd.DataFrame:
    path = drugs_path or TWOSIDES_DRUGS_PATH
    if not path.exists():
        raise MissingInputError(
            f"TWOSIDES drug catalog not found at {path}. Run: python -m src.data.pipeline twosides"
        )
    table = pd.read_parquet(path)
    if "drug_id" not in table.columns:
        raise MissingInputError(f"TWOSIDES catalog at {path} lacks drug_id column.")
    table = table.copy()
    table["drug_id"] = table["drug_id"].astype(str).str.strip()
    return table.drop_duplicates(subset=["drug_id"], keep="first")


def load_local_pubchem_cache(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load resolved CID lookups from local cache only (no API calls)."""
    root = cache_dir if cache_dir is not None else PUBCHEM_CACHE_DIR / "cid_lookup"
    success_path = root / "cid_success.jsonl"
    records = _load_jsonl_records(success_path)
    if not records:
        return pd.DataFrame(
            columns=[
                "mimic_drug_name",
                "pubchem_cid",
                "twosides_drug_id",
                "lookup_status",
                "cache_source",
            ]
        )
    rows = []
    for record in records:
        if record.get("lookup_status") != "resolved":
            continue
        cid = record.get("pubchem_cid")
        twosides_id = record.get("twosides_drug_id")
        if cid is None or twosides_id is None:
            continue
        rows.append(
            {
                "mimic_drug_name": normalize_pubchem_query(record.get("mimic_drug_name")),
                "mimic_drug_name_norm": normalize_pubchem_query(record.get("mimic_drug_name")),
                "pubchem_cid": int(cid),
                "twosides_drug_id": str(twosides_id).strip(),
                "lookup_status": record.get("lookup_status"),
                "cache_source": str(success_path),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["mimic_drug_name_norm"] = frame["mimic_drug_name_norm"].str.lower()
        frame = frame.drop_duplicates(subset=["mimic_drug_name_norm"], keep="first")
    return frame


def prepare_mimic_pubchem_mapping_candidates(
    exposures_path: Path | None = None,
    twosides_drugs_path: Path | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Build candidate mappings supported only by local data."""
    mimic_drugs = extract_mimic_drug_frequencies(exposures_path=exposures_path, batch_size=batch_size)
    twosides = load_twosides_cid_catalog(drugs_path=twosides_drugs_path)
    twosides_ids = set(twosides["drug_id"].astype(str))
    cache = load_local_pubchem_cache(cache_dir=cache_dir)
    cache_by_name = (
        cache.set_index("mimic_drug_name_norm").to_dict("index") if len(cache) else {}
    )

    rows = []
    matched_exact = 0
    matched_cache = 0
    cache_not_in_twosides = 0
    unresolved = 0

    for _, drug in mimic_drugs.iterrows():
        norm = drug["mimic_drug_name_norm"]
        norm_key = norm.lower()
        row = {
            "mimic_drug_name_norm": norm,
            "mimic_drug_name_raw": drug.get("mimic_drug_name_raw"),
            "exposure_count": int(drug["exposure_count"]),
            "twosides_drug_id": None,
            "pubchem_cid": None,
            "mapping_status": "unresolved",
            "mapping_method": None,
            "mapping_evidence": None,
        }

        if norm in twosides_ids:
            row.update(
                {
                    "twosides_drug_id": norm,
                    "pubchem_cid": int(norm.replace("CID", "")) if norm.startswith("CID") else None,
                    "mapping_status": "matched",
                    "mapping_method": "exact_cid_string",
                    "mapping_evidence": "mimic_drug_name_norm exactly equals TWOSIDES drug_id",
                }
            )
            matched_exact += 1
            rows.append(row)
            continue

        cached = cache_by_name.get(norm_key)
        if cached is not None:
            tw_id = cached["twosides_drug_id"]
            if tw_id in twosides_ids:
                row.update(
                    {
                        "twosides_drug_id": tw_id,
                        "pubchem_cid": int(cached["pubchem_cid"]),
                        "mapping_status": "matched",
                        "mapping_method": "local_pubchem_cache",
                        "mapping_evidence": cached["cache_source"],
                    }
                )
                matched_cache += 1
            else:
                row.update(
                    {
                        "pubchem_cid": int(cached["pubchem_cid"]),
                        "twosides_drug_id": tw_id,
                        "mapping_status": "unresolved",
                        "mapping_method": "local_pubchem_cache_not_in_twosides",
                        "mapping_evidence": (
                            f"Local cache maps to {tw_id}, which is absent from twosides_drugs.parquet"
                        ),
                    }
                )
                cache_not_in_twosides += 1
                unresolved += 1
            rows.append(row)
            continue

        unresolved += 1
        rows.append(row)

    table = pd.DataFrame(rows)
    matched_total = matched_exact + matched_cache
    stats = {
        "mode": "local_only",
        "api_calls": 0,
        "unique_mimic_drugs": int(len(mimic_drugs)),
        "twosides_drugs": int(len(twosides_ids)),
        "directly_matched_drugs": int(matched_total),
        "matched_exact_cid_string": int(matched_exact),
        "matched_local_pubchem_cache": int(matched_cache),
        "unresolved_drugs": int(unresolved),
        "cache_hits_not_in_twosides": int(cache_not_in_twosides),
        "local_pubchem_cache_entries_used": int(len(cache)),
        "local_pubchem_cache_path": str(
            (cache_dir or PUBCHEM_CACHE_DIR / "cid_lookup") / "cid_success.jsonl"
        ),
        "mimic_exposures_path": str(exposures_path or MIMIC_EXPOSURES_PATH),
        "twosides_drugs_path": str(twosides_drugs_path or TWOSIDES_DRUGS_PATH),
        "total_exposure_rows_summed": int(mimic_drugs["exposure_count"].sum()),
        "note": (
            "Local-only stage. No PubChem API calls. Mappings are created only for "
            "exact MIMIC/TWOSIDES CID string matches or prior on-disk PubChem cache entries "
            "whose twosides_drug_id exists in twosides_drugs.parquet. No fuzzy matching."
        ),
    }

    out = output_path if output_path is not None else MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH
    write_table(table, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "pubchem_candidate_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return table, stats
