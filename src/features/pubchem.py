"""PubChem PUG REST lookups with local caching.

Approved API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
CIDs and SMILES are never invented. Failed or ambiguous lookups stay unresolved.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
import requests

from src.data.config import (
    METRICS_DIR,
    MIMIC_NORMALIZED_PATH,
    MIMIC_PUBCHEM_MAPPING_PATH,
    PUBCHEM_CACHE_DIR,
    PUBCHEM_MAPPING_PATH,
)
from src.data.exceptions import MissingInputError
from src.data.io_utils import ensure_parent, load_table_chunked, write_json, write_table

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_TIMEOUT = 30
DEFAULT_PAUSE_SECONDS = 0.25
DEFAULT_MAX_RETRIES = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_pubchem_query(name: object) -> str | None:
    """Conservative query normalization: trim and collapse whitespace only."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = " ".join(str(name).strip().split())
    return text or None


def format_twosides_cid(cid: int) -> str:
    """Format an integer PubChem CID to the TWOSIDES zero-padded identifier."""
    return f"CID{int(cid):09d}"


def _cache_paths(cache_dir: Path, prefix: str = "smiles") -> tuple[Path, Path]:
    return cache_dir / f"{prefix}_success.jsonl", cache_dir / f"{prefix}_failed.jsonl"


def _load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records[item["cache_key"]] = item
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def cache_key(query: str, query_type: str = "name") -> str:
    return f"{query_type}:{query.strip().lower()}"


def _default_fetch(url: str, timeout: int) -> requests.Response:
    return requests.get(url, timeout=timeout)


def lookup_cid_by_name(
    query: str,
    timeout: int = DEFAULT_TIMEOUT,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fetch_fn: Callable[[str, int], requests.Response] | None = None,
) -> dict:
    """Resolve a drug name to PubChem CID(s) via PUG REST. Ambiguous multi-CID results are unresolved."""
    fetcher = fetch_fn or _default_fetch
    normalized = normalize_pubchem_query(query)
    if not normalized:
        return {
            "query_used": query,
            "mimic_drug_name": query,
            "lookup_status": "invalid_query",
            "lookup_detail": "Empty or null drug name.",
            "pubchem_cid": None,
            "twosides_drug_id": None,
            "n_cids_returned": 0,
            "http_status": None,
            "url": None,
            "source": "pubchem_pug_rest",
            "timestamp_utc": _utc_now(),
        }

    url = f"{PUG_BASE}/compound/name/{quote(normalized)}/cids/JSON"
    last_error = None
    for attempt in range(max_retries):
        try:
            response = fetcher(url, timeout)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", pause_seconds * (attempt + 2)))
                time.sleep(retry_after)
                last_error = "HTTP 429 rate limited"
                continue
            if response.status_code == 404:
                return {
                    "query_used": normalized,
                    "mimic_drug_name": query,
                    "lookup_status": "not_found",
                    "lookup_detail": "PubChem returned HTTP 404 for the name query.",
                    "pubchem_cid": None,
                    "twosides_drug_id": None,
                    "n_cids_returned": 0,
                    "http_status": 404,
                    "url": url,
                    "source": "pubchem_pug_rest",
                    "timestamp_utc": _utc_now(),
                }
            response.raise_for_status()
            payload = response.json()
            cids = payload.get("IdentifierList", {}).get("CID", []) or []
            n_cids = len(cids)
            if n_cids == 0:
                return {
                    "query_used": normalized,
                    "mimic_drug_name": query,
                    "lookup_status": "empty_result",
                    "lookup_detail": "PubChem returned no CIDs for the name query.",
                    "pubchem_cid": None,
                    "twosides_drug_id": None,
                    "n_cids_returned": 0,
                    "http_status": response.status_code,
                    "url": url,
                    "source": "pubchem_pug_rest",
                    "timestamp_utc": _utc_now(),
                }
            if n_cids > 1:
                return {
                    "query_used": normalized,
                    "mimic_drug_name": query,
                    "lookup_status": "ambiguous",
                    "lookup_detail": (
                        f"PubChem returned {n_cids} CIDs for the name query. "
                        "Ambiguous name matches are not mapped automatically."
                    ),
                    "pubchem_cid": None,
                    "twosides_drug_id": None,
                    "n_cids_returned": n_cids,
                    "candidate_cids": cids[:10],
                    "http_status": response.status_code,
                    "url": url,
                    "source": "pubchem_pug_rest",
                    "timestamp_utc": _utc_now(),
                }
            cid = int(cids[0])
            return {
                "query_used": normalized,
                "mimic_drug_name": query,
                "lookup_status": "resolved",
                "lookup_detail": "Single unambiguous PubChem CID returned for the name query.",
                "pubchem_cid": cid,
                "twosides_drug_id": format_twosides_cid(cid),
                "n_cids_returned": 1,
                "http_status": response.status_code,
                "url": url,
                "source": "pubchem_pug_rest",
                "timestamp_utc": _utc_now(),
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(pause_seconds * (attempt + 1))
    return {
        "query_used": normalized,
        "mimic_drug_name": query,
        "lookup_status": "error",
        "lookup_detail": last_error or "Unknown request error",
        "pubchem_cid": None,
        "twosides_drug_id": None,
        "n_cids_returned": 0,
        "http_status": None,
        "url": url,
        "source": "pubchem_pug_rest",
        "timestamp_utc": _utc_now(),
    }


def extract_unique_mimic_drug_names(
    input_path: Path | None = None,
    name_column: str = "drug_name_norm",
) -> pd.DataFrame:
    """Load unique MIMIC drug names from the normalized prescription table."""
    path = input_path or MIMIC_NORMALIZED_PATH
    if not path.exists():
        raise MissingInputError(
            f"MIMIC PubChem mapping requires normalized prescriptions at {path}. "
            "Run MIMIC ingest/clean/normalize first."
        )
    pf = pq.ParquetFile(path)
    if name_column not in pf.schema.names:
        raise MissingInputError(
            f"Column {name_column!r} not found in {path}. Available: {pf.schema.names}"
        )

    columns = [name_column]
    if "drug_name_raw" in pf.schema.names:
        columns.append("drug_name_raw")

    seen: set[str] = set()
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=200_000, columns=columns):
        frame = batch.to_pandas()
        for _, row in frame.iterrows():
            norm = normalize_pubchem_query(row[name_column])
            if norm is None:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "mimic_drug_name": norm,
                    "mimic_drug_name_raw": normalize_pubchem_query(row.get("drug_name_raw")),
                }
            )
    return pd.DataFrame(rows)


def map_mimic_drugs_to_pubchem(
    input_path: Path | None = None,
    name_column: str = "drug_name_norm",
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    fetch_fn: Callable[[str, int], requests.Response] | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map unique MIMIC drug names to PubChem CIDs with persistent cache and provenance."""
    drugs = extract_unique_mimic_drug_names(input_path=input_path, name_column=name_column)
    if limit is not None:
        drugs = drugs.head(limit).copy()

    cache_root = cache_dir if cache_dir is not None else PUBCHEM_CACHE_DIR / "cid_lookup"
    success_path, failed_path = _cache_paths(cache_root, prefix="cid")
    success_cache = _load_jsonl(success_path)
    failed_cache = _load_jsonl(failed_path)

    results = []
    n_cache_hits = 0
    n_api_requests = 0
    n_errors = 0

    for _, row in drugs.iterrows():
        mimic_name = row["mimic_drug_name"]
        key = cache_key(mimic_name, "cid")
        if key in success_cache:
            record = success_cache[key].copy()
            record["from_cache"] = True
            results.append(record)
            n_cache_hits += 1
            continue
        if key in failed_cache:
            record = failed_cache[key].copy()
            record["from_cache"] = True
            results.append(record)
            n_cache_hits += 1
            continue

        record = lookup_cid_by_name(
            mimic_name,
            fetch_fn=fetch_fn,
            pause_seconds=pause_seconds,
        )
        record["mimic_drug_name_raw"] = row.get("mimic_drug_name_raw")
        record["cache_key"] = key
        record["from_cache"] = False
        if record["lookup_status"] == "resolved":
            _append_jsonl(success_path, record)
        else:
            _append_jsonl(failed_path, record)
            if record["lookup_status"] == "error":
                n_errors += 1
        results.append(record)
        n_api_requests += 1
        if fetch_fn is None and pause_seconds:
            time.sleep(pause_seconds)

    table = pd.DataFrame(results)
    resolved = int((table["lookup_status"] == "resolved").sum()) if len(table) else 0
    total = int(len(table))
    unresolved = total - resolved
    stats = {
        "total_unique_mimic_drug_names": total,
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "resolution_percentage": round((resolved / total) * 100, 4) if total else 0.0,
        "cached_lookups": int(n_cache_hits),
        "api_requests": int(n_api_requests),
        "error_count": int(n_errors),
        "status_breakdown": table["lookup_status"].value_counts().astype(int).to_dict() if len(table) else {},
        "input_path": str(input_path or MIMIC_NORMALIZED_PATH),
        "name_column": name_column,
        "api": PUG_BASE,
        "cache_dir": str(cache_root),
        "note": (
            "Mappings are API-derived only. Ambiguous multi-CID PubChem name matches and "
            "lookup failures remain unresolved. No RxNorm IDs were substituted."
        ),
    }
    out = output_path if output_path is not None else MIMIC_PUBCHEM_MAPPING_PATH
    write_table(table, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "pubchem_mapping_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return table, stats


def lookup_pubchem(
    query: str,
    query_type: str = "name",
    timeout: int = DEFAULT_TIMEOUT,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fetch_fn: Callable[[str, int], requests.Response] | None = None,
) -> dict:
    """Query PubChem for Canonical SMILES. Does not guess a structure on failure."""
    fetcher = fetch_fn or _default_fetch
    if query_type == "cid":
        url = f"{PUG_BASE}/compound/cid/{quote(str(query))}/property/CanonicalSMILES,IsomericSMILES,Title/JSON"
    else:
        url = f"{PUG_BASE}/compound/name/{quote(str(query))}/property/CanonicalSMILES,IsomericSMILES,Title/JSON"

    last_error = None
    for attempt in range(max_retries):
        try:
            response = fetcher(url, timeout)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", pause_seconds * (attempt + 2)))
                time.sleep(retry_after)
                last_error = f"HTTP 429 rate limited"
                continue
            if response.status_code == 404:
                return {
                    "query": query,
                    "query_type": query_type,
                    "url": url,
                    "status": "not_found",
                    "http_status": 404,
                    "cid": None,
                    "smiles": None,
                    "title": None,
                }
            response.raise_for_status()
            payload = response.json()
            properties = payload.get("PropertyTable", {}).get("Properties", [])
            if not properties:
                return {
                    "query": query,
                    "query_type": query_type,
                    "url": url,
                    "status": "empty_properties",
                    "http_status": response.status_code,
                    "cid": None,
                    "smiles": None,
                    "title": None,
                }
            first = properties[0]
            smiles = first.get("CanonicalSMILES") or first.get("IsomericSMILES")
            return {
                "query": query,
                "query_type": query_type,
                "url": url,
                "status": "ok" if smiles else "missing_smiles",
                "http_status": response.status_code,
                "cid": first.get("CID"),
                "smiles": smiles,
                "title": first.get("Title"),
                "source": "pubchem_pug_rest",
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(pause_seconds * (attempt + 1))
    return {
        "query": query,
        "query_type": query_type,
        "url": url,
        "status": "error",
        "http_status": None,
        "cid": None,
        "smiles": None,
        "title": None,
        "error": last_error,
    }


def retrieve_smiles(
    queries: list[str] | pd.DataFrame | Path,
    query_column: str = "query",
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    fetch_fn: Callable[[str, int], requests.Response] | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
) -> tuple[pd.DataFrame, dict]:
    if isinstance(queries, Path):
        if not queries.exists():
            raise MissingInputError(f"PubChem step needs a query table at {queries}.")
        frame = load_table_chunked(queries)
        values = frame[query_column].dropna().astype(str).tolist()
    elif isinstance(queries, pd.DataFrame):
        if query_column in queries.columns:
            values = queries[query_column].dropna().astype(str).tolist()
        elif "rxnorm_id" in queries.columns:
            values = queries["rxnorm_id"].dropna().astype(str).tolist()
        elif "drug_name_norm" in queries.columns:
            values = queries["drug_name_norm"].dropna().astype(str).tolist()
        else:
            raise MissingInputError(
                f"PubChem queries need column {query_column!r} or rxnorm_id/drug_name_norm."
            )
    else:
        values = [str(item) for item in queries]

    unique_queries = []
    seen = set()
    for value in values:
        key = value.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        unique_queries.append(key)

    cache_root = cache_dir if cache_dir is not None else PUBCHEM_CACHE_DIR
    success_path, failed_path = _cache_paths(cache_root, prefix="smiles")
    success_cache = _load_jsonl(success_path)
    failed_cache = _load_jsonl(failed_path)

    results = []
    n_cache_hits = 0
    n_fetched = 0
    for query in unique_queries:
        key = cache_key(query, "name")
        if key in success_cache:
            results.append(success_cache[key])
            n_cache_hits += 1
            continue
        if key in failed_cache:
            results.append(failed_cache[key])
            n_cache_hits += 1
            continue
        record = lookup_pubchem(query, fetch_fn=fetch_fn, pause_seconds=pause_seconds)
        record["cache_key"] = key
        if record.get("status") == "ok" and record.get("smiles"):
            _append_jsonl(success_path, record)
        else:
            _append_jsonl(failed_path, record)
        results.append(record)
        n_fetched += 1
        if fetch_fn is None and pause_seconds:
            time.sleep(pause_seconds)

    table = pd.DataFrame(results)
    stats = {
        "n_queries": int(len(unique_queries)),
        "n_cache_hits": int(n_cache_hits),
        "n_fetched": int(n_fetched),
        "n_success": int((table["status"] == "ok").sum()) if len(table) else 0,
        "n_failed": int((table["status"] != "ok").sum()) if len(table) else 0,
        "api": PUG_BASE,
        "note": "SMILES are stored only when PubChem returned them. Failures are cached and not retried until the failed cache is cleared.",
    }
    out = output_path if output_path is not None else PUBCHEM_MAPPING_PATH
    write_table(table, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "pubchem_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return table, stats
