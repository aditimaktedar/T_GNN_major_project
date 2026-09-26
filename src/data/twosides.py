"""Load the approved TWOSIDES source from local files.

Approved source: https://github.com/jcsun-00/Twosides

Verified local files (2026-09-06):
- ddis.csv: interaction records with PubChem CID drug identifiers
- drug_smiles.csv: TWOSIDES drug catalog with SMILES
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import (
    DEFAULT_TWOSIDES_CHUNKSIZE,
    METRICS_DIR,
    TWOSIDES_DIR,
    TWOSIDES_DRUGS_PATH,
    TWOSIDES_INTERACTIONS_PATH,
    TWOSIDES_PAIRS_PATH,
    TWOSIDES_RAW_DDIS,
    TWOSIDES_RAW_DRUG_SMILES,
)
from src.data.drug_pairs import canonicalize_pair
from src.data.exceptions import MissingInputError, SchemaError
from src.data.io_utils import (
    iter_tabular_files,
    read_header,
    resolve_column,
    write_json,
    write_table,
)

PAIR_A_ALIASES = ("d1", "drug_1", "drug1", "drug_a")
PAIR_B_ALIASES = ("d2", "drug_2", "drug2", "drug_b")
TYPE_ALIASES = ("type", "interaction_type", "effect", "condition")
NEG_SAMPLE_ALIASES = ("neg samples", "neg_sample", "neg_samples", "negative_sample")
DRUG_ID_ALIASES = ("drug_id", "drug", "id", "cid")
SMILES_ALIASES = ("smiles", "canonical_smiles", "smile")
PREFERRED_DDIS_NAMES = ("ddis.csv", "ddis.tsv", "ddis.csv.gz")
PREFERRED_DRUG_NAMES = ("drug_smiles.csv", "drug_smiles.tsv", "drug_smiles.csv.gz")

CID_PATTERN = re.compile(r"^CID\d{9}$")


def _validate_cid(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    return bool(CID_PATTERN.match(text))


def validate_ddis_columns(columns: list[str]) -> dict[str, str]:
    """Validate ddis.csv headers and return resolved column map."""
    drug_a = resolve_column(columns, PAIR_A_ALIASES)
    drug_b = resolve_column(columns, PAIR_B_ALIASES)
    interaction = resolve_column(columns, TYPE_ALIASES)
    missing = []
    if drug_a is None:
        missing.append(f"drug-a column (expected one of {PAIR_A_ALIASES})")
    if drug_b is None:
        missing.append(f"drug-b column (expected one of {PAIR_B_ALIASES})")
    if interaction is None:
        missing.append(f"interaction-type column (expected one of {TYPE_ALIASES})")
    if missing:
        raise SchemaError(
            "TWOSIDES ddis table is missing required columns: "
            + "; ".join(missing)
            + f". Actual columns: {columns}"
        )
    resolved = {"drug_a": drug_a, "drug_b": drug_b, "interaction_type": interaction}
    neg_col = resolve_column(columns, NEG_SAMPLE_ALIASES)
    if neg_col is not None:
        resolved["neg_sample_drug"] = neg_col
    return resolved


def validate_drug_smiles_columns(columns: list[str]) -> dict[str, str]:
    drug_id = resolve_column(columns, DRUG_ID_ALIASES)
    smiles = resolve_column(columns, SMILES_ALIASES)
    missing = []
    if drug_id is None:
        missing.append(f"drug-id column (expected one of {DRUG_ID_ALIASES})")
    if smiles is None:
        missing.append(f"SMILES column (expected one of {SMILES_ALIASES})")
    if missing:
        raise SchemaError(
            "TWOSIDES drug_smiles table is missing required columns: "
            + "; ".join(missing)
            + f". Actual columns: {columns}"
        )
    return {"drug_id": drug_id, "smiles": smiles}


def discover_twosides_files(twosides_dir: Path | None = None) -> dict:
    root = twosides_dir if twosides_dir is not None else TWOSIDES_DIR
    if not root.exists():
        raise MissingInputError(
            "No TWOSIDES folder was found. Place the approved files from "
            "https://github.com/jcsun-00/Twosides under "
            f"{TWOSIDES_DIR} (decompress .7z to CSV first). Do not substitute another DDI dataset."
        )

    ddis_path = None
    drugs_path = None
    for path in iter_tabular_files(root):
        name = path.name.lower()
        if name in PREFERRED_DDIS_NAMES:
            ddis_path = path
        if name in PREFERRED_DRUG_NAMES:
            drugs_path = path

    if ddis_path is None:
        for path in iter_tabular_files(root):
            cols = read_header(path)
            if resolve_column(cols, PAIR_A_ALIASES) and resolve_column(cols, PAIR_B_ALIASES):
                ddis_path = path
                break

    if drugs_path is None:
        for path in iter_tabular_files(root):
            cols = read_header(path)
            if resolve_column(cols, DRUG_ID_ALIASES) and resolve_column(cols, SMILES_ALIASES):
                drugs_path = path
                break

    discovered = {"twosides_dir": str(root), "ddis_path": ddis_path, "drug_smiles_path": drugs_path}
    if ddis_path is not None:
        discovered["ddis_columns"] = read_header(ddis_path)
    if drugs_path is not None:
        discovered["drug_smiles_columns"] = read_header(drugs_path)
    return discovered


def load_twosides_drugs(
    twosides_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load and standardize drug_smiles.csv."""
    discovered = discover_twosides_files(twosides_dir)
    drugs_path = discovered.get("drug_smiles_path")
    if drugs_path is None:
        if TWOSIDES_RAW_DRUG_SMILES.exists():
            drugs_path = TWOSIDES_RAW_DRUG_SMILES
        else:
            raise MissingInputError(
                f"No TWOSIDES drug catalog found under {discovered['twosides_dir']}. "
                "Expected drug_smiles.csv with drug_id and smiles columns."
            )

    columns = read_header(drugs_path)
    resolved = validate_drug_smiles_columns(columns)
    raw = pd.read_csv(drugs_path)
    drug_id_col = resolved["drug_id"]
    smiles_col = resolved["smiles"]

    if raw[drug_id_col].isna().any():
        raise SchemaError(f"TWOSIDES drug_smiles has null values in {drug_id_col!r}.")
    if raw[smiles_col].isna().any():
        raise SchemaError(f"TWOSIDES drug_smiles has null values in {smiles_col!r}.")

    table = pd.DataFrame(
        {
            "drug_id": raw[drug_id_col].astype(str).str.strip(),
            "smiles": raw[smiles_col].astype(str).str.strip(),
            "source_file": str(drugs_path),
        }
    )
    table = table.drop_duplicates(subset=["drug_id"], keep="first")
    invalid_ids = table.loc[~table["drug_id"].map(_validate_cid), "drug_id"].tolist()
    cid_like = table["drug_id"].map(_validate_cid)
    stats = {
        "source_file": str(drugs_path),
        "actual_columns": columns,
        "drug_id_column": drug_id_col,
        "smiles_column": smiles_col,
        "n_rows_raw": int(len(raw)),
        "n_rows": int(len(table)),
        "n_unique_drugs": int(table["drug_id"].nunique()),
        "invalid_drug_id_count": int((~cid_like).sum()),
        "invalid_drug_id_examples": invalid_ids[:5],
        "identifier_scheme": (
            "Zero-padded PubChem Compound ID strings with CID prefix and 9 digits "
            "(example: CID000002173). Verified from drug_smiles.csv and ddis.csv."
        ),
    }
    out = output_path if output_path is not None else TWOSIDES_DRUGS_PATH
    write_table(table, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "twosides_drugs_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return table, stats


def _standardize_ddis_chunk(
    chunk: pd.DataFrame,
    resolved: dict[str, str],
    source_file: str,
) -> tuple[pd.DataFrame, int]:
    drug_a_col = resolved["drug_a"]
    drug_b_col = resolved["drug_b"]
    type_col = resolved["interaction_type"]
    neg_col = resolved.get("neg_sample_drug")

    rows = []
    skipped_self = 0
    neg_values = chunk[neg_col] if neg_col else [None] * len(chunk)
    for drug_a, drug_b, interaction, neg_sample in zip(
        chunk[drug_a_col], chunk[drug_b_col], chunk[type_col], neg_values
    ):
        pair = canonicalize_pair(drug_a, drug_b)
        if pair is None:
            skipped_self += 1
            continue
        item = {
            "drug_a": pair[0],
            "drug_b": pair[1],
            "pair_key": f"{pair[0]}||{pair[1]}",
            "interaction_type": int(interaction) if pd.notna(interaction) else None,
            "neg_sample_drug": str(neg_sample).strip() if neg_sample is not None and not pd.isna(neg_sample) else None,
            "source": "twosides",
            "source_file": source_file,
        }
        rows.append(item)
    return pd.DataFrame(rows), skipped_self


def load_twosides_interactions(
    twosides_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    chunksize: int = DEFAULT_TWOSIDES_CHUNKSIZE,
    streaming: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load and standardize ddis.csv interaction records."""
    discovered = discover_twosides_files(twosides_dir)
    ddis_path = discovered.get("ddis_path")
    if ddis_path is None:
        if TWOSIDES_RAW_DDIS.exists():
            ddis_path = TWOSIDES_RAW_DDIS
        else:
            archives = list(Path(discovered["twosides_dir"]).rglob("*.7z"))
            extra = " Found compressed .7z files; decompress them to CSV before loading." if archives else ""
            raise MissingInputError(
                "No TWOSIDES ddis table found. Expected ddis.csv with d1, d2, and type columns."
                f"{extra}"
            )

    columns = read_header(ddis_path)
    resolved = validate_ddis_columns(columns)
    out = output_path if output_path is not None else TWOSIDES_INTERACTIONS_PATH
    skipped_self = 0
    total_raw = 0
    writer: pq.ParquetWriter | None = None
    pair_keys: set[str] = set()
    drugs: set[str] = set()
    interaction_types: set[int] = set()
    n_interactions = 0

    if streaming and str(ddis_path).lower().endswith(".csv"):
        for chunk in pd.read_csv(ddis_path, chunksize=chunksize):
            total_raw += len(chunk)
            standardized, skipped = _standardize_ddis_chunk(chunk, resolved, str(ddis_path))
            skipped_self += skipped
            if len(standardized):
                n_interactions += len(standardized)
                pair_keys.update(standardized["pair_key"].astype(str).tolist())
                drugs.update(standardized["drug_a"].astype(str).tolist())
                drugs.update(standardized["drug_b"].astype(str).tolist())
                interaction_types.update(standardized["interaction_type"].dropna().astype(int).tolist())
                table = pa.Table.from_pandas(standardized, preserve_index=False)
                if writer is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
        if writer is not None:
            writer.close()
        elif not out.exists():
            write_table(pd.DataFrame(), out)
        unique_pairs = pd.DataFrame(
            [{"pair_key": key, "drug_a": key.split("||")[0], "drug_b": key.split("||")[1]} for key in sorted(pair_keys)]
        ) if pair_keys else pd.DataFrame(columns=["drug_a", "drug_b", "pair_key"])
        interactions = unique_pairs
    else:
        raw = pd.read_csv(ddis_path)
        total_raw = len(raw)
        interactions, skipped_self = _standardize_ddis_chunk(raw, resolved, str(ddis_path))
        if len(interactions):
            interactions = interactions.drop_duplicates(
                subset=["drug_a", "drug_b", "interaction_type"]
            ).reset_index(drop=True)
        write_table(interactions, out)
        pair_keys = set(interactions["pair_key"].astype(str)) if len(interactions) else set()
        drugs = set(interactions["drug_a"]).union(set(interactions["drug_b"])) if len(interactions) else set()
        interaction_types = set(interactions["interaction_type"].dropna().astype(int)) if len(interactions) else set()
        n_interactions = len(interactions)
        unique_pairs = (
            interactions[["drug_a", "drug_b", "pair_key"]].drop_duplicates(subset=["pair_key"])
            if len(interactions)
            else pd.DataFrame(columns=["drug_a", "drug_b", "pair_key"])
        )

    pairs_out = TWOSIDES_PAIRS_PATH
    write_table(unique_pairs, pairs_out)

    stats = {
        "approved_source": "https://github.com/jcsun-00/Twosides",
        "source_file": str(ddis_path),
        "actual_columns": columns,
        "resolved_columns": resolved,
        "n_rows_raw": int(total_raw),
        "n_interaction_records": int(n_interactions),
        "n_unique_pairs": int(len(pair_keys)),
        "n_unique_drugs": int(len(drugs)),
        "n_interaction_types": int(len(interaction_types)),
        "interaction_type_min": int(min(interaction_types)) if interaction_types else None,
        "interaction_type_max": int(max(interaction_types)) if interaction_types else None,
        "skipped_self_or_empty_pairs": int(skipped_self),
        "identifier_scheme": (
            "Zero-padded PubChem Compound ID strings with CID prefix and 9 digits "
            "(example: CID000002173). Each ddis.csv row is one unordered drug pair plus one "
            "interaction_type code (0–962 observed locally) and an associated neg_sample_drug CID."
        ),
        "pair_representation": (
            "ddis.csv stores one row per (drug_a, drug_b, interaction_type) triple. "
            "The same unordered drug pair may appear on multiple rows with different interaction_type values."
        ),
        "unique_pairs_output": str(pairs_out),
        "interactions_output": str(out),
        "mimic_alignment_status": (
            "blocked: MIMIC drug pairs currently use hospital formulary drug_name_norm strings; "
            "TWOSIDES uses PubChem CID identifiers. No RxNorm or other crosswalk has been applied."
        ),
    }
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "twosides_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return interactions, stats


def load_twosides(
    twosides_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    chunksize: int = DEFAULT_TWOSIDES_CHUNKSIZE,
    streaming: bool = True,
    load_drugs: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load TWOSIDES drugs and interactions from local approved files."""
    drug_stats = None
    drug_catalog: set[str] = set()
    if load_drugs:
        drugs, drug_stats = load_twosides_drugs(twosides_dir=twosides_dir)
        drug_catalog = set(drugs["drug_id"].astype(str))

    interactions, stats = load_twosides_interactions(
        twosides_dir=twosides_dir,
        output_path=output_path or TWOSIDES_INTERACTIONS_PATH,
        stats_path=stats_path,
        chunksize=chunksize,
        streaming=streaming,
    )

    if drug_catalog and stats.get("n_interaction_records", 0) > 0:
        interactions_path = output_path or TWOSIDES_INTERACTIONS_PATH
        involved: set[str] = set()
        if interactions_path.exists():
            pf = pq.ParquetFile(interactions_path)
            for batch in pf.iter_batches(batch_size=chunksize):
                chunk = batch.to_pandas()
                involved.update(chunk["drug_a"].astype(str))
                involved.update(chunk["drug_b"].astype(str))
                if "neg_sample_drug" in chunk.columns:
                    involved.update(chunk["neg_sample_drug"].dropna().astype(str).tolist())
        missing = sorted(involved - drug_catalog)
        stats["drugs_missing_from_catalog"] = missing[:10]
        stats["drugs_missing_from_catalog_count"] = int(len(missing))
        if missing:
            raise SchemaError(
                f"TWOSIDES ddis references {len(missing)} drug IDs absent from drug_smiles.csv. "
                f"Examples: {missing[:5]}"
            )

    if drug_stats is not None:
        stats["drug_catalog"] = drug_stats
    stats["note"] = "No interaction labels were invented. Values are copied from ddis.csv."
    return interactions, stats
