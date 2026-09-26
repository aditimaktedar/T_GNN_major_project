"""DrugBank sanity-check framework for locally supplied exports only.

DrugBank is licensed/private. This module never downloads DrugBank or stores
credentials. Schema is discovered from the supplied local file headers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import DRUGBANK_DIR
from src.data.io_utils import iter_tabular_files, load_table_chunked, read_header, resolve_column

DRUG_A_ALIASES = ("drug_a", "drug1", "drug_1", "d1")
DRUG_B_ALIASES = ("drug_b", "drug2", "drug_2", "d2")
MECHANISM_ALIASES = ("mechanism", "description", "interaction", "evidence")


def discover_drugbank_files(drugbank_dir: Path | None = None) -> list[dict]:
    root = drugbank_dir if drugbank_dir is not None else DRUGBANK_DIR
    if not root.exists():
        return []
    discovered = []
    for path in iter_tabular_files(root):
        columns = read_header(path)
        discovered.append(
            {
                "path": path,
                "columns": columns,
                "drug_a": resolve_column(columns, DRUG_A_ALIASES),
                "drug_b": resolve_column(columns, DRUG_B_ALIASES),
                "mechanism": resolve_column(columns, MECHANISM_ALIASES),
            }
        )
    return discovered


def check_predictions_against_drugbank(
    predictions: pd.DataFrame,
    drugbank_dir: Path | None = None,
) -> dict:
    """Compare predicted pairs against a locally supplied DrugBank export."""
    root = drugbank_dir if drugbank_dir is not None else DRUGBANK_DIR
    discovered = discover_drugbank_files(root)
    pair_files = [item for item in discovered if item["drug_a"] and item["drug_b"]]
    if not pair_files:
        return {
            "status": "drugbank_data_not_available",
            "message": (
                f"No DrugBank tabular export found under {root}. "
                "Place a licensed local export there after inspection."
            ),
            "matches": [],
        }

    chosen = pair_files[0]
    drugbank = load_table_chunked(chosen["path"])
    a_col, b_col = chosen["drug_a"], chosen["drug_b"]
    mech_col = chosen["mechanism"]
    db_pairs = set()
    evidence = {}
    for _, row in drugbank.iterrows():
        a = str(row[a_col]).strip()
        b = str(row[b_col]).strip()
        key = tuple(sorted((a, b)))
        db_pairs.add(key)
        if mech_col:
            evidence[key] = row[mech_col]

    matches = []
    for _, row in predictions.iterrows():
        a = str(row.get("drug_a", "")).strip()
        b = str(row.get("drug_b", "")).strip()
        key = tuple(sorted((a, b)))
        matched = key in db_pairs
        item = {
            "prediction_pair": f"{a}||{b}",
            "matched": matched,
            "mechanism": evidence.get(key) if matched and mech_col else None,
            "provenance": str(chosen["path"]) if matched else None,
        }
        matches.append(item)

    return {
        "status": "ok",
        "source_file": str(chosen["path"]),
        "actual_columns": chosen["columns"],
        "n_predictions": int(len(predictions)),
        "n_matched": int(sum(1 for m in matches if m["matched"])),
        "matches": matches,
    }


def check_explanations_against_drugbank(
    explanations: pd.DataFrame,
    drugbank_path: Path | None = None,
    drugbank_dir: Path | None = None,
) -> dict:
    """Compare explanation drug pairs against a local DrugBank export."""
    pairs = pd.DataFrame(
        {
            "drug_a": explanations["drug_a"],
            "drug_b": explanations["drug_b"],
        }
    )
    root = drugbank_dir
    if drugbank_path is not None:
        path = Path(drugbank_path)
        if not path.exists():
            return {
                "status": "drugbank_data_not_available",
                "message": f"DrugBank path not found: {path}",
                "total_explanation_pairs": int(len(explanations)),
                "matches": [],
            }
        columns = read_header(path)
        discovered = [
            {
                "path": path,
                "columns": columns,
                "drug_a": resolve_column(columns, DRUG_A_ALIASES),
                "drug_b": resolve_column(columns, DRUG_B_ALIASES),
                "mechanism": resolve_column(columns, MECHANISM_ALIASES),
            }
        ]
        pair_files = [item for item in discovered if item["drug_a"] and item["drug_b"]]
        if not pair_files:
            return {
                "status": "drugbank_data_not_available",
                "message": f"DrugBank file at {path} lacks recognizable drug pair columns.",
                "actual_columns": columns,
                "total_explanation_pairs": int(len(explanations)),
                "matches": [],
            }
        chosen = pair_files[0]
        drugbank = load_table_chunked(chosen["path"])
        a_col, b_col = chosen["drug_a"], chosen["drug_b"]
        mech_col = chosen["mechanism"]
        db_pairs = set()
        evidence = {}
        for _, row in drugbank.iterrows():
            a = str(row[a_col]).strip()
            b = str(row[b_col]).strip()
            key = tuple(sorted((a, b)))
            db_pairs.add(key)
            if mech_col:
                evidence[key] = row[mech_col]
        matches = []
        for _, row in explanations.iterrows():
            a = str(row.get("drug_a", "")).strip()
            b = str(row.get("drug_b", "")).strip()
            key = tuple(sorted((a, b)))
            matched = key in db_pairs
            matches.append(
                {
                    "explanation_pair": f"{a}||{b}",
                    "matched": matched,
                    "mechanism": evidence.get(key) if matched and mech_col else None,
                    "provenance": str(chosen["path"]) if matched else None,
                }
            )
        n_matched = sum(1 for m in matches if m["matched"])
        total = len(matches)
        return {
            "status": "ok",
            "source_file": str(chosen["path"]),
            "actual_columns": chosen["columns"],
            "total_explanation_pairs": total,
            "matched_drugbank_pairs": n_matched,
            "unmatched_pairs": total - n_matched,
            "match_rate": (n_matched / total) if total else None,
            "matches": matches,
        }

    result = check_predictions_against_drugbank(pairs, drugbank_dir=root)
    if result["status"] != "ok":
        result["total_explanation_pairs"] = int(len(explanations))
        return result
    total = result["n_predictions"]
    n_matched = result["n_matched"]
    return {
        "status": "ok",
        "source_file": result.get("source_file"),
        "actual_columns": result.get("actual_columns"),
        "total_explanation_pairs": total,
        "matched_drugbank_pairs": n_matched,
        "unmatched_pairs": total - n_matched,
        "match_rate": (n_matched / total) if total else None,
        "matches": result["matches"],
    }
