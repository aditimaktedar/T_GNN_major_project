"""PharmGKB / CPIC evidence attachment interface.

Approved source: https://www.clinpgx.org/downloads
No pharmacogenetic evidence is invented when files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import PHARMGKB_DIR
from src.data.io_utils import iter_tabular_files, load_table_chunked, read_header, resolve_column

DRUG_ALIASES = ("drug", "drug_name", "chemical", "medication")
GENE_ALIASES = ("gene", "gene_symbol", "gene_id")
EVIDENCE_ALIASES = ("evidence", "annotation", "guideline", "recommendation")


def discover_pharmgkb_files(pharmgkb_dir: Path | None = None) -> list[dict]:
    root = pharmgkb_dir if pharmgkb_dir is not None else PHARMGKB_DIR
    if not root.exists():
        return []
    discovered = []
    for path in iter_tabular_files(root):
        columns = read_header(path)
        discovered.append(
            {
                "path": path,
                "columns": columns,
                "drug": resolve_column(columns, DRUG_ALIASES),
                "gene": resolve_column(columns, GENE_ALIASES),
                "evidence": resolve_column(columns, EVIDENCE_ALIASES),
            }
        )
    return discovered


def attach_pharmacogenetic_evidence(
    pairs: pd.DataFrame,
    pharmgkb_dir: Path | None = None,
) -> dict:
    root = pharmgkb_dir if pharmgkb_dir is not None else PHARMGKB_DIR
    discovered = discover_pharmgkb_files(root)
    usable = [item for item in discovered if item["drug"]]
    if not usable:
        return {
            "status": "pharmgkb_data_not_available",
            "message": (
                f"No PharmGKB/CPIC tabular files found under {root}. "
                "Approved source: https://www.clinpgx.org/downloads"
            ),
            "records": [],
        }

    chosen = usable[0]
    table = load_table_chunked(chosen["path"])
    drug_col = chosen["drug"]
    gene_col = chosen["gene"]
    evidence_col = chosen["evidence"]
    lookup = {}
    for _, row in table.iterrows():
        drug = str(row[drug_col]).strip().lower()
        lookup.setdefault(drug, []).append(
            {
                "gene": row[gene_col] if gene_col else None,
                "evidence": row[evidence_col] if evidence_col else None,
                "source_file": str(chosen["path"]),
            }
        )

    records = []
    for _, row in pairs.iterrows():
        for drug_col_name in ("drug_a", "drug_b"):
            if drug_col_name not in row:
                continue
            drug = str(row[drug_col_name]).strip().lower()
            records.append(
                {
                    "pair_key": f"{row.get('drug_a')}||{row.get('drug_b')}",
                    "drug": drug,
                    "evidence": lookup.get(drug, []),
                    "provenance": str(chosen["path"]) if drug in lookup else None,
                }
            )
    return {
        "status": "ok",
        "source_file": str(chosen["path"]),
        "actual_columns": chosen["columns"],
        "records": records,
    }
