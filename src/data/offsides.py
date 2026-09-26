"""Load local OFFSIDES tables from the approved source tree.

Approved source: https://github.com/tatonetti-lab/offsides

That GitHub repository is a FAERS processing pipeline and may not contain a
packaged OFFSIDES table. This loader never invents DDI labels from OFFSIDES.
Default pipeline role: adverse-event context only, not DDI positives/negatives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import METRICS_DIR, OFFSIDES_DIR, OFFSIDES_TABLE_PATH
from src.data.exceptions import MissingInputError
from src.data.io_utils import (
    iter_tabular_files,
    load_table_chunked,
    read_header,
    resolve_column,
    write_json,
    write_table,
)

DRUG_ALIASES = ("drug", "drug_name", "drug_concept_name", "medication")
DRUG_ID_ALIASES = ("drug_id", "drug_rxnorm_id", "drug_rxcui", "rxcui", "rxnorm_id")
EVENT_ALIASES = ("condition", "condition_concept_name", "adverse_event", "outcome", "event")
EVENT_ID_ALIASES = ("condition_meddra_id", "meddra_id", "outcome_id")

DEFAULT_ROLE = "adverse_event_context_not_ddi_label"


def discover_offsides_files(offsides_dir: Path | None = None) -> list[dict]:
    root = offsides_dir if offsides_dir is not None else OFFSIDES_DIR
    if not root.exists():
        raise MissingInputError(
            "No OFFSIDES folder was found. Place files from the approved source "
            "https://github.com/tatonetti-lab/offsides under "
            f"{OFFSIDES_DIR}. That repository may contain processing code rather than "
            "a packaged table; do not silently substitute nsides-release or another dataset."
        )
    discovered = []
    for path in iter_tabular_files(root):
        columns = read_header(path)
        discovered.append(
            {
                "path": path,
                "columns": columns,
                "drug_column": resolve_column(columns, DRUG_ALIASES),
                "drug_id_column": resolve_column(columns, DRUG_ID_ALIASES),
                "event_column": resolve_column(columns, EVENT_ALIASES),
                "event_id_column": resolve_column(columns, EVENT_ID_ALIASES),
            }
        )
    return discovered


def load_offsides(
    offsides_dir: Path | None = None,
    output_path: Path | None = None,
    stats_path: Path | None = None,
    role: str = DEFAULT_ROLE,
) -> tuple[pd.DataFrame, dict]:
    root = offsides_dir if offsides_dir is not None else OFFSIDES_DIR
    discovered = discover_offsides_files(root)
    usable = [
        item
        for item in discovered
        if item["drug_column"] or item["drug_id_column"]
    ]
    if not usable:
        raise MissingInputError(
            "No OFFSIDES tabular file with a drug identifier/name column was found under "
            f"{root}. The approved GitHub repo often has code/notebooks only, not a data "
            "release. Do not invent an OFFSIDES table. Approved source: "
            "https://github.com/tatonetti-lab/offsides"
        )

    chosen = usable[0]
    raw = load_table_chunked(chosen["path"])
    table = pd.DataFrame()
    if chosen["drug_id_column"]:
        table["drug_id_source"] = raw[chosen["drug_id_column"]]
    if chosen["drug_column"]:
        table["drug_name_source"] = raw[chosen["drug_column"]]
    if chosen["event_id_column"]:
        table["event_id_source"] = raw[chosen["event_id_column"]]
    if chosen["event_column"]:
        table["event_source"] = raw[chosen["event_column"]]
    table["source"] = "offsides"
    table["source_file"] = str(chosen["path"])
    table["pipeline_role"] = role

    drug_values = []
    if "drug_id_source" in table.columns:
        drug_values.extend(table["drug_id_source"].dropna().astype(str).tolist())
    if "drug_name_source" in table.columns:
        drug_values.extend(table["drug_name_source"].dropna().astype(str).tolist())

    stats = {
        "approved_source": "https://github.com/tatonetti-lab/offsides",
        "source_file": str(chosen["path"]),
        "actual_columns": chosen["columns"],
        "drug_column": chosen["drug_column"],
        "drug_id_column": chosen["drug_id_column"],
        "event_column": chosen["event_column"],
        "event_id_column": chosen["event_id_column"],
        "n_rows": int(len(table)),
        "n_unique_drugs": int(len(set(drug_values))),
        "pipeline_role": role,
        "ddi_label_policy": (
            "OFFSIDES is not used as a DDI positive/negative label source unless the "
            "team later changes pipeline_role. Absence from OFFSIDES is not a DDI negative."
        ),
        "identifier_mapping_requirement": (
            "OFFSIDES drug identifiers are preserved from the file. They are not rewritten "
            "to RxNorm unless a later approved mapping exists."
        ),
        "discovered_files": [
            {"path": str(item["path"]), "columns": item["columns"]} for item in discovered
        ],
    }
    out = output_path if output_path is not None else OFFSIDES_TABLE_PATH
    write_table(table, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "offsides_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return table, stats
