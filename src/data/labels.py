"""Join MIMIC drug pairs with an approved DDI source.

Default policy: a pair present in TWOSIDES is `positive`. A MIMIC pair absent
from TWOSIDES is `unknown`, NOT a true negative.

negative_mode:
- none: unmatched pairs stay unknown (default; does not claim they are non-interacting)
- random_unmatched: sample unmatched MIMIC pairs as `negative` with an explicit
  documented assumption. This is NOT justified as gold-standard true negatives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import DRUG_PAIRS_PATH, LABELED_PAIRS_PATH, METRICS_DIR, TWOSIDES_PAIRS_PATH
from src.data.exceptions import ConfigurationError, MissingInputError
from src.data.io_utils import load_table_chunked, write_json, write_table

VALID_NEGATIVE_MODES = ("none", "random_unmatched")


def integrate_labels(
    mimic_pairs: pd.DataFrame | Path,
    ddi_pairs: pd.DataFrame | Path,
    negative_mode: str = "none",
    negative_fraction: float = 1.0,
    seed: int = 42,
    output_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if negative_mode not in VALID_NEGATIVE_MODES:
        raise ConfigurationError(
            f"Unknown negative_mode {negative_mode!r}. Valid: {VALID_NEGATIVE_MODES}"
        )
    if isinstance(mimic_pairs, Path):
        if not mimic_pairs.exists():
            raise MissingInputError(f"Label step needs MIMIC pairs at {mimic_pairs}.")
        mimic = load_table_chunked(mimic_pairs)
    else:
        mimic = mimic_pairs.copy()
    if isinstance(ddi_pairs, Path):
        if not ddi_pairs.exists():
            raise MissingInputError(
                f"Label step needs an approved DDI pair table at {ddi_pairs}. "
                "Run TWOSIDES ingestion first. Do not invent labels."
            )
        ddi = load_table_chunked(ddi_pairs)
    else:
        ddi = ddi_pairs.copy()

    if "pair_key" not in mimic.columns or "pair_key" not in ddi.columns:
        raise ConfigurationError("Both pair tables must contain pair_key.")

    ddi_keys = set(ddi["pair_key"].astype(str))
    labeled = mimic.copy()
    labeled["pair_key"] = labeled["pair_key"].astype(str)
    labeled["label_source"] = "twosides"
    labeled["in_ddi_source"] = labeled["pair_key"].isin(ddi_keys)
    labeled["label"] = "unknown"
    labeled.loc[labeled["in_ddi_source"], "label"] = "positive"

    unmatched = labeled.loc[~labeled["in_ddi_source"]].copy()
    if negative_mode == "random_unmatched" and len(unmatched):
        n_neg = max(1, int(round(len(unmatched) * negative_fraction))) if negative_fraction < 1 else len(unmatched)
        n_neg = min(n_neg, len(unmatched))
        sampled_index = unmatched.sample(n=n_neg, random_state=seed).index
        labeled.loc[sampled_index, "label"] = "negative"
        assumption = (
            "random_unmatched treats sampled MIMIC pairs that are absent from TWOSIDES as "
            "negatives. This is a modeling assumption, not a verified non-interaction."
        )
    else:
        assumption = (
            "Pairs absent from TWOSIDES are labeled unknown. They are not treated as true negatives."
        )

    positive_count = int((labeled["label"] == "positive").sum())
    negative_count = int((labeled["label"] == "negative").sum())
    unknown_count = int((labeled["label"] == "unknown").sum())
    defined = positive_count + negative_count
    stats = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "unknown_unmatched_count": unknown_count,
        "class_ratio_positive_over_defined": (positive_count / defined) if defined else None,
        "unique_patients": int(labeled["patient_id"].nunique()) if "patient_id" in labeled.columns else None,
        "unique_drugs": int(
            pd.unique(labeled[["drug_a", "drug_b"]].to_numpy().ravel()).size
        )
        if {"drug_a", "drug_b"}.issubset(labeled.columns)
        else None,
        "unique_pairs": int(labeled["pair_key"].nunique()),
        "negative_mode": negative_mode,
        "assumption": assumption,
        "ddi_source_unique_pairs": int(len(ddi_keys)),
        "note": "No labels were invented beyond the configured join against the approved DDI source.",
    }
    out = output_path if output_path is not None else LABELED_PAIRS_PATH
    write_table(labeled, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "label_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return labeled, stats
