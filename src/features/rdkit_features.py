"""RDKit molecular fingerprints from validated SMILES only.

Missing or invalid SMILES do not receive invented fingerprints.
Default fingerprint: Morgan, radius=2, n_bits=2048 (documented, configurable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.data.config import (
    DEFAULT_FINGERPRINT_N_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_FINGERPRINT_TYPE,
    METRICS_DIR,
    MOLECULAR_FEATURES_PATH,
)
from src.data.exceptions import ConfigurationError, MissingInputError
from src.data.io_utils import load_table_chunked, write_json, write_table


def parse_smiles(smiles: object):
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    return Chem.MolFromSmiles(text)


def morgan_fingerprint(
    mol,
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_N_BITS,
) -> list[int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bitvect = generator.GetFingerprint(mol)
    return [int(bit) for bit in bitvect]


def featurize_smiles(
    chemicals: pd.DataFrame | Path,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    fingerprint_type: str = DEFAULT_FINGERPRINT_TYPE,
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_N_BITS,
    output_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if fingerprint_type != "morgan":
        raise ConfigurationError(
            f"Unsupported fingerprint_type {fingerprint_type!r}. Implemented: 'morgan'."
        )
    if isinstance(chemicals, Path):
        if not chemicals.exists():
            raise MissingInputError(f"Feature step needs a chemical table at {chemicals}.")
        frame = load_table_chunked(chemicals)
    else:
        frame = chemicals.copy()
    if smiles_column not in frame.columns:
        raise MissingInputError(
            f"SMILES column {smiles_column!r} is missing. Available: {list(frame.columns)}"
        )

    if id_column is None:
        for candidate in ("query", "rxnorm_id", "cid", "drug_name_norm", "drug_id"):
            if candidate in frame.columns:
                id_column = candidate
                break
    if id_column is None:
        frame = frame.copy()
        frame["_row_id"] = np.arange(len(frame))
        id_column = "_row_id"

    rows = []
    n_valid = 0
    n_invalid = 0
    n_missing = 0
    for _, record in frame.iterrows():
        drug_id = record[id_column]
        smiles = record[smiles_column]
        if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)) or str(smiles).strip() == "":
            n_missing += 1
            rows.append(
                {
                    "drug_id": drug_id,
                    "smiles": None,
                    "smiles_valid": False,
                    "fingerprint": None,
                    "skip_reason": "missing_smiles",
                }
            )
            continue
        mol = parse_smiles(smiles)
        if mol is None:
            n_invalid += 1
            rows.append(
                {
                    "drug_id": drug_id,
                    "smiles": str(smiles),
                    "smiles_valid": False,
                    "fingerprint": None,
                    "skip_reason": "invalid_smiles",
                }
            )
            continue
        n_valid += 1
        rows.append(
            {
                "drug_id": drug_id,
                "smiles": str(smiles),
                "smiles_valid": True,
                "fingerprint": morgan_fingerprint(mol, radius=radius, n_bits=n_bits),
                "skip_reason": None,
            }
        )

    features = pd.DataFrame(rows)
    stats = {
        "total_drugs": int(len(features)),
        "valid_smiles": int(n_valid),
        "invalid_smiles": int(n_invalid),
        "missing_smiles": int(n_missing),
        "feature_dimension": int(n_bits) if n_valid else None,
        "fingerprint_type": fingerprint_type,
        "radius": radius,
        "n_bits": n_bits,
        "include_chirality": False,
        "note": "No fingerprint was generated for missing or invalid SMILES.",
    }
    out = output_path if output_path is not None else MOLECULAR_FEATURES_PATH
    write_table(features, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "molecular_feature_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return features, stats
