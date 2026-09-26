import pandas as pd
from pathlib import Path

out = Path("data/selected")
out.mkdir(parents=True, exist_ok=True)

# MIMIC-IV
mimic_cols = [
    "subject_id",
    "hadm_id",
    "drug",
    "starttime",
    "stoptime",
    "drug_type",
    "formulary_drug_cd",
    "route",
    "dose_val_rx",
]

mimic = pd.read_csv(
    "data/raw/mimic/mimic-iv-2.1/hosp/prescriptions.csv",
    usecols=mimic_cols
)

mimic.to_parquet(out / "mimic_prescriptions_selected.parquet", index=False)

# TWOSIDES interactions
twosides_cols = ["d1", "d2", "type", "Neg samples"]

twosides = pd.read_csv(
    "data/raw/twosides/ddis.csv",
    usecols=twosides_cols
)

twosides.to_parquet(out / "twosides_interactions_selected.parquet", index=False)

# TWOSIDES molecular information
smiles_cols = ["drug_id", "smiles"]

smiles = pd.read_csv(
    "data/raw/twosides/drug_smiles.csv",
    usecols=smiles_cols
)

smiles.to_parquet(out / "twosides_drug_smiles_selected.parquet", index=False)

print("Done.")
print("MIMIC:", mimic.shape)
print("TWOSIDES interactions:", twosides.shape)
print("TWOSIDES SMILES:", smiles.shape)
