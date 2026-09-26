"""Project-relative data and result paths.

Paths are resolved from the repository root, not from a machine-specific
absolute location. No credentials, usernames, API keys, or secrets belong here.
"""

from pathlib import Path

# src/data/config.py -> src/data -> src -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
RESULTS_DIR = PROJECT_ROOT / "results"

# Expected local folders once each approved source is obtained.
# These directories may not exist yet; creating them is a later data-setup step.
MIMIC_DIR = RAW_DATA_DIR / "mimic"
# Preferred MIMIC-IV 2.1 layout when downloaded from the approved Kaggle bundle.
MIMIC_IV_ROOT = MIMIC_DIR / "mimic-iv-2.1"
MIMIC_HOSP_PRESCRIPTIONS = MIMIC_IV_ROOT / "hosp" / "prescriptions.csv"
MIMIC_HOSP_PATIENTS = MIMIC_IV_ROOT / "hosp" / "patients.csv"
TWOSIDES_DIR = RAW_DATA_DIR / "twosides"
OFFSIDES_DIR = RAW_DATA_DIR / "offsides"
PHARMGKB_DIR = RAW_DATA_DIR / "pharmgkb"
DRUGBANK_DIR = RAW_DATA_DIR / "drugbank"

DATA_SEARCH_DIRS = (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR)

METRICS_DIR = RESULTS_DIR / "metrics"
MASTER_RESULTS_PATH = METRICS_DIR / "MASTER_RESULTS.json"
PUBCHEM_CACHE_DIR = INTERIM_DATA_DIR / "pubchem"
MIMIC_PUBCHEM_MAPPING_PATH = INTERIM_DATA_DIR / "mimic_pubchem_mapping.parquet"
MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH = INTERIM_DATA_DIR / "mimic_pubchem_mapping_candidates.parquet"
RXNORM_MAPPING_DIR = RAW_DATA_DIR / "rxnorm"

# Default pipeline outputs. These are created by pipeline stages; they are not raw data.
MIMIC_PRESCRIPTIONS_PATH = INTERIM_DATA_DIR / "mimic_prescriptions.parquet"
MIMIC_CLEAN_PATH = INTERIM_DATA_DIR / "mimic_prescriptions_clean.parquet"
MIMIC_EXPOSURES_PATH = INTERIM_DATA_DIR / "mimic_drug_exposures.parquet"
MIMIC_NORMALIZED_PATH = INTERIM_DATA_DIR / "mimic_prescriptions_normalized.parquet"
DRUG_PAIRS_PATH = INTERIM_DATA_DIR / "drug_pairs.parquet"

# Default chunk size for streaming MIMIC CSV/Parquet processing.
DEFAULT_MIMIC_CHUNKSIZE = 200_000
TWOSIDES_PAIRS_PATH = INTERIM_DATA_DIR / "twosides_unique_pairs.parquet"
TWOSIDES_INTERACTIONS_PATH = INTERIM_DATA_DIR / "twosides_interactions.parquet"
TWOSIDES_DRUGS_PATH = INTERIM_DATA_DIR / "twosides_drugs.parquet"
TWOSIDES_RAW_DDIS = TWOSIDES_DIR / "ddis.csv"
TWOSIDES_RAW_DRUG_SMILES = TWOSIDES_DIR / "drug_smiles.csv"
DEFAULT_TWOSIDES_CHUNKSIZE = 200_000
OFFSIDES_TABLE_PATH = INTERIM_DATA_DIR / "offsides_table.parquet"
LABELED_PAIRS_PATH = INTERIM_DATA_DIR / "labeled_pairs.parquet"
MIMIC_TWOSIDES_LABELED_PATH = INTERIM_DATA_DIR / "mimic_twosides_labeled_pairs.parquet"
MIMIC_TWOSIDES_ML_POSITIVES_PATH = INTERIM_DATA_DIR / "mimic_twosides_ml_positives.parquet"
MIMIC_TWOSIDES_ML_NEGATIVES_PATH = INTERIM_DATA_DIR / "mimic_twosides_ml_negatives.parquet"
MIMIC_TWOSIDES_ML_DATASET_PATH = INTERIM_DATA_DIR / "mimic_twosides_ml_dataset.parquet"
MIMIC_TWOSIDES_ML_SPLITS_PATH = INTERIM_DATA_DIR / "mimic_twosides_ml_patient_splits.parquet"
DEFAULT_NEGATIVE_RATIO = 1.0
PUBCHEM_MAPPING_PATH = PUBCHEM_CACHE_DIR / "chemical_mapping.parquet"
MOLECULAR_FEATURES_PATH = PROCESSED_DATA_DIR / "molecular_features.parquet"
PATIENT_SPLITS_PATH = PROCESSED_DATA_DIR / "patient_splits.csv"
ML_DATASET_PATH = PROCESSED_DATA_DIR / "ml_dataset.parquet"
PIPELINE_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

SELECTED_MIMIC_TWOSIDES_DATASET_PATH = DATA_DIR / "selected" / "final_mimic_twosides_dataset.csv"
FINAL_MIMIC_TWOSIDES_ML_PATH = PROCESSED_DATA_DIR / "final_mimic_twosides_ml.csv"
FINAL_MIMIC_TWOSIDES_ML_REPORT_PATH = METRICS_DIR / "final_mimic_twosides_ml_preprocessing_report.json"

# Frequency-filtered formulation: types with >= FREQUENT363_MIN_TRAIN_ROWS train rows.
FREQUENT363_MIN_TRAIN_ROWS = 20
FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH = (
    PROCESSED_DATA_DIR / "final_mimic_twosides_ml_frequent363.csv"
)
FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_REPORT_PATH = (
    METRICS_DIR / "final_mimic_twosides_ml_frequent363_report.json"
)

# Documented default split ratios (used only because the project did not specify others).
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SPLIT_SEED = 42

# Documented default Morgan fingerprint settings (not hidden).
DEFAULT_FINGERPRINT_TYPE = "morgan"
DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_N_BITS = 2048

# Multi-label preparation output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
MULTILABEL_FREQUENT363_DATASET_PATH = PROCESSED_DATA_DIR / "multilabel_frequent363_dataset.parquet"
MULTILABEL_FREQUENT363_DATASET_CSV = PROCESSED_DATA_DIR / "multilabel_frequent363_dataset.csv"
MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH = PROCESSED_DATA_DIR / "multilabel_frequent363_label_mapping.json"
MULTILABEL_PAIR_SPLITS_PATH = PROCESSED_DATA_DIR / "multilabel_pair_splits.json"
MULTILABEL_DATASET_AUDIT_PATH = METRICS_DIR / "multilabel_dataset_audit.json"
MULTILABEL_EXPERIMENT_RESULTS_CSV = REPORTS_DIR / "multilabel_experiment_results.csv"
MULTILABEL_EXPERIMENT_RESULTS_MD = REPORTS_DIR / "multilabel_experiment_results.md"

# Temporal Pair Features output paths
TEMPORAL_EMAR_EVENTS_PATH = PROCESSED_DATA_DIR / "temporal_emar_events.csv"
TEMPORAL_PAIR_FEATURES_CSV = PROCESSED_DATA_DIR / "temporal_pair_features.csv"
TEMPORAL_PAIR_FEATURES_REPORT_MD = REPORTS_DIR / "temporal_pair_features_validation.md"
TEMPORAL_PAIR_FEATURES_AUDIT_JSON = METRICS_DIR / "temporal_pair_features_validation.json"

# Temporal Multi-Label Dataset output paths
TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV = PROCESSED_DATA_DIR / "temporal_multilabel_frequent363_dataset.csv"
TEMPORAL_MULTILABEL_FREQUENT363_DATASET_PATH = PROCESSED_DATA_DIR / "temporal_multilabel_frequent363_dataset.parquet"
TEMPORAL_MULTILABEL_REPORT_MD = REPORTS_DIR / "temporal_multilabel_validation.md"
TEMPORAL_MULTILABEL_AUDIT_JSON = METRICS_DIR / "temporal_multilabel_validation.json"

# Temporal Multi-Label Baseline output paths
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
TEMPORAL_BASELINE_RESULTS_JSON = METRICS_DIR / "temporal_baseline_results.json"
TEMPORAL_BASELINE_REPORT_MD = REPORTS_DIR / "temporal_baseline_evaluation.md"
TEMPORAL_BASELINE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "temporal_baseline.pt"

# Temporal Feature Ablation output paths
TEMPORAL_ABLATION_RESULTS_JSON = METRICS_DIR / "temporal_feature_ablation_results.json"
TEMPORAL_ABLATION_REPORT_MD = REPORTS_DIR / "temporal_feature_ablation_report.md"

