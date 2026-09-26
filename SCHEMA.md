# TemporalDDI-GNN Data Schema

Schemas below describe **intermediate outputs actually generated** from the local MIMIC-IV 2.1 copy. Raw MIMIC files remain under `data/raw/mimic/mimic-iv-2.1/` and are not modified.

## Raw MIMIC-IV 2.1 (inspected, read-only)

Primary medication source: `data/raw/mimic/mimic-iv-2.1/hosp/prescriptions.csv`

| Column | Type (observed) | Role |
| --- | --- | --- |
| `subject_id` | int64 | Patient identifier → canonical `patient_id` |
| `hadm_id` | int64 | Hospital admission → canonical `admission_id` |
| `pharmacy_id` | int64 | Pharmacy order identifier |
| `poe_id` | string | Provider order entry identifier |
| `poe_seq` | float64 | POE sequence |
| `starttime` | string/datetime | Prescription start → canonical `start_time` |
| `stoptime` | string/datetime | Prescription stop → canonical `end_time` |
| `drug_type` | string | e.g. MAIN |
| `drug` | string | Medication name → canonical `drug_name_raw` |
| `formulary_drug_cd` | string | Hospital formulary code |
| `gsn` | string | Generic sequence number |
| `ndc` | float64 | National Drug Code |
| `prod_strength` | string | Product strength text |
| `form_rx` | string | Prescription form (mostly null in sample) |
| `dose_val_rx` | string | Dose value → canonical `dose` |
| `dose_unit_rx` | string | Dose unit |
| `form_val_disp` | string | Dispense form value |
| `form_unit_disp` | string | Dispense form unit |
| `doses_per_24_hrs` | float64 | Dosing frequency |
| `route` | string | Administration route |

Other MIMIC-IV tables (`pharmacy.csv`, `emar.csv`, ICU tables) are present locally but **not ingested** for this stage. Ingestion uses `hosp/prescriptions.csv` only.

## `data/interim/mimic_prescriptions.parquet`

Standardized prescription table produced by `python -m src.data.pipeline ingest-mimic`.

| Column | Description |
| --- | --- |
| `patient_id` | From `subject_id` |
| `admission_id` | From `hadm_id` |
| `drug_name_raw` | From `drug` |
| `start_time` | Parsed from `starttime` |
| `end_time` | Parsed from `stoptime` |
| `ndc`, `gsn`, `formulary_drug_cd`, `drug_type`, `dose`, `route` | Copied when present in source |
| `source_file` | Provenance path of the ingested CSV |

Metadata sidecar: `data/interim/mimic_prescriptions_metadata.json`

## `data/interim/mimic_prescriptions_clean.parquet`

Cleaned prescriptions from `python -m src.data.pipeline clean`.

| Column | Description |
| --- | --- |
| All columns from standardized table | Retained when row passes validation |
| `drug_name_norm` | Lowercased, whitespace-normalized drug name |

Dropped rows (if any): `data/interim/mimic_prescriptions_dropped.parquet` with `_drop_reason`.

Cleaning statistics: `results/metrics/mimic_cleaning_statistics.json`

## `data/interim/mimic_prescriptions_normalized.parquet`

Prepared drug table from `python -m src.data.pipeline normalize` **without** a mapping file.

| Column | Description |
| --- | --- |
| All clean-table columns | Preserved |
| `rxnorm_id` | Null while mapping is pending |
| `rxnorm_mapped` | `false` |
| `rxnorm_status` | `pending` |

Statistics: `results/metrics/rxnorm_normalization_statistics.json`

## `data/interim/mimic_drug_exposures.parquet`

Unique `(patient_id, admission_id, drug)` exposures collapsed before pairing.

| Column | Description |
| --- | --- |
| `patient_id` | Patient identifier |
| `admission_id` | Admission identifier |
| `drug_name_norm` | Normalized drug name used as pair key |
| `drug_name_raw`, `route`, `ndc`, `gsn`, `formulary_drug_cd` | Retained when available |
| `start_time`, `end_time` | Min/max over collapsed prescription rows |

Statistics: `results/metrics/mimic_exposure_statistics.json`

## `data/interim/drug_pairs.parquet`

Concurrent drug pairs from `python -m src.data.pipeline pairs` or `prepare-mimic`.

| Column | Description |
| --- | --- |
| `patient_id` | Patient identifier |
| `admission_id` | Present for `same_admission` rule |
| `drug_a`, `drug_b` | Unordered pair (`drug_a < drug_b`) |
| `pair_key` | `drug_a||drug_b` |
| `pair_rule` | Rule used (`same_admission` for full MIMIC run) |
| `drug_id_field` | Identifier column used (`drug_name_norm` until RxNorm mapping) |

Statistics: `results/metrics/drug_pair_statistics.json`

## `data/interim/twosides_drugs.parquet`

Drug catalog from `drug_smiles.csv`.

| Column | Description |
| --- | --- |
| `drug_id` | TWOSIDES PubChem CID string (e.g. `CID000002173`) |
| `smiles` | SMILES from source file |
| `source_file` | Provenance path |

Statistics: `results/metrics/twosides_drugs_statistics.json`

## `data/interim/twosides_interactions.parquet`

Full interaction table from `ddis.csv`.

| Column | Description |
| --- | --- |
| `drug_a`, `drug_b` | Canonical unordered TWOSIDES drug IDs |
| `pair_key` | `drug_a||drug_b` |
| `interaction_type` | Integer code from `type` (0–962 observed) |
| `neg_sample_drug` | CID from `Neg samples` |
| `source` | `twosides` |
| `source_file` | Provenance path |

Statistics: `results/metrics/twosides_statistics.json`

## `data/interim/twosides_unique_pairs.parquet`

One row per unordered TWOSIDES drug pair (derived from interactions).

| Column | Description |
| --- | --- |
| `drug_a`, `drug_b`, `pair_key` | Unordered TWOSIDES CID pair |

## `data/interim/mimic_pubchem_mapping_candidates.parquet`

Local-only candidate crosswalk (`python -m src.data.pipeline pubchem-candidates`). **No API calls.**

| Column | Description |
| --- | --- |
| `mimic_drug_name_norm` | MIMIC drug name from exposures |
| `mimic_drug_name_raw` | Example raw MIMIC name |
| `exposure_count` | Rows in `mimic_drug_exposures.parquet` |
| `twosides_drug_id` | TWOSIDES CID when locally matched |
| `pubchem_cid` | Integer CID when known from local evidence |
| `mapping_status` | `matched` or `unresolved` |
| `mapping_method` | `exact_cid_string`, `local_pubchem_cache`, or null |
| `mapping_evidence` | Local evidence description/path |

Statistics: `results/metrics/pubchem_candidate_statistics.json`

## `data/interim/mimic_twosides_labeled_pairs.parquet`

Real labeled dataset from `python -m src.data.pipeline labels-twosides`. One row per MIMIC pair × TWOSIDES `interaction_type`.

| Column | Description |
| --- | --- |
| `patient_id`, `admission_id` | MIMIC identifiers |
| `mimic_drug_a`, `mimic_drug_b`, `mimic_pair_key` | MIMIC formulary names |
| `twosides_drug_a`, `twosides_drug_b`, `twosides_pair_key` | Canonical TWOSIDES CIDs |
| `interaction_type` | TWOSIDES interaction code (preserved) |
| `neg_sample_drug` | TWOSIDES negative-sample CID |
| `label` | `positive` |
| `label_source` | `twosides` |

Statistics: `results/metrics/mimic_twosides_label_statistics.json`

## `data/interim/mimic_twosides_ml_positives.parquet`

Streaming deduplicated positive rows from `prepare-ml-twosides`. One row per `(patient_id, drug_a, drug_b, interaction_type)`.

| Column | Description |
| --- | --- |
| `patient_id` | MIMIC subject identifier |
| `drug_a`, `drug_b` | Canonical ordered MIMIC formulary names (`drug_a <= drug_b`) |
| `mimic_pair_key` | `drug_a||drug_b` |
| `twosides_drug_a`, `twosides_drug_b`, `twosides_pair_key` | Canonical TWOSIDES CIDs |
| `interaction_type` | TWOSIDES interaction code (preserved, not collapsed) |
| `label` | `positive` |
| `label_source` | `twosides` |
| `pair_rule` | MIMIC pair-construction rule (e.g. `same_admission`) |

Dedup key: `(patient_id, drug_a, drug_b, interaction_type)`.

## `data/interim/mimic_twosides_ml_negatives.parquet`

Deterministically sampled negative rows from real MIMIC pairs whose verified TWOSIDES CID pair is absent from TWOSIDES.

| Column | Description |
| --- | --- |
| `patient_id`, `admission_id` | MIMIC identifiers |
| `drug_a`, `drug_b`, `mimic_pair_key` | Canonical MIMIC pair |
| `twosides_drug_a`, `twosides_drug_b`, `twosides_pair_key` | Verified TWOSIDES CIDs (non-interacting pair) |
| `interaction_type` | null (no TWOSIDES interaction) |
| `label` | `negative` |
| `label_source` | `mimic_unmatched_twosides_pair` |

## `data/interim/mimic_twosides_ml_patient_splits.parquet`

Patient-level train/validation/test assignments. Each `patient_id` appears in exactly one split.

| Column | Description |
| --- | --- |
| `patient_id` | MIMIC subject identifier (string) |
| `split` | `train`, `val`, or `test` |

## `data/interim/mimic_twosides_ml_dataset.parquet`

Combined ML-ready dataset: deduped positives + sampled negatives + `split` column.

Statistics: `results/metrics/mimic_twosides_ml_dataset_statistics.json`

## `data/interim/mimic_pubchem_mapping.parquet`

MIMIC drug-name → PubChem CID crosswalk from `python -m src.data.pipeline pubchem-map`.

| Column | Description |
| --- | --- |
| `mimic_drug_name` | Normalized MIMIC formulary name queried |
| `mimic_drug_name_raw` | Example raw MIMIC name when available |
| `query_used` | Exact string sent to PubChem |
| `pubchem_cid` | Integer CID when **resolved**; null otherwise |
| `twosides_drug_id` | Zero-padded TWOSIDES-style ID (e.g. `CID000002173`) when resolved |
| `lookup_status` | `resolved`, `not_found`, `ambiguous`, `error`, `empty_result`, `invalid_query` |
| `lookup_detail` | Human-readable reason / error |
| `n_cids_returned` | Number of CIDs PubChem returned |
| `http_status`, `url` | API provenance |
| `source` | `pubchem_pug_rest` |
| `timestamp_utc` | Lookup timestamp |
| `from_cache` | Whether result came from local cache on this run |

Statistics: `results/metrics/pubchem_mapping_statistics.json`

**Verified mappings** have `lookup_status=resolved` with a single PubChem CID.
**Unresolved names** include `not_found`, `ambiguous`, and `error`.

## Pending / not yet generated

- MIMIC ↔ TWOSIDES label join using resolved CID crosswalk
- PubChem SMILES cache
- RDKit fingerprints
- Patient splits on real labeled dataset
- `data/processed/ml_dataset.parquet`
