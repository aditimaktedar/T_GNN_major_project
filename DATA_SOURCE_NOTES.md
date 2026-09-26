# Data source notes

This file records what is established from **approved project sources** and from **local inspection** of downloaded files.

## MIMIC-IV 2.1 (locally present)

- Approved source: https://www.kaggle.com/datasets/mangeshwagle/mimic-iv-2-1
- Local path: `data/raw/mimic/mimic-iv-2.1/`
- Layout observed:
  - `hosp/` — hospital module (21 CSV tables)
  - `icu/` — ICU module (8 CSV tables)
  - `CHANGELOG.txt`, `LICENSE.txt`, `SHA256SUMS.txt`

### Files discovered (hosp/)

| File | Approx. size | Used in Member B stage |
| --- | --- | --- |
| `prescriptions.csv` | 2.36 GB / 15,399,811 rows | **Yes — primary medication source** |
| `pharmacy.csv` | 2.82 GB | No (excluded; inpatient pharmacy orders, not used for this stage) |
| `emar.csv` / `emar_detail.csv` | 3.8 GB / 5.4 GB | No (eMAR administration records; excluded) |
| `poe.csv` / `poe_detail.csv` | 3.3 GB / 178 MB | No |
| `admissions.csv` | 67 MB | No (not required for prescription-based pair construction) |
| `patients.csv` | 9.4 MB | No |
| Other hosp tables | various | No |

### `hosp/prescriptions.csv` — inspected columns

`subject_id`, `hadm_id`, `pharmacy_id`, `poe_id`, `poe_seq`, `starttime`, `stoptime`, `drug_type`, `drug`, `formulary_drug_cd`, `gsn`, `ndc`, `prod_strength`, `form_rx`, `dose_val_rx`, `dose_unit_rx`, `form_val_disp`, `form_unit_disp`, `doses_per_24_hrs`, `route`

Sample values confirm patient/admission identifiers, drug names, and start/stop timestamps are populated for most rows.

### Ingestion decisions

- **Single source table:** `hosp/prescriptions.csv` only.
- **Canonical renaming:** `subject_id→patient_id`, `hadm_id→admission_id`, `drug→drug_name_raw`, `starttime→start_time`, `stoptime→end_time`, `dose_val_rx→dose`.
- **Streaming reads:** CSV ingested in chunks (`DEFAULT_MIMIC_CHUNKSIZE=200_000`) to parquet under `data/interim/`.
- **No raw modification:** Original CSVs are read-only.

### Cleaning decisions

Rows dropped when:
- missing `patient_id` or `drug_name_raw`
- unparseable `start_time` / `end_time` when present
- inverted interval (`start_time > end_time`)

`drug_name_norm` = lowercase, whitespace-normalized `drug_name_raw`.

### RxNorm normalization

- **Status: pending.** No RxNorm mapping file is available yet.
- Pipeline command `normalize` without `--mapping` writes `mimic_prescriptions_normalized.parquet` with `rxnorm_status=pending` and preserves MIMIC drug names.
- **No RxNorm IDs were invented.**

### Drug-pair construction

- Exposures collapsed to unique `(patient_id, admission_id, drug_name_norm)` before pairing.
- Full MIMIC run uses **`same_admission`** rule: distinct drugs co-occurring within the same hospital admission.
- `interval_overlap` remains available when start/end times are present but is not the default for the full-scale MIMIC command because admission-level pairing is more tractable on 15M+ prescription rows.
- Pair key uses **`drug_name_norm`** until RxNorm mapping is supplied.

### Limitations

- MIMIC drug names are hospital formulary strings, not yet mapped to RxNorm or DrugBank.
- ICU-only medication paths (`icu/inputevents.csv`, etc.) are not included in this stage.
- eMAR/pharmacy tables may contain complementary administration data but were excluded to avoid duplicate/conflicting exposure definitions.
- DDI labels (TWOSIDES) are not yet joined.

### MIMIC ↔ TWOSIDES label join (completed)

Command: `python -m src.data.pipeline labels-twosides`

Uses only **489 verified** local PubChem CID mappings (`local_pubchem_cache`). Output: `data/interim/mimic_twosides_labeled_pairs.parquet`.

Verified counts (real run):

| Metric | Count |
| --- | ---: |
| MIMIC pairs considered | 130,753,703 |
| Pairs with both drugs mapped | 5,382,930 |
| Pairs joined to TWOSIDES | 5,039,872 |
| Positive interaction records | 849,453,614 |
| TWOSIDES interaction types preserved | 963 |
| Unique patients | 148,258 |
| Negative labels in join output | 0 |

Statistics: `results/metrics/mimic_twosides_label_statistics.json`

### ML dataset preparation (streaming)

Command: `python -m src.data.pipeline prepare-ml-twosides`

**Before running on real data**, verify interim TWOSIDES files are real-scale (63,472 unique pairs). If pytest or a partial run overwrote them, reload from raw CSV first (read-only on raw):

```bash
python -m src.data.pipeline twosides
```

Then build the ML dataset (reads labeled pairs, drug pairs, mappings, and TWOSIDES unique pairs; does **not** modify them):

```bash
python -m src.data.pipeline prepare-ml-twosides --seed 42 --negative-ratio 1.0
```

Preflight checks refuse to run if labeled rows < 800M, TWOSIDES unique pairs < 60K, or verified mappings < 489.

Builds a compact ML-ready dataset **without loading all 849M join rows into RAM**:

1. Stream-deduplicate positives at `(patient_id, drug_a, drug_b, interaction_type)`.
2. Collect eligible negatives by streaming `drug_pairs.parquet`: both drugs must map to verified TWOSIDES CIDs and the canonical TWOSIDES pair must be **absent** from `twosides_unique_pairs.parquet`.
3. Sample negatives deterministically: `min(round(n_positives × negative_ratio), eligible_pool)` with explicit seed (default ratio 1.0, seed 42).
4. Assign patient-level train/val/test splits (default 0.70/0.15/0.15).

Outputs:

- `data/interim/mimic_twosides_ml_positives.parquet`
- `data/interim/mimic_twosides_ml_negatives.parquet`
- `data/interim/mimic_twosides_ml_patient_splits.parquet`
- `data/interim/mimic_twosides_ml_dataset.parquet`
- `results/metrics/mimic_twosides_ml_dataset_statistics.json`

**Negative assumption:** sampled negatives are MIMIC co-prescription pairs with verified TWOSIDES CIDs but no TWOSIDES interaction record. This is a modeling assumption, not a verified non-interaction.

## TWOSIDES (locally present)

Approved source: https://github.com/jcsun-00/Twosides

Local path: `data/raw/twosides/`

| File | Size / rows | Role |
| --- | --- | --- |
| `ddis.csv` | 186.65 MB / 4,576,287 rows | DDI interaction records |
| `drug_smiles.csv` | 43 KB / 645 rows | Drug catalog with SMILES |

### `ddis.csv` — inspected columns (4 columns, 0 missing values)

| Column | dtype | Meaning |
| --- | --- | --- |
| `d1` | string | First drug identifier |
| `d2` | string | Second drug identifier |
| `type` | int64 | Interaction type code (0–962 observed) |
| `Neg samples` | string | Negative-sample drug identifier for the row |

Each row is one **(drug_a, drug_b, interaction_type)** record. The same unordered drug pair appears on multiple rows with different `type` values (median 52 rows/pair locally; max 505).

Verified counts:
- 4,576,287 interaction records
- 63,472 unique unordered drug pairs
- 645 unique drugs referenced in `d1`/`d2`
- 963 distinct `type` values (0–962)

### `drug_smiles.csv` — inspected columns (3 columns)

| Column | dtype | Meaning |
| --- | --- | --- |
| `Unnamed: 0` | int64 | Row index from export (ignored in pipeline) |
| `drug_id` | string | TWOSIDES drug identifier |
| `smiles` | string | SMILES string for the drug |

All 645 `drug_id` values match drugs referenced in `ddis.csv`. All `Neg samples` values also match the catalog.

### Identifier scheme (verified from files)

TWOSIDES uses **zero-padded PubChem Compound ID strings**: `CID` + 9 digits (example: `CID000002173`).

This is **not RxNorm** and **not** MIMIC formulary text. Crosswalk to MIMIC remains **blocked/pending**.

### Pipeline outputs

```bash
python -m src.data.pipeline twosides
```

Writes:
- `data/interim/twosides_drugs.parquet`
- `data/interim/twosides_interactions.parquet`
- `data/interim/twosides_unique_pairs.parquet`

Statistics: `results/metrics/twosides_statistics.json`, `results/metrics/twosides_drugs_statistics.json`

## OFFSIDES

Approved source: https://github.com/tatonetti-lab/offsides

- **Not downloaded locally.**
- Approved GitHub repo contains processing code/notebooks; no ready-made OFFSIDES table verified locally.

## DrugBank

- Approved download URL: https://go.drugbank.com/releases/5-1-16/downloads/all-full-database
- Licensed/private. **Not downloaded.** Credentials must never be stored in this repository.

## PharmGKB / CPIC

- Approved page: https://www.clinpgx.org/downloads
- **Not downloaded.**

## PubChem / MIMIC ↔ TWOSIDES alignment

### Local candidate preparation (no API)

```bash
python -m src.data.pipeline pubchem-candidates
```

Reads:
- `data/interim/mimic_drug_exposures.parquet` (unique MIMIC drug names + exposure counts)
- `data/interim/twosides_drugs.parquet` (645 TWOSIDES CIDs)
- Optional prior cache: `data/interim/pubchem/cid_lookup/cid_success.jsonl`

Writes:
- `data/interim/mimic_pubchem_mapping_candidates.parquet`
- `results/metrics/pubchem_candidate_statistics.json`

**Direct local mapping rules only:**
1. `exact_cid_string` — MIMIC `drug_name_norm` exactly equals a TWOSIDES `drug_id`
2. `local_pubchem_cache` — prior on-disk PubChem cache entry with `lookup_status=resolved` and `twosides_drug_id` present in `twosides_drugs.parquet`

No fuzzy matching. No new API calls in this stage.

### PubChem PUG REST (optional later stage)

- Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- Separate command: `python -m src.data.pipeline pubchem-map` (calls API; not run during local candidate prep)
- Cache directory: `data/interim/pubchem/cid_lookup/`
- **No RxNorm substitution.**

### MIMIC → PubChem crosswalk policy

- Query: conservative normalization (trim + collapse whitespace) of `drug_name_norm`
- Resolved only when PubChem returns **exactly one CID** for the name query
- Multiple CIDs → `ambiguous` (unresolved)
- HTTP 404 / empty → `not_found` (unresolved)
- Output: `data/interim/mimic_pubchem_mapping.parquet`
- Statistics: `results/metrics/pubchem_mapping_statistics.json`

## RxNorm

Required normalization target for MIMIC drug names. The mapping source/procedure is **not chosen yet**. Pipeline accepts a user-supplied CSV/TSV mapping when available.

## Pipeline commands for real MIMIC (this repository)

```bash
source .venv/bin/activate
python -m src.data.pipeline prepare-mimic
```

Or stage-by-stage:

```bash
python -m src.data.pipeline ingest-mimic
python -m src.data.pipeline clean
python -m src.data.pipeline normalize
python -m src.data.pipeline pairs --rule same_admission
```
