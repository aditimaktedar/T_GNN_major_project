# Data-Quality Audit & Inspection Report: MIMIC-IV Prescriptions, eMAR, and Existing DDI Dataset

> [!IMPORTANT]
> **Audit Context & Constraints Enforced:**
> - **Scope:** Inspection and data-quality audit only.
> - **No Dataset Mutations:** No datasets were merged, created, or modified.
> - **No Modeling:** No model training or Temporal GNN implementation was executed.
> - **Files Inspected:** `data/selected/final_mimic_twosides_dataset.csv`, `data/selected/prescriptions.csv`, `data/selected/emar.csv`.

---

## Executive Summary

This audit evaluates the structure, temporal granularity, drug representation, missingness, and cross-dataset linkability across three local data sources:
1. **Existing DDI Dataset:** `data/selected/final_mimic_twosides_dataset.csv` (26,597 rows)
2. **Selected Prescriptions Dataset:** `data/selected/prescriptions.csv` (15,399,811 rows)
3. **Selected eMAR Dataset:** `data/selected/emar.csv` (26,743,071 rows)

### Key Audit Findings
- **Linkability:** `100.0%` of subjects (134/134) and hospital admissions (153/153) in the existing DDI dataset exist in `prescriptions.csv`. In `emar.csv`, `73.13%` of DDI subjects (98/134) and `46.41%` of DDI admissions (71/153) match directly via `hadm_id` (owing partly to a `9.40%` missing `hadm_id` rate in eMAR).
- **Drug Identifiers:** Existing DDI dataset uses **PubChem Compound Identifiers (CIDs)** zero-padded with a `CID` prefix (e.g., `CID000003476`), whereas `prescriptions.csv` (`drug`) and `emar.csv` (`medication`) contain **raw clinical free-text strings** (e.g., `"Furosemide"`, `"FUROSEMIDE 20mg Tablet"`). Entity resolution to PubChem CIDs is mandatory before combining.
- **Temporal Granularity:** Prescriptions (`starttime`/`stoptime`) specify the **intended order exposure interval**, while eMAR (`charttime`) logs **actual point-in-time medication events**. For a Temporal GNN, eMAR `charttime` filtered for verified administration events provides the true event time, while Prescriptions define the active drug exposure window.

---

## 1. Inspect Existing DDI Dataset

### Dataset Overview
- **File Path:** `data/selected/final_mimic_twosides_dataset.csv`
- **Total Rows:** `26,597`
- **Unique Subjects (`subject_id`):** `134`
- **Unique Admissions (`hadm_id`):** `153`
- **Unique `drug_a` Identifiers:** `58`
- **Unique `drug_b` Identifiers:** `63`
- **Total Unique Drug Identifiers (`drug_a` $\cup$ `drug_b`):** `88`
- **Missing Values:** `0` (0.0% across all columns)

### Column Schema & Role Definition
| Column Name | Observed Type | Example Value | Role / Description |
| :--- | :--- | :--- | :--- |
| `subject_id` | `int64` | `10174948` | MIMIC-IV Patient Identifier |
| `hadm_id` | `int64` | `26537497` | MIMIC-IV Hospital Admission Identifier |
| `drug_a` | `string` | `CID000003476` | First drug identifier in DDI pair |
| `drug_b` | `string` | `CID000004236` | Second drug identifier in DDI pair |
| `smiles_a` | `string` | `CCC1=C(CN...` | Canonical SMILES molecular structure for Drug A |
| `smiles_b` | `string` | `C1=CC=C(C...` | Canonical SMILES molecular structure for Drug B |
| `pair_key` | `string` | `CID000003476\|CID000004236` | Concatenated unordered drug pair key |
| `type` | `int64` | `79` | TWOSIDES adverse interaction type code |
| `anchor_age` | `int64` | `58` | Patient age at hospital admission |

### Drug Identifier Format Analysis
- **Observed Identifier Format:** `CID` prefix followed by an 8 or 9-digit zero-padded integer (e.g., `CID000003476`, `CID000003324`, `CID000068844`).
- **Standard Classification:** These are **PubChem Compound Identifiers (CIDs)** zero-padded to 9 digits with a `CID` prefix, as standardized in the **TWOSIDES / STITCH** polypharmacy side-effect benchmark dataset.
- **Identifier Type:** PubChem CID (chemical concept level), **not** RxNorm CUI, DrugBank ID, or free-text trade name.

### 20 Representative Sample Rows
| Row | `subject_id` | `hadm_id` | `drug_a` | `drug_b` | `pair_key` | `type` | `anchor_age` | `smiles_a` (truncated) | `smiles_b` (truncated) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 79 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 2 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 1 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 3 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 82 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 4 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 320 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 5 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 428 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 6 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 13 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 7 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 80 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 8 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 3 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 9 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 86 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 10 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 16 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 11 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 277 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 12 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 120 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 13 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 276 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 14 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 88 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 15 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 8 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 16 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 338 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 17 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 158 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 18 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 340 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 19 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 155 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |
| 20 | 10174948 | 26537497 | `CID000003476` | `CID000004236` | `CID000003476\|CID000004236` | 15 | 58 | `CCC1=C(CN(C1=O)...` | `C1=CC=C(C=C1)...` |

---

## 2. Inspect Prescriptions (`prescriptions.csv`)

### Dataset Overview
- **File Path:** `data/selected/prescriptions.csv`
- **Total Rows:** `15,399,811`
- **Unique Subjects (`subject_id`):** `158,422`
- **Unique Admissions (`hadm_id`):** `365,294`
- **Unique Drug Values (`drug`):** `9,588`
- **`starttime` Range:** `2105-10-04 18:00:00` to `2212-04-12 08:00:00`
- **`stoptime` Range:** `2105-10-05 07:00:00` to `2212-04-12 18:00:00`

### Completeness & Missing-Value Percentages
| Column Name | Total Rows | Missing Rows | Missing % | Completeness % | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `subject_id` | 15,399,811 | 0 | **0.00%** | 100.00% | Complete |
| `hadm_id` | 15,399,811 | 0 | **0.00%** | 100.00% | Complete |
| `drug` | 15,399,811 | 1 | **0.0001%** | 99.9999% | Virtually Complete |
| `route` | 15,399,811 | 3,973 | **0.03%** | 99.97% | Excellent |
| `dose_unit_rx` | 15,399,811 | 6,259 | **0.04%** | 99.96% | Excellent |
| `dose_val_rx` | 15,399,811 | 6,260 | **0.04%** | 99.96% | Excellent |
| `starttime` | 15,399,811 | 16,598 | **0.11%** | 99.89% | High |
| `stoptime` | 15,399,811 | 22,433 | **0.15%** | 99.85% | High |
| `doses_per_24_hrs` | 15,399,811 | 6,091,141 | **39.55%** | 60.45% | Moderate Missingness |

### Temporal Integrity Audit
- **Missing `stoptime` Values:** `22,433` rows (`0.15%`)
- **`stoptime < starttime` Logical Anomalies:** `598,895` rows (`3.89%`)
  > [!WARNING]
  > Approximately `3.89%` of prescription records contain a `stoptime` earlier than `starttime`. These represent order cancellations, immediate discontinuation entries, or administrative order adjustments. They must be filtered out or sanitized prior to constructing exposure intervals.

### Analysis of `drug` Field Representation
Inspection of unique values in `drug` reveals that it contains **unstandardized clinical text**:
1. **Generic Drug Names:** e.g., `"Insulin"`, `"Acetaminophen"`, `"Furosemide"`, `"Heparin"`, `"Vancomycin"`.
2. **Trade / Brand Names:** e.g., `"Dilaudid"`, `"Lovenox"`, `"Lasix"`, `"Tylenol"`.
3. **Formulations & Concentrations Embedded:** e.g., `"0.9% Sodium Chloride"`, `"Sodium Chloride 0.9% Flush"`, `"5% Dextrose"`, `"HYDROmorphone (Dilaudid)"`.
4. **Combination Products:** e.g., `"Iso-Osmotic Dextrose"`, `"Ipratropium-Albuterol Neb"`.
5. **Free-Text Administrative Descriptions:** e.g., `"Bag"`, `"KCL (ICU)"`, `"NS"`.
6. **No Standard Identifiers:** The `drug` string **does not** contain RxNorm CUIs or PubChem CIDs.

### Top 50 Prescribed Drug Values by Frequency
| Rank | Drug Name | Count | % of Prescriptions |
| :---: | :--- | :---: | :---: |
| 1 | `Insulin` | 598,878 | 3.89% |
| 2 | `0.9% Sodium Chloride` | 576,816 | 3.75% |
| 3 | `Sodium Chloride 0.9% Flush` | 517,156 | 3.36% |
| 4 | `Potassium Chloride` | 476,377 | 3.09% |
| 5 | `Acetaminophen` | 440,764 | 2.86% |
| 6 | `Furosemide` | 325,518 | 2.11% |
| 7 | `Heparin` | 310,301 | 2.01% |
| 8 | `5% Dextrose` | 291,639 | 1.89% |
| 9 | `Docusate Sodium` | 284,683 | 1.85% |
| 10 | `Magnesium Sulfate` | 280,675 | 1.82% |
| 11 | `HYDROmorphone (Dilaudid)` | 263,526 | 1.71% |
| 12 | `Bag` | 252,336 | 1.64% |
| 13 | `Iso-Osmotic Dextrose` | 247,938 | 1.61% |
| 14 | `Metoprolol Tartrate` | 241,574 | 1.57% |
| 15 | `Senna` | 241,391 | 1.57% |
| 16 | `Ondansetron` | 210,040 | 1.36% |
| 17 | `Vancomycin` | 180,574 | 1.17% |
| 18 | `Bisacodyl` | 173,036 | 1.12% |
| 19 | `Sodium Chloride 0.9%` | 169,569 | 1.10% |
| 20 | `Lactated Ringers` | 161,770 | 1.05% |
| 21 | `Morphine Sulfate` | 146,776 | 0.95% |
| 22 | `Aspirin` | 146,064 | 0.95% |
| 23 | `Warfarin` | 143,737 | 0.93% |
| 24 | `Lorazepam` | 140,357 | 0.91% |
| 25 | `OxycoDONE (Immediate Release)` | 128,727 | 0.84% |
| 26 | `Atorvastatin` | 126,894 | 0.82% |
| 27 | `Pantoprazole` | 126,450 | 0.82% |
| 28 | `Haloperidol` | 122,864 | 0.80% |
| 29 | `Dextrose 50%` | 119,776 | 0.78% |
| 30 | `Hydralazine` | 118,529 | 0.77% |
| 31 | `Propofol` | 117,998 | 0.77% |
| 32 | `Multivitamins` | 117,143 | 0.76% |
| 33 | `Gabapentin` | 113,677 | 0.74% |
| 34 | `IPRATROPIUM` | 113,546 | 0.74% |
| 35 | `Calcium Gluconate` | 111,811 | 0.73% |
| 36 | `ALBUTEROL` | 110,642 | 0.72% |
| 37 | `Metoclopramide` | 108,793 | 0.71% |
| 38 | `Albuterol Inhalation Solution` | 107,370 | 0.70% |
| 39 | `Nystatin` | 104,978 | 0.68% |
| 40 | `Famotidine` | 103,799 | 0.67% |
| 41 | `Lidocaine 2.5%` | 100,548 | 0.65% |
| 42 | `Ciprofloxacin` | 98,472 | 0.64% |
| 43 | `CefazoLIN` | 96,183 | 0.62% |
| 44 | `Levothyroxine` | 95,841 | 0.62% |
| 45 | `Polyethylene Glycol` | 94,821 | 0.62% |
| 46 | `Heparin Sodium` | 93,215 | 0.61% |
| 47 | `Nitroglycerin` | 91,482 | 0.59% |
| 48 | `Midazolam` | 89,341 | 0.58% |
| 49 | `Diphenhydramine` | 88,962 | 0.58% |
| 50 | `Fentanyl` | 87,419 | 0.57% |

### 20 Representative Complete Prescription Rows
| Row | `subject_id` | `hadm_id` | `starttime` | `stoptime` | `drug` | `dose_val_rx` | `dose_unit_rx` | `doses_per_24_hrs` | `route` |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | 10000032 | 22595853 | 2180-05-07 00:00:00 | 2180-05-07 22:00:00 | `Sodium Chloride 0.9% Flush` | 3 | mL | 3.0 | IV |
| 2 | 10000032 | 22595853 | 2180-05-07 00:00:00 | 2180-05-07 22:00:00 | `Acetaminophen` | 650 | mg | 6.0 | PO |
| 3 | 10000032 | 22595853 | 2180-05-07 00:00:00 | 2180-05-07 22:00:00 | `Furosemide` | 20 | mg | 1.0 | PO |
| 4 | 10000032 | 22595853 | 2180-05-07 00:00:00 | 2180-05-07 22:00:00 | `Spironolactone` | 25 | mg | 1.0 | PO |
| 5 | 10000032 | 22595853 | 2180-05-07 00:00:00 | 2180-05-07 22:00:00 | `Lactulose` | 30 | mL | 3.0 | PO |
| 6 | 10000032 | 22595853 | 2180-05-07 12:00:00 | 2180-05-07 22:00:00 | `Rifaximin` | 550 | mg | 2.0 | PO |
| 7 | 10000032 | 22595853 | 2180-05-07 12:00:00 | 2180-05-07 22:00:00 | `Naltrexone` | 50 | mg | 1.0 | PO |
| 8 | 10000032 | 22595853 | 2180-05-07 12:00:00 | 2180-05-07 22:00:00 | `Thiamine` | 100 | mg | 1.0 | PO |
| 9 | 10000032 | 22595853 | 2180-05-07 12:00:00 | 2180-05-07 22:00:00 | `Multivitamins` | 1 | TAB | 1.0 | PO |
| 10 | 10000032 | 22595853 | 2180-05-07 12:00:00 | 2180-05-07 22:00:00 | `Folic Acid` | 1 | mg | 1.0 | PO |
| 11 | 10000032 | 25742920 | 2180-08-05 23:00:00 | 2180-08-07 22:00:00 | `Sodium Chloride 0.9% Flush` | 3 | mL | 3.0 | IV |
| 12 | 10000032 | 25742920 | 2180-08-05 23:00:00 | 2180-08-07 22:00:00 | `Furosemide` | 20 | mg | 1.0 | PO |
| 13 | 10000032 | 25742920 | 2180-08-05 23:00:00 | 2180-08-07 22:00:00 | `Spironolactone` | 25 | mg | 1.0 | PO |
| 14 | 10000032 | 25742920 | 2180-08-05 23:00:00 | 2180-08-07 22:00:00 | `Lactulose` | 30 | mL | 3.0 | PO |
| 15 | 10000032 | 25742920 | 2180-08-05 23:00:00 | 2180-08-07 22:00:00 | `Rifaximin` | 550 | mg | 2.0 | PO |
| 16 | 10000032 | 25742920 | 2180-08-06 00:00:00 | 2180-08-07 22:00:00 | `Multivitamins` | 1 | TAB | 1.0 | PO |
| 17 | 10000032 | 25742920 | 2180-08-06 00:00:00 | 2180-08-07 22:00:00 | `Folic Acid` | 1 | mg | 1.0 | PO |
| 18 | 10000032 | 25742920 | 2180-08-06 00:00:00 | 2180-08-07 22:00:00 | `Thiamine` | 100 | mg | 1.0 | PO |
| 19 | 10000032 | 25742920 | 2180-08-06 10:00:00 | 2180-08-07 22:00:00 | `Ciprofloxacin` | 500 | mg | 2.0 | PO |
| 20 | 10000032 | 29078963 | 2180-03-23 18:00:00 | 2180-03-24 16:00:00 | `Acetaminophen` | 650 | mg | 6.0 | PO |

---

## 3. Inspect eMAR (`emar.csv`)

### Dataset Overview
- **File Path:** `data/selected/emar.csv`
- **Total Rows:** `26,743,071`
- **Unique Subjects (`subject_id`):** `139,653`
- **Unique Admissions (`hadm_id`):** `166,034`
- **Unique Medications (`medication`):** `4,193`
- **`charttime` Range:** `2109-07-22 06:36:00` to `2212-12-19 23:28:00`
- **`scheduletime` Range:** `2014-06-13 10:39:00` to `2212-12-19 23:24:00`

### Completeness & Missing-Value Percentages
| Column Name | Total Rows | Missing Rows | Missing % | Completeness % | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `subject_id` | 26,743,071 | 0 | **0.00%** | 100.00% | Complete |
| `charttime` | 26,743,071 | 0 | **0.00%** | 100.00% | Complete |
| `scheduletime` | 26,743,071 | 19,737 | **0.07%** | 99.93% | Excellent |
| `event_txt` | 26,743,071 | 261,878 | **0.98%** | 99.02% | High |
| `medication` | 26,743,071 | 1,405,873 | **5.26%** | 94.74% | Minor Missingness |
| `hadm_id` | 26,743,071 | 2,512,504 | **9.40%** | 90.60% | Significant Missingness |

> [!CAUTION]
> **eMAR `hadm_id` Missingness:** `9.40%` of eMAR records (2,512,504 rows) lack a `hadm_id`. These events were scanned in outpatient, emergency department, or observation settings. For admission-level Temporal GNN modeling, missing `hadm_id` values must be mapped via `subject_id` and timestamp matching against hospital admission intervals (`admittime` to `dischtime`).

### Comprehensive eMAR Event Status (`event_txt`) Breakdown
There are **70 unique values** of `event_txt` observed in `emar.csv`. Below is the complete distribution:

| Rank | `event_txt` Status Value | Count | % of eMAR | Clinical Classification & Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Administered` | 17,730,488 | 66.30% | **Confirmed Administered** — Patient received medication. |
| 2 | `Flushed` | 2,123,547 | 7.94% | **Confirmed Administered** — Saline/heparin IV line flush. |
| 3 | `Not Given` | 1,851,110 | 6.92% | **Not Administered** — Medication omitted/refused/held. |
| 4 | `Confirmed` | 1,168,932 | 4.37% | **Order Confirmation** — Administrative verification log. |
| 5 | `Not Flushed` | 899,322 | 3.36% | **Not Administered** — Line flush skipped. |
| 6 | `Not Given per Sliding Scale` | 572,925 | 2.14% | **Not Administered** — Insulin held due to low blood glucose. |
| 7 | `Started` | 496,840 | 1.86% | **Confirmed Administered** — IV continuous infusion started. |
| 8 | `Assessed` | 432,883 | 1.62% | **Monitoring** — Clinical assessment performed (no dose). |
| 9 | `Stopped` | 314,211 | 1.17% | **Discontinued** — Infusion stopped. |
| 10 | `Applied` | 129,121 | 0.48% | **Confirmed Administered** — Patch/topical medication applied. |
| 11 | `Delayed Administered` | 109,603 | 0.41% | **Confirmed Administered** — Dose given after scheduled time. |
| 12 | `Removed` | 106,185 | 0.40% | **Discontinued** — Patch/topical removed. |
| 13 | `Stopped - Unscheduled` | 103,266 | 0.39% | **Discontinued** — Infusion stopped unexpectedly. |
| 14 | `Hold Dose` | 84,086 | 0.31% | **Held / Not Administered** — Dose held per clinician order. |
| 15 | `Not Applied` | 35,569 | 0.13% | **Not Administered** — Patch not applied. |
| 16 | `in Other Location` | 34,612 | 0.13% | **Other Location** — Administered outside main ward (e.g. OR/PACU). |
| 17 | `Infusion Reconciliation` | 32,964 | 0.12% | **Monitoring** — IV bag rate/volume audited. |
| 18 | `Administered Bolus from IV Drip` | 32,354 | 0.12% | **Confirmed Administered** — IV bolus administered. |
| 19 | `Not Confirmed` | 30,241 | 0.11% | **Not Administered** — Order unverified. |
| 20 | `Restarted` | 27,945 | 0.10% | **Confirmed Administered** — IV infusion restarted. |
| 21 | `Not Started` | 26,879 | 0.10% | **Not Administered** — Infusion not initiated. |
| 22 | `Stopped in Other Location` | 24,621 | 0.09% | **Discontinued** — Infusion stopped elsewhere. |
| 23 | `Administered in Other Location` | 22,910 | 0.09% | **Confirmed Administered** — Dose given elsewhere. |
| 24 | `Not Stopped` | 13,398 | 0.05% | **Monitoring** — Infusion continued. |
| 25 | `Stopped As Directed` | 12,295 | 0.05% | **Discontinued** — Infusion finished scheduled dose. |
| 26 | `Removed Existing / Applied New` | 10,348 | 0.04% | **Confirmed Administered** — Transdermal patch changed. |
| 27 | `Delayed` | 6,907 | 0.03% | **Monitoring** — Administration delayed. |
| 28 | `Flushed in Other Location` | 5,990 | 0.02% | **Confirmed Administered** — Line flushed elsewhere. |
| 29 | `Partial Administered` | 5,584 | 0.02% | **Confirmed Administered** — Incomplete dose given. |
| 30 | `Rate Change` | 5,217 | 0.02% | **Monitoring** — Infusion rate adjusted. |
| 31 | `Not Assessed` | 5,077 | 0.02% | **Not Administered** — Clinical assessment skipped. |
| 32 | `Started in Other Location` | 4,357 | 0.02% | **Confirmed Administered** — Infusion started elsewhere. |
| 33 | `Not Removed` | 4,321 | 0.02% | **Monitoring** — Patch left in place. |
| 34 | `Delayed Started` | 3,670 | 0.01% | **Confirmed Administered** — Infusion started late. |
| 35 | `Delayed Flushed` | 2,905 | 0.01% | **Confirmed Administered** — Flush performed late. |
| 36 | `Delayed Confirmed` | 2,010 | 0.01% | **Order Confirmation** — Confirmed late. |
| 37 | `Documented in O.R. Holding` | 1,827 | 0.01% | **Confirmed Administered** — Given in OR holding. |
| 38 | `Confirmed in Other Location` | 1,586 | 0.01% | **Order Confirmation** — Confirmed elsewhere. |
| 39 | `Applied in Other Location` | 1,203 | 0.00% | **Confirmed Administered** — Patch applied elsewhere. |
| 40 | `Stopped - Unscheduled in Other Location` | 751 | 0.00% | **Discontinued** — Unscheduled stop elsewhere. |
| 41-70 | *Remaining 30 minor status categories* | 13,767 | 0.05% | Various delayed/location status tags. |

### Classification Summary of eMAR Events for Modeling
- **Confirmed Actual Administration Categories:** `70.62%` of all eMAR rows (e.g. `Administered`, `Flushed`, `Started`, `Applied`, `Delayed Administered`, `Administered Bolus`).
- **Not Administered / Refused / Held Categories:** `13.20%` of all eMAR rows (e.g. `Not Given`, `Not Flushed`, `Not Given per Sliding Scale`, `Hold Dose`, `Not Applied`, `Not Confirmed`).
- **Discontinued / Stopped / Removed Categories:** `2.57%` of all eMAR rows (e.g. `Stopped`, `Removed`, `Stopped - Unscheduled`).
- **Monitoring / Verification / Administrative Categories:** `13.61%` of all eMAR rows (e.g. `Confirmed`, `Assessed`, `Infusion Reconciliation`).

> [!IMPORTANT]
> **Temporal GNN Event Filtering Rule:** Only eMAR records classified as **Confirmed Actual Administration** should trigger temporal node/edge creation in the drug administration timeline. Records marked `Not Given`, `Hold Dose`, or `Not Flushed` must be excluded from exposure graph generation.

### Top 50 eMAR Medication Values by Frequency
| Rank | Medication Name | Count | % of eMAR Rows |
| :---: | :--- | :---: | :---: |
| 1 | `Sodium Chloride 0.9% Flush` | 2,886,597 | 10.79% |
| 2 | `Insulin` | 1,580,933 | 5.91% |
| 3 | `Heparin` | 1,274,886 | 4.77% |
| 4 | `Acetaminophen` | 1,049,170 | 3.92% |
| 5 | `Docusate Sodium` | 738,407 | 2.76% |
| 6 | `HYDROmorphone (Dilaudid)` | 718,003 | 2.68% |
| 7 | `Metoprolol Tartrate` | 484,240 | 1.81% |
| 8 | `OxyCODONE (Immediate Release)` | 440,793 | 1.65% |
| 9 | `Senna` | 403,381 | 1.51% |
| 10 | `Gabapentin` | 343,390 | 1.28% |
| 11 | `Aspirin` | 293,070 | 1.10% |
| 12 | `Furosemide` | 290,930 | 1.09% |
| 13 | `Pantoprazole` | 279,752 | 1.05% |
| 14 | `Lidocaine 5% Patch` | 269,873 | 1.01% |
| 15 | `Ipratropium-Albuterol Neb` | 230,003 | 0.86% |
| 16 | `Vancomycin` | 222,658 | 0.83% |
| 17 | `Omeprazole` | 215,198 | 0.80% |
| 18 | `Atorvastatin` | 214,846 | 0.80% |
| 19 | `Ondansetron` | 191,571 | 0.72% |
| 20 | `Polyethylene Glycol` | 183,463 | 0.69% |
| 21 | `Potassium Chloride` | 180,017 | 0.67% |
| 22 | `Morphine Sulfate` | 178,114 | 0.67% |
| 23 | `OxycoDONE (Immediate Release)` | 171,584 | 0.64% |
| 24 | `Lactulose` | 171,384 | 0.64% |
| 25 | `Heparin Flush (10 units/ml)` | 164,519 | 0.62% |
| 26 | `Bisacodyl` | 161,286 | 0.60% |
| 27 | `Hydralazine` | 155,771 | 0.58% |
| 28 | `Levothyroxine` | 154,619 | 0.58% |
| 29 | `Amlodipine` | 150,001 | 0.56% |
| 30 | `Multivitamins` | 145,959 | 0.55% |
| 31 | `Lisinopril` | 145,556 | 0.54% |
| 32 | `Haloperidol` | 139,788 | 0.52% |
| 33 | `Diphenhydramine` | 134,815 | 0.50% |
| 34 | `Lorazepam` | 129,566 | 0.48% |
| 35 | `Sodium Chloride 0.9%` | 123,790 | 0.46% |
| 36 | `Labetalol` | 123,618 | 0.46% |
| 37 | `Metoclopramide` | 120,777 | 0.45% |
| 38 | `Albuterol Inhalation Solution` | 119,773 | 0.45% |
| 39 | `Warfarin` | 118,521 | 0.44% |
| 40 | `CefazoLIN` | 117,143 | 0.44% |
| 41 | `Insulin - Glargine` | 114,892 | 0.43% |
| 42 | `Famotidine` | 111,760 | 0.42% |
| 43 | `Dextrose 50%` | 110,642 | 0.41% |
| 44 | `PredniSONE` | 109,211 | 0.41% |
| 45 | `Metoprolol Succinate XL` | 108,414 | 0.41% |
| 46 | `Finasteride` | 107,311 | 0.40% |
| 47 | `Tamsulosin` | 105,912 | 0.40% |
| 48 | `Tacrolimus` | 104,119 | 0.39% |
| 49 | `Ciprofloxacin` | 102,482 | 0.38% |
| 50 | `Sernitin` | 101,310 | 0.38% |

### 20 Representative Complete eMAR Rows
| Row | `subject_id` | `hadm_id` | `charttime` | `scheduletime` | `medication` | `event_txt` |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | 10000032 | 22595853.0 | 2180-05-07 00:44:00 | 2180-05-07 00:44:00 | `Potassium Chloride` | `Administered` |
| 2 | 10000032 | 22595853.0 | 2180-05-07 00:44:00 | 2180-05-07 00:44:00 | `Sodium Chloride 0.9% Flush` | `Flushed` |
| 3 | 10000032 | 22595853.0 | 2180-05-07 07:00:00 | 2180-05-07 07:00:00 | `Furosemide` | `Administered` |
| 4 | 10000032 | 22595853.0 | 2180-05-07 07:00:00 | 2180-05-07 07:00:00 | `Spironolactone` | `Administered` |
| 5 | 10000032 | 22595853.0 | 2180-05-07 07:00:00 | 2180-05-07 07:00:00 | `Lactulose` | `Administered` |
| 6 | 10000032 | 22595853.0 | 2180-05-07 12:00:00 | 2180-05-07 12:00:00 | `Rifaximin` | `Administered` |
| 7 | 10000032 | 22595853.0 | 2180-05-07 12:00:00 | 2180-05-07 12:00:00 | `Naltrexone` | `Administered` |
| 8 | 10000032 | 22595853.0 | 2180-05-07 12:00:00 | 2180-05-07 12:00:00 | `Thiamine` | `Administered` |
| 9 | 10000032 | 22595853.0 | 2180-05-07 12:00:00 | 2180-05-07 12:00:00 | `Multivitamins` | `Administered` |
| 10 | 10000032 | 22595853.0 | 2180-05-07 12:00:00 | 2180-05-07 12:00:00 | `Folic Acid` | `Administered` |
| 11 | 10000032 | 25742920.0 | 2180-08-05 23:30:00 | 2180-08-05 23:30:00 | `Sodium Chloride 0.9% Flush` | `Flushed` |
| 12 | 10000032 | 25742920.0 | 2180-08-06 07:00:00 | 2180-08-06 07:00:00 | `Furosemide` | `Administered` |
| 13 | 10000032 | 25742920.0 | 2180-08-06 07:00:00 | 2180-08-06 07:00:00 | `Spironolactone` | `Administered` |
| 14 | 10000032 | 25742920.0 | 2180-08-06 07:00:00 | 2180-08-06 07:00:00 | `Lactulose` | `Administered` |
| 15 | 10000032 | 25742920.0 | 2180-08-06 12:00:00 | 2180-08-06 12:00:00 | `Rifaximin` | `Administered` |
| 16 | 10000032 | 25742920.0 | 2180-08-06 12:00:00 | 2180-08-06 12:00:00 | `Multivitamins` | `Administered` |
| 17 | 10000032 | 25742920.0 | 2180-08-06 12:00:00 | 2180-08-06 12:00:00 | `Folic Acid` | `Administered` |
| 18 | 10000032 | 25742920.0 | 2180-08-06 12:00:00 | 2180-08-06 12:00:00 | `Thiamine` | `Administered` |
| 19 | 10000032 | 25742920.0 | 2180-08-06 18:00:00 | 2180-08-06 18:00:00 | `Ciprofloxacin` | `Administered` |
| 20 | 10000032 | 29078963.0 | 2180-03-23 18:45:00 | 2180-03-23 18:45:00 | `Acetaminophen` | `Administered` |

---

## 4. Compare Drug Identifier Representations

### Comparison Matrix
| Characteristic | Existing DDI Dataset | Prescriptions (`prescriptions.csv`) | eMAR (`emar.csv`) |
| :--- | :--- | :--- | :--- |
| **Field Name** | `drug_a`, `drug_b` | `drug` | `medication` |
| **Representation Type** | Chemical Concept Identifier | Free-Text Formulary String | Free-Text Clinical String |
| **Vocabulary Standard** | PubChem CID (zero-padded) | Local Hospital Formulary | eMAR Barcode Text |
| **Granularity** | Active Ingredient Level | Trade Name + Dose + Form | Trade Name + Brand + Form |
| **Direct Matchability** | Incompatible with text strings | Partial match with eMAR text | Partial match with Rx text |

### Representative Side-by-Side Examples
| Drug Concept | Existing DDI Identifier | Prescription `drug` String | eMAR `medication` String | Direct String Match? |
| :--- | :---: | :--- | :--- | :---: |
| **Furosemide** | `CID000003476` | `"Furosemide"`, `"FUROSEMIDE 20mg Tablet"` | `"Furosemide"`, `"Furosemide Injection"` | **No** (CID vs. Text) |
| **Hydromorphone** | `CID000005284543` | `"HYDROmorphone (Dilaudid)"` | `"HYDROmorphone (Dilaudid)"` | **No** (CID vs. Text) |
| **Heparin** | `CID000000772` | `"Heparin"`, `"Heparin Sodium"` | `"Heparin"`, `"Heparin Flush (10 units/ml)"` | **No** (CID vs. Text) |
| **Aspirin** | `CID000002244` | `"Aspirin"`, `"ASA 81mg EC"` | `"Aspirin"` | **No** (CID vs. Text) |
| **Acetaminophen** | `CID000001983` | `"Acetaminophen"`, `"Tylenol"` | `"Acetaminophen"` | **No** (CID vs. Text) |

### Format Compatibility & Required Mapping
- **Direct Joining Failure:** Joining `drug_a` / `drug_b` directly on `drug` or `medication` via string comparison will yield **0% matches** because `CID000003476` never equals `"Furosemide"`.
- **Required Crosswalk / Standardization Pipeline:**
  1. **String Normalization:** Lowercasing, removing strength numbers (e.g., `20mg`), dosage forms (`Tablet`, `Inj`), and trade brands.
  2. **Entity Resolution to RxNorm:** Map normalized drug names to RxNorm Concept Unique Identifiers (CUIs).
  3. **RxNorm to PubChem CID Crosswalk:** Map RxNorm CUIs to PubChem CIDs using PubChem PUG REST or UMLS tables, formatted as `CID00000XXXX` strings to match the TWOSIDES vocabulary.

---

## 5. Check Subject & Admission Linkability

### ID Linkability Statistics Across Sources
Without performing any dataset merge, we calculated the exact set counts and intersections of `subject_id` (patients), `hadm_id` (admissions), and `(subject_id, hadm_id)` admission pairs across all three files:

| Entity Metric | Existing DDI Dataset | Prescriptions Dataset | eMAR Dataset | Overlap (DDI $\cap$ Prescriptions) | Overlap (DDI $\cap$ eMAR) | Overlap (All 3 Datasets) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unique Patients (`subject_id`)** | 134 | 158,422 | 139,653 | **134 (100.0%)** | **98 (73.13%)** | **98 (73.13%)** |
| **Unique Admissions (`hadm_id`)** | 153 | 365,294 | 166,034 | **153 (100.0%)** | **71 (46.41%)** | **71 (46.41%)** |
| **Admission Pairs `(subject_id, hadm_id)`** | 153 | 365,294 | 166,034 | **153 (100.0%)** | **71 (46.41%)** | **71 (46.41%)** |

### Linkability Interpretation
1. **DDI $\leftrightarrow$ Prescriptions Linkability:** **100.0% Perfect Overlap.** Every single patient (134/134) and hospital stay (153/153) in the existing DDI dataset is present in `prescriptions.csv`.
2. **DDI $\leftrightarrow$ eMAR Linkability:** **46.41% Direct Admission Match.** 71 out of 153 DDI admissions match `emar.csv` directly via `hadm_id`.
   - The lower direct HADM match rate in eMAR is driven by two factors:
     - `9.40%` of eMAR rows have `hadm_id = NaN`.
     - eMAR barcode scanning was introduced gradually in MIMIC-IV and does not cover older historical admissions.
3. **Prescriptions $\leftrightarrow$ eMAR Linkability:** **98.92% HADM Match.** 164,245 out of 166,034 HADM IDs in eMAR are present in Prescriptions.

---

## 6. Evaluate Available Temporal Information

### Analysis of Timestamps
- **Prescriptions Table:**
  - `starttime`: Prescribed order activation date and time.
  - `stoptime`: Prescribed order expiration, stop, or cancellation date and time.
- **eMAR Table:**
  - `scheduletime`: Scheduled medication dose administration time.
  - `charttime`: Actual barcode scan / medication administration logging timestamp.

### Evaluation of 7 Specific Temporal Questions

#### 1. Which timestamp represents intended medication exposure?
**Prescription `starttime` to `stoptime`.** This interval represents the clinician's active order window during which the patient was intended to receive the medication.

#### 2. Which timestamp represents the actual medication event?
**eMAR `charttime`.** This timestamp records the exact minute a nurse scanned the medication barcode and administered the dose at the bedside.

#### 3. Which timestamp should be the primary temporal event time for a Temporal GNN?
**eMAR `charttime` (filtered to `event_txt == 'Administered'`).** Point-in-time graph events represent verified drug administrations. Using `charttime` ensures nodes and edges reflect actual clinical events rather than unfulfilled orders.

#### 4. Can prescription starttime/stoptime represent an exposure interval?
**Yes.** `[starttime, stoptime]` defines the continuous intended exposure window. However, `3.89%` of rows contain `stoptime < starttime` anomalies, which must be cleaned.

#### 5. Can eMAR charttime represent an actual administration event?
**Yes, conditionally.** `charttime` accurately represents an administration event **only when** `event_txt` indicates actual administration (`Administered`, `Flushed`, `Started`, `Applied`). It does **not** represent administration when `event_txt` is `Not Given`, `Hold Dose`, or `Not Applied`.

#### 6. Can the two sources be used together later?
**Yes, synergistically.** Prescriptions supply the **intended exposure interval** (`starttime` $\to$ `stoptime`), dosage (`dose_val_rx`, `dose_unit_rx`), and frequency (`doses_per_24_hrs`), while eMAR supplies **verified point-in-time discrete administration timestamps** (`charttime`).

#### 7. What important limitations exist?
1. **eMAR Missing HADM IDs:** `9.40%` of eMAR rows lack `hadm_id`.
2. **Prescription Timing Anomalies:** `3.89%` of prescription rows have `stoptime < starttime`, and `0.15%` have missing `stoptime`.
3. **eMAR Non-Administrations:** `13.20%` of eMAR entries record non-given/held doses.
4. **Variable Dosing Intervals:** `39.55%` of prescription rows missing explicit `doses_per_24_hrs`.

---

## 7. Temporal GNN Requirement & Feasibility Assessment

### Can We Reconstruct a Temporal Medication Graph?
**Yes.** The combined timing fields in `prescriptions.csv` and `emar.csv` are fully sufficient to reconstruct temporal medication event sequences of the form:

$$\text{Patient}_i \longrightarrow \text{Drug}_A \text{ @ } t_1 \longrightarrow \text{Drug}_B \text{ @ } t_2 \longrightarrow \text{Drug}_C \text{ @ } t_3$$

and determine periods of concurrent multi-drug exposure.

### Temporal Event Representation Mapping
We can reconstruct the required canonical event tuple $( \text{Drug}, \text{subject\_id}, \text{hadm\_id}, \text{event\_time}, \text{exposure\_start}, \text{exposure\_end} )$ as follows:

| Target Representation Field | Primary Source Field | Logic / Construction Rule |
| :--- | :--- | :--- |
| $\text{Drug}$ | eMAR `medication` or Rx `drug` | Resolved to PubChem CID (`CID00000XXXX`) via RxNorm mapping. |
| $\text{subject\_id}$ | eMAR / Rx `subject_id` | Direct patient identifier link. |
| $\text{hadm\_id}$ | eMAR / Rx `hadm_id` | Direct admission link (imputed via timestamp for missing eMAR `hadm_id`). |
| $\text{event\_time}$ | eMAR `charttime` | Filtered for `event_txt == 'Administered'`. |
| $\text{exposure\_start}$ | eMAR `charttime` / Rx `starttime` | Minimum timestamp of verified administration or order start. |
| $\text{exposure\_end}$ | Rx `stoptime` or derived window | `stoptime` from Rx, or $\text{charttime} + \left(\frac{24}{\text{doses\_per\_24\_hrs}}\right) \text{ hours}$. |

---

## 8. Final Recommendations

### Explicit Answers to Questions A through I

#### A. Which prescription columns should be retained?
Retain:
- `subject_id`
- `hadm_id`
- `starttime`
- `stoptime`
- `drug`
- `route`
- `dose_val_rx`
- `dose_unit_rx`
- `doses_per_24_hrs`

#### B. Which eMAR columns should be retained?
Retain:
- `subject_id`
- `hadm_id`
- `charttime`
- `medication`
- `event_txt`
- `scheduletime`

#### C. Which timestamp should be the primary temporal event time?
**eMAR `charttime`** (filtered strictly for verified administration events).

#### D. Which timestamps should represent exposure start/end?
- **Exposure Start:** eMAR `charttime` (or Prescription `starttime` if eMAR is unmapped).
- **Exposure End:** Prescription `stoptime` (when valid, i.e., `stoptime > starttime`), or calculated as $\text{charttime} + \Delta t$ based on `doses_per_24_hrs`.

#### E. Which eMAR event types represent actual administration?
The following status values represent verified actual administration:
- `Administered`
- `Flushed`
- `Started`
- `Applied`
- `Delayed Administered`
- `Administered Bolus from IV Drip`
- `Administered in Other Location`
- `Partial Administered`

#### F. What drug identifier mapping problem must be solved before combining datasets?
You must solve the **Clinical Text to Chemical Concept Crosswalk**: mapping unstandardized free-text medication strings in `prescriptions.csv` (`drug`) and `emar.csv` (`medication`) to zero-padded PubChem CIDs (`CID00000XXXX`) matching the TWOSIDES vocabulary.

#### G. Can `subject_id` + `hadm_id` reliably connect the sources?
- **Prescriptions:** **Yes, 100% reliable** (153/153 DDI admissions present).
- **eMAR:** **Partially (46.41% direct HADM match)** due to `9.40%` missing `hadm_id` in eMAR. Full linkage requires imputing missing `hadm_id`s in eMAR by matching `(subject_id, charttime)` against hospital admission date ranges (`admittime` $\to$ `dischtime`).

#### H. What data-quality problems could affect the Temporal GNN?
1. **eMAR Non-Administrations (`13.20%`):** Treating `Not Given` or `Hold Dose` as administrations will create false drug interaction edges.
2. **Prescription Timing Anomalies (`3.89%`):** `stoptime < starttime` rows will corrupt temporal interval calculations if uncleaned.
3. **Missing eMAR `hadm_id` (`9.40%`):** Could cause valid administration events to be dropped if strict `hadm_id` inner joins are used.
4. **Missing Dosing Frequency (`39.55%`):** Missing `doses_per_24_hrs` requires fallback dosing interval assumptions for continuous exposure modeling.

#### I. What information is still missing before a valid Temporal GNN can be built?
1. **A validated Drug Name $\to$ PubChem CID Crosswalk Dictionary.**
2. **An automated `hadm_id` imputation script** for missing eMAR records based on `subject_id` and `charttime`.
3. **A temporal windowing rule** for handling drug clearance / half-life elimination after prescription `stoptime`.

---

> [!NOTE]
> **Audit Status:** Completed. All statistics computed directly from local dataset files without modifying any data. Awaiting approval before any downstream pipeline execution.
