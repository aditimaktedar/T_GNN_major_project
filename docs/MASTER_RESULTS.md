# Master experiment results

**Single source of truth:** `results/metrics/MASTER_RESULTS.json`

All multiclass experiment metrics (955-class, frequent363, Logistic Regression, Static GAT, Molecular GNN, and future models) are stored in this one file.

- **Do not** add a new `*_summary.json` or `*_protocol.json` when a model is trained.
- A rerun **overwrites** `formulations.<formulation>.models.<model>` in place.
- Each overwrite sets a new `run_id` and `updated_at`, and records `supersedes_run_id` for the previous entry.
- Duplicate keys for the same formulation + model are not created unless `keep_run_history=True` (compact prior-run index only).

## Layout

```text
formulations
  955
    models
      logistic_regression
      static_gat
      molecular_gnn
  frequent363
    models
      logistic_regression
      static_gat
      molecular_gnn
```

Each model entry contains:

- `run_id`, `updated_at`
- `n_classes`, `dataset_path`, `checkpoint`
- `metrics` — row-level train/val/test from the existing metric functions
- `evaluation_protocol` — unchanged protocol payload (slices and `pair_type_dedup`)
- `splits` — normalized view of the same numbers for reporting

## Training writes

```bash
python -m src.baselines.run_multiclass_baselines --formulation frequent363
python -m src.baselines.run_molecular_gnn --formulation frequent363
```

These commands upsert the trained model(s) in `MASTER_RESULTS.json`. They do not create a new summary filename.

Merge archived per-run files (already done once; safe to re-run, overwrites from those archives):

```bash
python -m src.evaluation.master_results --ingest-existing
```

## Archived files (do not delete yet)

These older files were ingested and can later be moved to an archive folder. They are **not** updated by new training:

- `final_mimic_twosides_multiclass_*_summary.json`
- `final_mimic_twosides_multiclass_*_protocol.json`
- `final_mimic_twosides_multiclass_run_summary.json`
- `final_mimic_twosides_frequent363_multiclass_*`
- `baseline_comparison.json` (derived 955-class comparison; regenerate from master if needed)

Keep as non-result archives (not model scores):

- `final_mimic_twosides_ml_preprocessing_report.json`
- `final_mimic_twosides_ml_frequent363_report.json`
- `target_formulation_analysis.json`
- pipeline `*_statistics.json` files
- synthetic demo / XAI / DrugBank / RAG files
