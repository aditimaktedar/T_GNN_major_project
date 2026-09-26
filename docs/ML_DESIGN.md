# ML Design — DDI Interaction-Type Prediction

Design document for the model stage using the preprocessed dataset:

`data/processed/final_mimic_twosides_ml.csv`

**Status:** design only — no training code or dataset changes in this stage.

---

## 1. Problem definition

**Task:** Given two drugs represented by SMILES strings (`smiles_a`, `smiles_b`), predict the TWOSIDES **interaction type** (`type`).

Each row is one observed MIMIC admission context where a drug pair is associated with one TWOSIDES interaction type. The dataset contains **26,597 rows**, **145 unique drug pairs**, and **962 unique interaction types** (integers in range 0–961; one TWOSIDES code from the full 0–962 catalog does not appear in this subset).

This is **not** the binary “interaction vs no interaction” task assumed by the existing Member B baselines (`label` ∈ {positive, negative}). The current dataset is a **multi-class interaction-type classification** problem.

**Project naming note:** The repository is called *TemporalDDI-GNN*, but the selected dataset has **no timestamps** and `src/models/tgnn.py` / `src/graph/graph_builder.py` are currently stubs. Temporal modeling is **out of scope for the first ML stage** on this file unless admission time is added later.

---

## 2. Input features

### Primary inputs (preserved from dataset)

| Column | Role |
|--------|------|
| `smiles_a` | Molecular structure of drug A |
| `smiles_b` | Molecular structure of drug B |

### Context columns (not molecular inputs)

| Column | Role |
|--------|------|
| `pair_key` | Unordered CID pair identifier; defines the underlying DDI pair |
| `subject_id`, `hadm_id` | Patient/admission context; **not** fed as model inputs initially |
| `split` | Train/val/test assignment |

### Existing project feature path (Member B)

The implemented pipeline (`src/features/rdkit_features.py`, `src/data/pyg_dataset.py`) currently uses:

- **Morgan fingerprints** (radius 2, 2048 bits) derived from SMILES
- **Not atom-level molecular graphs**

The graph-based model stage should **introduce true molecular graphs from SMILES** as intended by the GNN/T-GNN direction, while optionally retaining Morgan fingerprints for baselines.

---

## 3. Molecular graph construction

Use **RDKit** (`Chem.MolFromSmiles`) — already a project dependency — to convert each SMILES into an atom-bond graph.

### Per-molecule graph `G = (V, E)`

| Component | Construction |
|-----------|--------------|
| **Nodes** | One node per heavy atom (RDKit atom index) |
| **Edges** | Undirected chemical bonds; store both directions for PyG |
| **Node features** | Atomic number, degree, formal charge, hybridization, aromaticity, total H count, in-ring flag (standard chemoinformatics featurization) |
| **Edge features** | Bond type (single/double/triple/aromatic) |

Invalid SMILES must be **skipped or flagged**, not invented. Inspection of the selected dataset found **0 invalid SMILES**.

### Caching

Because only **58 unique `smiles_a`** and **63 unique `smiles_b`** values exist, graphs should be **cached by SMILES string** to avoid rebuilding identical molecular graphs for 26,597 rows.

---

## 4. Proposed GNN / T-GNN architecture

### Current repository state

| Module | Status | Current behavior |
|--------|--------|------------------|
| `src/models/tgnn.py` | Stub | No model defined |
| `src/graph/graph_builder.py` | Stub | No graph builder |
| `src/baselines/static_gat.py` | Implemented | 2-node **meta-graph** with fingerprint vectors; **binary** output |
| `src/data/pyg_dataset.py` | Implemented | Expects fingerprints + binary `label`; **not compatible** with current CSV without adaptation |

### Recommended architecture (phase 1): Dual molecular GNN encoder

Because the dataset has no temporal fields, phase 1 should be a **static pair GNN**, reserving temporal extensions for Member A / future work.

```
smiles_a → MolGraph_a → GNN encoder → h_a ─┐
                                            ├→ pair fusion → classifier → type (962 classes)
smiles_b → MolGraph_b → GNN encoder → h_b ─┘
```

**GNN encoder (per drug):**

- Input: PyG `Data` object (atom nodes, bond edges, node/edge features)
- 2–3 message-passing layers (e.g. GINConv or GATConv — consistent with existing Static GAT baseline spirit)
- Readout: global mean pooling (or attention pooling) → graph embedding `h ∈ R^d`

**Pair fusion:**

- `h_pair = MLP([h_a ; h_b ; |h_a − h_b| ; h_a ⊙ h_b])`  
  (concatenation + difference + element-wise product is a standard DDI pairing pattern)

**Classifier:**

- Linear layer → **962 logits** (one per observed TWOSIDES type)

### Relation to existing Static GAT baseline

The existing Static GAT builds a **2-node graph** whose node features are **whole-molecule fingerprint vectors**, not atom-level graphs:

```python
# src/data/pyg_dataset.py — current design
x = stack([fp_a, fp_b])          # 2 nodes
edge_index = [[0,1], [1,0]]      # single interaction edge
```

For the new stage, the Static GAT pattern can be **extended** in two ways:

1. **Baseline path:** keep fingerprint 2-node graphs + adapt classifier to 962 classes.
2. **Primary GNN path:** replace node features with atom-level graphs + GNN encoder as above.

### T-GNN (future / Member A)

True temporal modeling would require:

- Timestamps or ordered admission sequences per patient
- A graph builder that connects drug events over time (`src/graph/graph_builder.py`)

None of this exists in the current CSV. **Do not implement T-GNN training on this file until temporal inputs are defined.**

---

## 5. DDI pair representation

Two drugs are represented **symmetrically at the feature level** but **ordered in the dataset** (`drug_a`, `drug_b`, `smiles_a`, `smiles_b` are already canonicalized by CID ordering in `pair_key`).

| Level | Representation |
|-------|----------------|
| **Drug A** | Molecular graph from `smiles_a` → embedding `h_a` |
| **Drug B** | Molecular graph from `smiles_b` → embedding `h_b` |
| **Pair** | Fusion of `h_a` and `h_b` (see §4) |

The existing `pair_key` should be used for **splitting and leakage checks**, not as a learned input (it is a discrete identifier tied to the same SMILES pair).

---

## 6. Target / label formulation

| Aspect | Decision |
|--------|----------|
| **Target column** | `type` (integer TWOSIDES interaction code) |
| **Task type** | Multi-class classification |
| **Number of classes** | **962** observed classes (labels 0–961 present; full TWOSIDES catalog has 963 types 0–962) |
| **Label encoding** | Map observed `type` values to class indices `{0, …, 961}`; keep a fixed `type → index` dictionary for reproducibility |
| **Negatives** | **None** in current dataset — every row is a positive TWOSIDES interaction type |
| **Rare classes** | **Do not remove** in this stage (per project instruction) |

### Label distribution issues (measured on current file)

| Statistic | Value |
|-----------|------:|
| Rows | 26,597 |
| Unique types | 962 |
| Min / median / max rows per type | 1 / 20 / 131 |
| Types with ≤ 5 rows | 112 |
| Types with ≤ 10 rows | 279 |
| Unique `(pair_key, type)` combinations | 18,841 |
| `(pair_key, type)` with multiple admission rows | 3,836 |

**Implications:**

1. **Heavy class imbalance** — macro-averaged metrics will differ sharply from micro-averaged / accuracy.
2. **Many rare types** — 112 classes have ≤ 5 training examples (after split, fewer still).
3. **Repeated rows** — the same `(smiles_a, smiles_b, type)` appears across different `(subject_id, hadm_id)` records. Features and label are identical; only admission context differs. This creates **non-independent samples**.

**Recommended training/evaluation unit (to decide at implementation):**

- **Option A (current row-level):** one sample per row — simple but duplicates inflate sample count.
- **Option B (deduplicated pair–type):** one sample per `(pair_key, type)` — 18,841 unique samples; better reflects distinct prediction cases.
- **Option C (deduplicated pair, multi-label):** one sample per `pair_key` with a set/matrix of types — different task formulation; **not** chosen unless explicitly requested.

**Recommendation:** document both row-level and deduplicated `(pair_key, type)` evaluation; use deduplicated evaluation as the **primary** generalization metric unless clinical repetition is considered part of the task.

---

## 7. Train / validation / test strategy

### Current split (already in dataset)

Splitting was performed at the **`pair_key` level** (all rows for a drug pair stay in one split):

| Split | Rows | Unique pairs | Unique types |
|-------|-----:|-------------:|-------------:|
| train | 18,709 | 101 | 955 |
| val | 3,916 | 21 | 738 |
| test | 3,972 | 23 | 843 |

Ratios: ~70 / 15 / 15 by **pair count** (101 / 21 / 23).

### Appropriateness

**Strengths:**

- Prevents the same drug pair (and therefore the same SMILES pair) from appearing in multiple splits.
- Verified: **0 pairs** span more than one split; **0 `(pair_key, type)`** spans more than one split.

**Weaknesses:**

1. **Small val/test pair holdout** — only 21–23 pairs; metrics will have high variance.
2. **Unseen classes in val/test:**
   - 6 types appear in val but not train
   - 4 types appear in test but not train  
   A standard classifier **cannot predict unseen classes**; these rows contribute only to error unless handled with a dedicated “unknown/other” bucket (not recommended without explicit approval).
3. **Row-level duplication within splits** — repeated `(pair_key, type)` across admissions may make validation metrics optimistic if duplicates are counted multiple times.
4. **Conflict with existing `pyg_dataset.py`** — that module validates **patient-level** splits (`patient_id`), not pair-level. The current CSV uses `subject_id` and pair-level splits; an adapter must be written rather than reusing `validate_patient_splits` as-is.

### Recommendation

- **Keep the existing split** for phase 1 (do not re-split without approval).
- Report **pair-level** and **deduplicated (pair_key, type)** metrics separately.
- Optionally report **seen-class-only** metrics excluding val/test types absent from train, documented as a secondary analysis — not as a substitute for full metrics.

---

## 8. Data leakage considerations

| Leakage type | Status | Notes |
|--------------|--------|-------|
| Same `pair_key` in train and test | **None** | Pair-level split enforced |
| Same `(pair_key, type)` in train and test | **None** | Verified |
| Same SMILES in train and test | **None** | Follows from pair-level split |
| Patient-level leakage | **Possible across rows within a split** | Same patient can appear in multiple rows in the same split — acceptable if pair is the split unit |
| Duplicate feature–label rows | **Present** | Same `(smiles_a, smiles_b, type)` repeated across admissions within a split |
| Label leakage via repeated pair observations | **Low risk across splits** | Pairs are split-disjoint |
| Preprocessing leakage | **None** | SMILES were not normalized or altered; graphs built from stored strings only |

---

## 9. Loss function

**Primary:** `CrossEntropyLoss` over 962 classes.

**Class imbalance handling (choose one at implementation, document choice):**

| Option | Description |
|--------|-------------|
| **Class weights** | Inverse-frequency weights in `CrossEntropyLoss` |
| **Weighted sampling** | Oversample rare `type` classes in the train loader |
| **Focal loss** | Down-weight easy examples; useful with 962 imbalanced classes |

**Not applicable:** `BCEWithLogitsLoss` (used by current Static GAT for binary labels).

---

## 10. Evaluation metrics

Existing `src/evaluation/classification.py` implements **binary** metrics only. Multi-class evaluation requires extension.

### Primary metrics (multi-class)

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness (misleading with imbalance; report alongside others) |
| **Macro F1** | Equal weight per class — highlights rare-type performance |
| **Micro F1** | Global TP/FP/FN — dominated by frequent types |
| **Balanced accuracy** | Mean per-class recall |
| **Top-k accuracy** (k=3, 5) | Useful when exact type match is strict |

### Secondary metrics

| Metric | Purpose |
|--------|---------|
| **Confusion matrix** | Inspect systematic confusions between related types |
| **Per-class precision/recall** | For frequent classes and sampled rare classes |
| **OvR macro AUROC** | One-vs-rest AUROC averaged over classes (where enough samples exist) |
| **MCC (multiclass)** | Single summary of correlation |

### Reporting splits

Report metrics on:

1. Full val/test rows
2. Deduplicated `(pair_key, type)` val/test
3. Seen-classes-only subset (optional, documented)

---

## 11. Baseline models

Use the existing baseline **infrastructure** but adapt for **962-class** outputs and **SMILES-based** inputs.

| Baseline | Existing code | Adaptation needed |
|----------|---------------|-------------------|
| **Logistic Regression** | `src/baselines/logistic_regression.py` | Build features from Morgan fingerprints of `smiles_a`/`smiles_b`; switch to multiclass LR (`multi_class='multinomial'`) |
| **Static GAT (fingerprint)** | `src/baselines/static_gat.py` | Keep 2-node graph; replace binary head with 962-class head; read SMILES from CSV |
| **MLP on concatenated fingerprints** | Not implemented | Simple strong baseline before atom-level GNN |
| **Dual GNN (proposed primary)** | Not implemented | New module in `src/models/` |

Existing baselines assume columns: `drug_a_fingerprint`, `drug_b_fingerprint`, `label`, `patient_id`. The new dataset has `smiles_a`, `smiles_b`, `type`, `subject_id` — a **dataset adapter** is required; do not silently overwrite the old schema.

---

## 12. Recommended next implementation steps

**Await approval before executing these steps.**

1. **Dataset adapter** — new PyG dataset class reading `final_mimic_twosides_ml.csv`:
   - Parse SMILES → molecular graphs (cached)
   - Emit `type` as class index
   - Respect existing `split` column
   - Do **not** modify the CSV

2. **Graph builder** — implement `src/graph/graph_builder.py`:
   - `smiles_to_pyg_data(smiles) → Data`
   - Shared featurization constants

3. **Multiclass metrics** — extend `src/evaluation/classification.py` or add `multiclass.py`

4. **Fingerprint baselines first** — adapt LR + Static GAT to multiclass + SMILES-derived fingerprints (validates pipeline end-to-end)

5. **Dual GNN model** — implement in `src/models/` (separate from Member A `tgnn.py` stub until coordinated)

6. **Evaluation protocol** — document and implement row-level vs deduplicated `(pair_key, type)` evaluation

7. **Training script** — wire `src/models/train.py` with config (seed, paths, class weights)

8. **Defer:** temporal GNN, negative sampling, label collapsing, rare-type removal, XAI/RAG/DrugBank stages

---

## Conflicts with existing project architecture

| Existing design | Current dataset | Resolution |
|-----------------|-----------------|------------|
| Binary `label` (positive/negative) | `type` with 962 classes | New multiclass head and loss; do not force binary |
| Morgan fingerprint node features | Raw SMILES in CSV | Add graph builder; keep fingerprints for baselines only |
| Patient-level split validation | Pair-level split in CSV | New split validator for `pair_key`; do not re-split |
| `patient_id` column name | `subject_id` | Map in adapter; no CSV change |
| Temporal GNN (`tgnn.py`) | No timestamps | Static GNN phase 1; temporal deferred |
| 963 TWOSIDES types (0–962) | 962 types observed | Model output size = 962; document missing type |

---

## Concise summary (for approval)

**Proposed approach:**

- **Task:** Multi-class prediction of TWOSIDES interaction `type` from two SMILES strings.
- **Inputs:** Atom-level molecular graphs built from `smiles_a` and `smiles_b` (RDKit → PyG), with SMILES-level caching.
- **Architecture:** Dual GNN encoders (one per drug) → pair fusion (concat / diff / product) → linear classifier with **962 outputs**.
- **Baselines first:** Multiclass Logistic Regression and fingerprint-based Static GAT on the same CSV, then upgrade to atom-level GNN.
- **Split:** Use existing pair-level `split` column; no pair leakage; be aware of duplicate rows and unseen val/test classes.
- **Loss:** Weighted cross-entropy (class imbalance).
- **Metrics:** Macro/micro F1, balanced accuracy, top-k accuracy, confusion matrix; report deduplicated evaluation separately.
- **Not in this stage:** training, negatives, label removal, temporal GNN, dataset modification.

**Awaiting your approval before any implementation or training.**
