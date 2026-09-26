# XAI Evaluation — Operational Definitions

These are **our implementation choices** for TemporalDDI-GNN Member B. They are model-agnostic and work with explanation masks supplied by Member A.

## Input schema

Explanation files must provide at minimum:

| Field | Meaning |
| --- | --- |
| `patient_id` | Patient identifier |
| `drug_a`, `drug_b` | Drug pair |
| `probability` | Original model output probability |
| `masked_probability` | Model output after explanation mask is applied |
| `explanation_mask` | Binary or numeric mask over explanation elements |

Optional: `important_features`, `important_nodes`, `important_edges`, `prediction`.

Member A's current `src/xai/explain.py` is a **stub**. Real evaluation requires explanation files from Member A once implemented.

## Fidelity

For each example with original probability \(p\) and masked probability \(p'\):

\[
\text{Fidelity} = \max\left(0,\ 1 - |p - p'|\right)
\]

**Interpretation:** Measures how much model behavior is preserved when only explanation-selected information is retained (or removed, depending on Member A's protocol). A value of 1 means identical output; 0 means completely different output.

Aggregate: mean fidelity across the evaluation set.

## Sparsity

For explanation mask \(m \in \{0,1\}^d\) (or thresholded numeric mask):

\[
\text{Sparsity} = 1 - \frac{1}{d}\sum_{i=1}^{d} m_i
\]

**Interpretation:** Proportion of available explanation elements **not** selected. Higher sparsity = smaller explanation relative to full input.

Aggregate: mean sparsity across examples.

## Stability

For two explanation masks \(m^{(1)}\) and \(m^{(2)}\) (e.g. original run vs perturbed/repeated run):

\[
\text{Stability} = 1 - \frac{1}{d}\sum_{i=1}^{d} \mathbb{1}[m^{(1)}_i \neq m^{(2)}_i]
\]

**Interpretation:** Consistency of explanations under repeated or perturbed evaluation. 1 = identical masks; 0 = completely different.

Aggregate: mean stability where repeated explanations are supplied.

## Demo mode

`python -m src.evaluation.run_evaluation --demo` uses **synthetic explanation objects** labeled `SYNTHETIC SOFTWARE VALIDATION ONLY`. These are not research results.

## References

Implementation: `src/evaluation/xai_metrics.py`, `src/evaluation/run_xai_evaluation.py`
