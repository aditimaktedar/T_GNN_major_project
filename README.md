# TemporalDDI-GNN - model_v2

Standalone T-GNN package for the model_v2 branch.

## Architecture

Drug A/B -> drug embeddings -> 2 SafeGraphConv layers
-> residual node representation -> pair representation
-> shared MLP -> Presence + Severity heads.

Pair representation:

[hA, hB, hA*hB, |hA-hB|, hA+hB]

## Presence

- Dataset: MIMIC-IV 2.1
- Labels: TWOSIDES
- Temporal window: 24 hours
- Binary classes: No / Yes

The strict unseen-drug experiment is the important cold-start
generalization test.

## Severity

- Dataset: MIMIC-IV Demo 2.2
- Classes: Minor / Moderate / Major
- Evaluation: 5-fold patient-disjoint CV

## Important limitation

Presence and Severity were evaluated on different MIMIC cohorts.
Therefore this repository does not claim one end-to-end jointly
trained multitask checkpoint.

The architecture contains both heads, but the reported experiments
are task-specific.

## RAG/API

RAG is intentionally outside the numerical T-GNN model.

Recommended flow:

API -> drug resolver -> T-GNN -> prediction -> RAG -> response

RAG evidence must remain separate from the numerical prediction.

See api/rag_contract.py.

## Important

A checkpoint requires its exact drug-ID mapping and graph.
Do not silently create a new drug-index mapping at inference.

## Data

Raw MIMIC, TWOSIDES and huge preprocessing artifacts are excluded.

## Interpretation

TWOSIDES presence does not establish patient-level causal
adverse-event relationships.
