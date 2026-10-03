# Deployment Notes

The model requires the exact training-time drug-ID mapping.

The graph construction must also match the checkpoint.

RAG should retrieve evidence independently from T-GNN.

API responses should distinguish:
- model prediction
- severity probabilities
- retrieved evidence
- evidence sources
- limitations

The current severity checkpoints are CV fold checkpoints.
They are not represented as one production checkpoint.
