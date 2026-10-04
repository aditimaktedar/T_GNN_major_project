"""Temporal graph neural network model definitions and model_v2 exports."""

try:
    from src.model import TemporalDDIGNN, SafeGraphConv
except ImportError:
    TemporalDDIGNN = None
    SafeGraphConv = None

from src.inference import (
    load_checkpoint,
    predict_pair,
    load_checkpoint_metadata,
    StandaloneInferenceEngine,
    SEVERITY_CLASSES
)

__all__ = [
    "TemporalDDIGNN",
    "SafeGraphConv",
    "load_checkpoint",
    "predict_pair",
    "load_checkpoint_metadata",
    "StandaloneInferenceEngine",
    "SEVERITY_CLASSES"
]
