"""Unified experiment configurations for multi-label DDI experiments.

Defines model, feature set, task, label space, split, and random seed
configurations for reproducible experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for a single multi-label DDI experiment."""

    # Model
    model_name: str
    model_class: str  # Python import path for the model class

    # Task
    task: str = "multilabel_classification"
    label_space: str = "FREQUENT363"
    n_classes: int = 363

    # Features
    feature_set: str = "drug_pair"  # 'drug_pair' or 'drug_pair_age'
    fingerprint_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048

    # Data
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.2
    threshold: float = 0.5
    loss_fn: str = "BCEWithLogitsLoss"

    # Architecture-specific
    hidden_dim: int = 64
    num_layers: int = 3
    heads: int = 4
    fusion_hidden: int = 128

    # Metadata
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- Pre-defined experiment configurations ----

MULTILABEL_LR_DRUG_PAIR = ExperimentConfig(
    model_name="MultilabelLogisticRegression",
    model_class="src.models.multilabel_logistic_regression.MultilabelLogisticRegression",
    feature_set="drug_pair",
    notes="Baseline LR on Morgan FP pair features only.",
)

MULTILABEL_LR_DRUG_PAIR_AGE = ExperimentConfig(
    model_name="MultilabelLogisticRegression",
    model_class="src.models.multilabel_logistic_regression.MultilabelLogisticRegression",
    feature_set="drug_pair_age",
    notes="LR on Morgan FP pair features + normalized patient age.",
)

MULTILABEL_GAT_DRUG_PAIR = ExperimentConfig(
    model_name="MultilabelStaticGAT",
    model_class="src.models.multilabel_static_gat.MultilabelStaticGAT",
    feature_set="drug_pair",
    notes="Static GAT on drug pair graphs, no age.",
)

MULTILABEL_GAT_DRUG_PAIR_AGE = ExperimentConfig(
    model_name="MultilabelStaticGAT",
    model_class="src.models.multilabel_static_gat.MultilabelStaticGAT",
    feature_set="drug_pair_age",
    notes="Static GAT on drug pair graphs + normalized age.",
)

MULTILABEL_GNN_DRUG_PAIR = ExperimentConfig(
    model_name="MultilabelMolecularGNN",
    model_class="src.models.multilabel_molecular_gnn.MultilabelMolecularGNN",
    feature_set="drug_pair",
    notes="Dual Molecular GNN (GIN) on molecular graphs, no age.",
)

MULTILABEL_GNN_DRUG_PAIR_AGE = ExperimentConfig(
    model_name="MultilabelMolecularGNN",
    model_class="src.models.multilabel_molecular_gnn.MultilabelMolecularGNN",
    feature_set="drug_pair_age",
    notes="Dual Molecular GNN (GIN) on molecular graphs + normalized age.",
)


ALL_EXPERIMENTS: list[ExperimentConfig] = [
    MULTILABEL_LR_DRUG_PAIR,
    MULTILABEL_LR_DRUG_PAIR_AGE,
    MULTILABEL_GAT_DRUG_PAIR,
    MULTILABEL_GAT_DRUG_PAIR_AGE,
    MULTILABEL_GNN_DRUG_PAIR,
    MULTILABEL_GNN_DRUG_PAIR_AGE,
]


def get_experiment_by_name(
    model_name: str, feature_set: str = "drug_pair"
) -> ExperimentConfig | None:
    """Look up an experiment config by model name and feature set."""
    for exp in ALL_EXPERIMENTS:
        if exp.model_name == model_name and exp.feature_set == feature_set:
            return exp
    return None
