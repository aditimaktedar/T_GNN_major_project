"""Multi-label ablation comparison framework.

Structures age ablation experiments (Config A: drug_pair vs Config B: drug_pair_age)
across all model architectures. Does NOT execute training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationConfig:
    """Configuration for one ablation experiment run."""
    model_name: str
    feature_set: str  # 'drug_pair' or 'drug_pair_age'
    n_classes: int = 363
    label_space: str = "FREQUENT363"
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 64
    threshold: float = 0.5
    notes: str = ""


@dataclass
class AblationResult:
    """Result from one ablation experiment run (placeholder for future use)."""
    config: AblationConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"  # PENDING | RUNNING | COMPLETED | FAILED


def generate_age_ablation_configs(
    models: list[str] | None = None,
    n_classes: int = 363,
    seed: int = 42,
) -> list[AblationConfig]:
    """Generate matched pairs of configs for age ablation study.

    For each model, generates two configs:
    - Config A: feature_set='drug_pair' (SMILES/fingerprints only)
    - Config B: feature_set='drug_pair_age' (SMILES/fingerprints + normalized age)

    Parameters
    ----------
    models : list[str], optional
        Model names to include. Defaults to all three architectures.
    n_classes : int
        Number of output classes.
    seed : int
        Random seed.

    Returns
    -------
    list[AblationConfig]
        Ordered list of ablation configs (A then B for each model).
    """
    if models is None:
        models = [
            "MultilabelLogisticRegression",
            "MultilabelStaticGAT",
            "MultilabelMolecularGNN",
        ]

    configs = []
    for model_name in models:
        # Config A: drug_pair only
        configs.append(AblationConfig(
            model_name=model_name,
            feature_set="drug_pair",
            n_classes=n_classes,
            seed=seed,
            notes="Age Ablation Config A: Drug pair features only.",
        ))
        # Config B: drug_pair + age
        configs.append(AblationConfig(
            model_name=model_name,
            feature_set="drug_pair_age",
            n_classes=n_classes,
            seed=seed,
            notes="Age Ablation Config B: Drug pair features + normalized patient age.",
        ))

    return configs


def format_ablation_comparison(
    results: list[AblationResult],
    primary_metric: str = "micro_f1",
) -> list[dict[str, Any]]:
    """Format completed ablation results into a comparison table.

    Parameters
    ----------
    results : list[AblationResult]
        Completed ablation results.
    primary_metric : str
        Primary metric for comparison.

    Returns
    -------
    list[dict]
        Formatted comparison rows.
    """
    rows = []
    for r in results:
        row = {
            "model": r.config.model_name,
            "feature_set": r.config.feature_set,
            "status": r.status,
            "n_classes": r.config.n_classes,
            "seed": r.config.seed,
        }
        if r.metrics:
            row[primary_metric] = r.metrics.get(primary_metric, None)
            row["hamming_loss"] = r.metrics.get("hamming_loss", None)
            row["mAP"] = r.metrics.get("mAP", None)
        rows.append(row)
    return rows
