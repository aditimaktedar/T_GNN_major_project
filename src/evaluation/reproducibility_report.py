"""Generate reproducibility report for Member B experiments."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scipy

from src.data.config import METRICS_DIR, PROJECT_ROOT
from src.data.io_utils import write_json
from src.evaluation.reproducibility import collect_environment_metadata


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_reproducibility_report(
    config: dict[str, Any] | None = None,
    dataset_paths: dict[str, str] | None = None,
    dataset_sources: list[str] | None = None,
    synthetic_demo: bool = False,
) -> dict[str, Any]:
    metadata = collect_environment_metadata(config)
    report = {
        **metadata,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "os": platform.system(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_file": str(PROJECT_ROOT / "configs" / "baselines.json"),
        "git_commit": _git_commit(),
        "dataset_paths": dataset_paths or {},
        "dataset_sources": dataset_sources or [],
        "synthetic_demo": synthetic_demo,
        "synthetic_vs_real_status": (
            "SYNTHETIC SOFTWARE VALIDATION ONLY" if synthetic_demo else "real_data_mode"
        ),
        "credentials_included": False,
    }
    return report


def save_reproducibility_report(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    path = output_path or METRICS_DIR / "reproducibility.json"
    write_json(path, report)
    return path
