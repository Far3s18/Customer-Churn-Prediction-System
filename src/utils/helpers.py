from __future__ import annotations

import os
import json
import mlflow
import numpy as np

from typing import Any


def log_metrics(prefix: str, metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log a dictionary of metrics to MLflow using a common prefix.
    """

    for metric_name, value in metrics.items():
        mlflow.log_metric(
            key=f"{prefix}_{metric_name}",
            value=float(value),
            step=step,
        )


def make_json_serializable(value: Any) -> Any:
    """
    Convert NumPy and nested objects into JSON-serializable Python objects.
    """

    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return tuple(make_json_serializable(item) for item in value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value