"""Load JSON training configuration with optional env-based overrides."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_training_config(path: str | Path) -> dict[str, Any]:
    """
    Load a training config JSON file.

    If ``mlflow.tracking_uri_env`` is set, the tracking URI is taken from
    that environment variable (falls back to empty string if unset).
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    mlflow_cfg = raw.get("mlflow") or {}
    env_key = mlflow_cfg.get("tracking_uri_env")
    if env_key:
        mlflow_cfg = {**mlflow_cfg, "tracking_uri": os.environ.get(env_key, "")}
        raw["mlflow"] = mlflow_cfg

    return raw
