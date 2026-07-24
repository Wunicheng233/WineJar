"""Persistent base-frame correction for fixed-camera visual targets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_bias_mm(path: Path) -> np.ndarray:
    """Return the configured XYZ correction, or zero when none has been measured."""
    if not path.exists():
        return np.zeros(3, dtype=float)
    data = json.loads(path.read_text(encoding="utf-8"))
    bias = np.asarray(data["bias_base_mm"], dtype=float)
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError(f"Invalid positioning bias in {path}")
    return bias


def append_bias_sample(path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    """Append one manual alignment and recompute a robust median correction."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = list(data.get("samples", []))
    else:
        samples = []
    samples.append(sample)

    corrections = np.asarray([item["correction_base_mm"] for item in samples], dtype=float)
    if corrections.ndim != 2 or corrections.shape[1] != 3:
        raise ValueError("Positioning-bias samples must contain three-axis corrections")
    median = np.median(corrections, axis=0)
    spread = np.max(corrections, axis=0) - np.min(corrections, axis=0)
    result = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "description": (
            "Base-frame correction added to fixed-camera visual targets. "
            "Measured by manually aligning the physical suction center over detected targets."
        ),
        "sample_count": len(samples),
        "bias_base_mm": median.tolist(),
        "sample_range_base_mm": spread.tolist(),
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result
