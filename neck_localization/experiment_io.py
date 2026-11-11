"""Small offline experiment IO without renderer/ML/control imports."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, default=encode, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    v=np.asarray(values)
    return dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
