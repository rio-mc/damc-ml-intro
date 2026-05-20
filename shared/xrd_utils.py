from pathlib import Path
import numpy as np
import pandas as pd


def load_xrd_npz(path: str | Path):
    data = np.load(path)

    X = data["X"].astype("float32")
    y = data["y"]

    if y.dtype.kind not in {"i", "u"}:
        y = y.astype("float32")

    return X, y


def normalise_xrd_intensity(X, eps: float = 1e-8):
    max_intensity = X.max(axis=1, keepdims=True)
    return X / (max_intensity + eps)


def save_xrd_preview(X, y, path: str | Path, n: int = 5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preview = pd.DataFrame(X[:n])
    preview.insert(0, "target", y[:n])
    preview.to_csv(path, index=False)

