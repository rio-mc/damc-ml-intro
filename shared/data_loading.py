from pathlib import Path
import pandas as pd


def load_tabular_dataset(
    path: str | Path,
    target_column: str,
    feature_columns: list[str],
    drop_missing: bool = True,
    metadata_columns: list[str] | None = None,
):
    metadata_columns = metadata_columns or []

    if target_column in feature_columns:
        raise ValueError(
            f"Target leakage: '{target_column}' is listed as both a feature and the target."
        )

    df = pd.read_csv(path)

    required = feature_columns + [target_column] + metadata_columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[required]

    if drop_missing:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
    else:
        dropped = 0

    X = df[feature_columns].to_numpy(dtype="float32")
    y = df[target_column].to_numpy(dtype="float32")
    metadata = df[metadata_columns].reset_index(drop=True)

    return X, y, feature_columns, metadata, dropped