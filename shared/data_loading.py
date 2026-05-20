from pathlib import Path
import pandas as pd


def load_tabular_dataset(
    path: str | Path,
    target_column: str,
    feature_columns: list[str],
    drop_missing: bool = True,
):
    df = pd.read_csv(path)

    required = feature_columns + [target_column]
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

    return X, y, feature_columns, dropped