from sklearn.model_selection import train_test_split


def train_val_test_split(X, y, metadata, train_fraction: float, val_fraction: float, test_fraction: float, seed: int):
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    # Pass metadata into the first split
    X_train, X_tmp, y_train, y_tmp, meta_train, meta_tmp = train_test_split(
        X, y, metadata, train_size=train_fraction, random_state=seed
    )

    val_ratio = val_fraction / (val_fraction + test_fraction)

    # Pass the remaining metadata temp pool into the final split
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_tmp, y_tmp, meta_tmp, train_size=val_ratio, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test