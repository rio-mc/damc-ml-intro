from sklearn.model_selection import train_test_split


def train_val_test_split(X, y, train_fraction: float, val_fraction: float, test_fraction: float, seed: int):
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=train_fraction, random_state=seed
    )

    val_ratio = val_fraction / (val_fraction + test_fraction)

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_ratio, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test