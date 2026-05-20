import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.config import load_config
from shared.data_loading import load_tabular_dataset
from shared.metrics import generalisation_gap, regression_metrics
from shared.plotting import (
    save_feature_importance_plot,
    save_parity_plot,
    save_split_metric_plot,
)
from shared.seed import set_seed
from shared.splits import train_val_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="easy_ml/config.yaml")
    return parser.parse_args()


def main():
    cfg = load_config(parse_args().config)
    set_seed(cfg["experiment"]["seed"])

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("TASK 1: CLASSICAL MACHINE LEARNING WITH RANDOM FORESTS")
    print("============================================================")

    print("\nThis task explores:")
    print("- handcrafted molecular descriptors")
    print("- model capacity and overfitting")
    print("- train / validation / test behaviour")
    print("- feature importance")

    print("\n[1/5] Loading dataset")

    X, y, features, metadata, dropped = load_tabular_dataset(
        path=cfg["data"]["path"],
        target_column=cfg["data"]["target_column"],
        feature_columns=cfg["features"]["columns"],
        drop_missing=cfg["data"]["drop_missing"],
        metadata_columns=cfg["data"].get("metadata_columns", []),
    )

    print(f"Dataset path: {cfg['data']['path']}")
    print(f"Target property: {cfg['data']['target_column']}")
    print(f"Selected features ({len(features)}): {features}")
    print(f"Rows dropped due to missing values: {dropped}")

    print("\nThink:")
    print("Which descriptors do you expect to be most important?")
    print("Would adding more descriptors always improve generalisation?")

    print("\n[2/5] Creating train / validation / test splits")

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = train_val_test_split(
        X,
        y,
        metadata,
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
        cfg["experiment"]["seed"],
    )

    print(
        f"Train={len(X_train)} | "
        f"Validation={len(X_val)} | "
        f"Test={len(X_test)}"
    )

    print("\nExample training rows")
    print("------------------------------------------------------------")
    print("These are examples of the descriptor values the model uses during training.")
    print("Each row is one molecule represented only by the selected features.\n")

    preview = pd.DataFrame(X_train[:5], columns=features)
    preview[cfg["data"]["target_column"]] = y_train[:5]
    
    # Safely check if meta_train exists, is a DataFrame, and actually has columns
    if meta_train is not None and hasattr(meta_train, 'columns') and len(meta_train.columns) > 0:
        name_col_name = meta_train.columns[0]
        names_preview = meta_train.iloc[:5, 0].values
        preview.insert(0, name_col_name, names_preview)
    elif meta_train is not None and len(meta_train) > 0:
        # Fallback if meta_train is a simple list or numpy array instead of a DataFrame
        preview.insert(0, "Molecule Name", meta_train[:5])

    print(preview.round(3).to_string(index=False))

    print("\nReminder:")
    print("- Train data updates model parameters.")
    print("- Validation data guides model decisions.")
    print("- Test data estimates final generalisation.")

    print("\n[3/5] Configuring random forest")

    print("\nModel parameters:")
    print(f"n_estimators      = {cfg['model']['n_estimators']}")
    print(f"max_depth         = {cfg['model']['max_depth']}")
    print(f"min_samples_leaf  = {cfg['model']['min_samples_leaf']}")

    print("\nParameter guide:")
    print("- n_estimators:")
    print("  The number of decision trees in the forest.")
    print("  More trees usually make predictions more stable, but increase training time.")

    print("- max_depth:")
    print("  The maximum number of split decisions each tree can make from root to leaf.")
    print("  Deeper trees can fit more detailed patterns, but may also memorise noise.")

    print("- min_samples_leaf:")
    print("  The minimum number of training examples allowed in a final leaf node.")
    print("  A leaf is where a tree makes its final prediction.")
    print("  If this value is very small, a tree can create leaves for tiny groups of points.")
    print("  That can fit quirks in the training set rather than patterns that generalise.")
    print("  Increasing it forces each prediction to be based on more examples, which usually")
    print("  makes the tree smoother and less prone to overfitting.")

    model = RandomForestRegressor(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        min_samples_leaf=cfg["model"]["min_samples_leaf"],
        random_state=cfg["experiment"]["seed"],
        n_jobs=-1,
    )

    print("\n[4/5] Training model")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    print(f"Training completed in {train_time:.2f} seconds")

    print("\n[5/5] Evaluating model")

    predictions = {
        "train": model.predict(X_train),
        "validation": model.predict(X_val),
        "test": model.predict(X_test),
    }

    print("\nExample test predictions")
    print("------------------------------------------------------------")
    print("These molecules were not used to train the model.")
    print("The model receives only the descriptors, then predicts the target property.\n")

    prediction_preview = pd.DataFrame(X_test[:8], columns=features)
    
    # Insert the compound names for the test subset
    if meta_test is not None and len(meta_test) > 0:
        name_col_name = metadata.columns[0] if hasattr(metadata, 'columns') else "Compound Name"
        names_test_preview = meta_test.iloc[:8].iloc[:, 0].values if hasattr(meta_test, 'iloc') else meta_test[:8]
        prediction_preview.insert(0, name_col_name, names_test_preview)

    prediction_preview["measured"] = y_test[:8]
    prediction_preview["predicted"] = predictions["test"][:8]
    prediction_preview["error"] = prediction_preview["predicted"] - prediction_preview["measured"]

    print(prediction_preview.round(3).to_string(index=False))
    prediction_preview.to_csv(
        output_dir / "prediction_examples.csv",
        index=False,
    )

    metrics = {
        "train": regression_metrics(y_train, predictions["train"]),
        "validation": regression_metrics(y_val, predictions["validation"]),
        "test": regression_metrics(y_test, predictions["test"]),
    }

    print("\nMetrics")
    print("------------------------------------------------------------")

    for split, values in metrics.items():
        print(
            f"{split:10s} | "
            f"RMSE={values['rmse']:.3f} | "
            f"MAE={values['mae']:.3f} | "
            f"R2={values['r2']:.3f}"
        )

    gap = generalisation_gap(
        metrics["train"]["rmse"],
        metrics["validation"]["rmse"],
    )

    print("\nGeneralisation gap")
    print("------------------------------------------------------------")
    print(f"Validation RMSE - Train RMSE = {gap:.3f}")

    print("\nInterpretation:")
    print("- Train RMSE measures how well the model fits data it learned from.")
    print("- Validation RMSE measures how well it performs on data not used to fit the trees.")
    print("- If train RMSE is much lower than validation RMSE, the model may be overfitting.")
    print("- If both train and validation RMSE are high, the model may be underfitting.")

    pd.DataFrame(metrics).T.to_csv(output_dir / "metrics.csv")

    pd.DataFrame(
        {
            "split": "test",
            "y_true": y_test,
            "y_pred": predictions["test"],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    save_split_metric_plot(
        metrics,
        output_dir / "split_rmse.png",
        metric_name="rmse",
        title="Task 1: Train vs validation vs test error",
        ylabel="RMSE",
    )

    feature_importances = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importances.to_csv(
        output_dir / "feature_importances.csv",
        index=False,
    )

    save_feature_importance_plot(
        feature_importances,
        output_dir / "feature_importance.png",
        title="Task 1: Random Forest feature importance",
        xlabel="Relative importance",
        ylabel="Descriptor",
    )

    save_parity_plot(
        y_test,
        predictions["test"],
        output_dir / "parity_plot.png",
        title="Task 1: Test-set solubility predictions",
        xlabel="Measured solubility",
        ylabel="Predicted solubility",
        label="Test molecules",
    )

    print(f"\nOutputs saved to: {output_dir}")

    print("\nNext steps:")
    print("- Add one descriptor at a time and check whether validation RMSE improves.")
    print("- Increase max_depth and watch whether train RMSE improves more than validation RMSE.")
    print("- Increase min_samples_leaf and check whether the train-validation gap shrinks.")
    print("- Inspect feature_importances.csv and ask whether the ranking makes chemical sense.")


if __name__ == "__main__":
    main()