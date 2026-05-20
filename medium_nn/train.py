import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from medium_nn.model import MLPRegressor
from shared.config import load_config
from shared.data_loading import load_tabular_dataset
from shared.metrics import generalisation_gap, regression_metrics
from shared.plotting import save_loss_curve, save_parity_plot, save_split_metric_plot
from shared.seed import set_seed
from shared.splits import train_val_test_split
from shared.torch_utils import (
    EarlyStopping,
    count_parameters,
    evaluate_regression,
    make_regression_loader,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medium_nn/config.yaml")
    return parser.parse_args()


def main():
    cfg = load_config(parse_args().config)
    set_seed(cfg["experiment"]["seed"])

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("TASK 2: NEURAL NETWORKS WITH AN MLP")
    print("============================================================")

    print("\nThis task explores:")
    print("- nonlinear mappings from the same descriptors used in Task 1")
    print("- optimisation over epochs")
    print("- dropout, learning rate, and early stopping")
    print("- capacity versus generalisation")

    print("\n[1/6] Loading descriptor dataset")

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

    print("\nBridge from Task 1:")
    print("The representation is unchanged. The model class is now more flexible.")

    print("\n[2/6] Creating train / validation / test splits")

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = train_val_test_split(
        X,
        y,
        metadata,
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
        cfg["experiment"]["seed"],
    )

    print(f"Train={len(X_train)} | Validation={len(X_val)} | Test={len(X_test)}")

    print("\n[3/6] Scaling input features")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("Neural networks are sensitive to feature scale.")
    print("We fit the scaler on training data only, then apply it to validation and test data.")

    print("\nExample scaled training rows")
    print("------------------------------------------------------------")
    preview = pd.DataFrame(X_train[:5], columns=features)
    preview[cfg["data"]["target_column"]] = y_train[:5]
    
    # Safely insert the molecule name column if metadata exists
    if metadata is not None and hasattr(metadata, 'shape') and metadata.shape[1] > 0:
        name_col_name = metadata.columns[0]
        names_preview = metadata.iloc[:5, 0].values
        preview.insert(0, name_col_name, names_preview)
    else:
        print("(Note: No metadata/molecule names found in configuration, displaying descriptors only)")

    print(preview.round(3).to_string(index=False))

    print("\n[4/6] Building data loaders")

    train_loader = make_regression_loader(X_train, y_train, cfg["data"]["batch_size"], shuffle=True)
    val_loader = make_regression_loader(X_val, y_val, cfg["data"]["batch_size"], shuffle=False)
    test_loader = make_regression_loader(X_test, y_test, cfg["data"]["batch_size"], shuffle=False)

    print(f"Batch size: {cfg['data']['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("A batch is a small subset of training rows used for one optimiser update.")

    print("\n[5/6] Configuring neural network")

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
        dropout_p=cfg["model"]["dropout_p"],
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    loss_fn = nn.MSELoss()

    scheduler = None
    if cfg["training"]["reduce_lr_on_plateau"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=cfg["training"]["lr_factor"],
            patience=cfg["training"]["lr_patience"],
        )

    stopper = EarlyStopping(cfg["training"]["patience"])

    hidden_dims = cfg["model"]["hidden_dims"]

    print("\nNetwork architecture:")

    input_dim = X_train.shape[1]
    print(f"Input layer: {input_dim} descriptor features")

    for i, hidden_dim in enumerate(hidden_dims, start=1):
        print(f"Hidden layer {i}: {hidden_dim} neurons")

    print("Output layer: 1 prediction value")

    print("\nArchitecture guide:")
    print("- Each hidden layer learns intermediate nonlinear representations.")
    print("- More neurons increase model capacity.")
    print("- More layers allow more complex transformations.")
    print("- Larger networks may fit training data better, but can overfit.")

    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Dropout: {'ON' if cfg['model']['dropout'] else 'OFF'}")
    print(f"Learning rate: {cfg['training']['learning_rate']}")
    print(f"Early stopping: {'ON' if cfg['training']['early_stopping'] else 'OFF'}")
    print(f"ReduceLROnPlateau: {'ON' if scheduler else 'OFF'}")

    print("\nParameter guide:")
    print("- hidden_dims controls network capacity.")
    print("  Wider or deeper networks can learn more complex relationships, but may overfit.")
    print("- learning_rate controls the size of each optimiser step.")
    print("  Too small can learn slowly; too large can make training unstable.")
    print("- dropout randomly switches off hidden units during training.")
    print("  This makes the model less reliant on any single pathway and can reduce overfitting.")
    print("- early stopping stops training when validation loss stops improving.")
    print("  This prevents the model from continuing to fit training noise.")

    print("\n[6/6] Training model")

    history = []
    start_total = time.perf_counter()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        epoch_start = time.perf_counter()

        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimiser.step()

            train_losses.append(loss.item() * len(X_batch))

        train_loss = sum(train_losses) / len(train_loader.dataset)
        val_loss, _, _ = evaluate_regression(model, val_loader, loss_fn)

        if scheduler:
            scheduler.step(val_loss)

        lr = optimiser.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch_time_s": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={lr:.5f} | "
            f"time={epoch_time:.2f}s"
        )

        should_stop = stopper.step(val_loss, model)

        if cfg["training"]["early_stopping"] and should_stop:
            print("Early stopping triggered.")
            break

    stopper.restore(model)
    total_time = time.perf_counter() - start_total

    _, y_train_true, y_train_pred = evaluate_regression(model, train_loader, loss_fn)
    _, y_val_true, y_val_pred = evaluate_regression(model, val_loader, loss_fn)
    _, y_test_true, y_test_pred = evaluate_regression(model, test_loader, loss_fn)

    metrics = {
        "train": regression_metrics(y_train_true, y_train_pred),
        "validation": regression_metrics(y_val_true, y_val_pred),
        "test": regression_metrics(y_test_true, y_test_pred),
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

    print("\nExample test predictions")
    print("------------------------------------------------------------")
    prediction_preview = pd.DataFrame(X_test[:8], columns=features)
    
    if metadata is not None and hasattr(metadata, 'shape') and metadata.shape[1] > 0:
        name_col_name = metadata.columns[0]
        names_test_preview = metadata.iloc[:8, 0].values
        prediction_preview.insert(0, name_col_name, names_test_preview)

    prediction_preview["measured"] = y_test_true[:8]
    prediction_preview["predicted"] = y_test_pred[:8]
    prediction_preview["error"] = prediction_preview["predicted"] - prediction_preview["measured"]

    print(prediction_preview.round(3).to_string(index=False))

    print(
        "Generalisation gap, validation RMSE - train RMSE: "
        f"{generalisation_gap(metrics['train']['rmse'], metrics['validation']['rmse']):.3f}"
    )

    print(f"Total training time: {total_time:.2f}s")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(output_dir / "metrics.csv")
    pd.DataFrame({"y_true": y_test_true, "y_pred": y_test_pred}).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    save_loss_curve(
        history,
        output_dir / "loss_curve.png",
        title="Task 2: Train vs validation loss",
        ylabel="MSE loss",
    )

    save_split_metric_plot(
        metrics,
        output_dir / "split_rmse.png",
        title="Task 2: Train vs validation vs test error",
        ylabel="RMSE",
    )

    save_parity_plot(
        y_test_true,
        y_test_pred,
        output_dir / "parity_plot.png",
        title="Task 2: Test-set solubility predictions",
        xlabel="Measured solubility",
        ylabel="Predicted solubility",
        label="Test molecules",
    )

    print(f"\nOutputs saved to: {output_dir}")

    print("\nNext steps:")
    print("- Change hidden_dims and compare training time and validation RMSE.")
    print("- Turn dropout on and inspect the train-validation gap.")
    print("- Increase learning_rate and watch for unstable loss values.")
    print("- Compare Task 2 against the Random Forest from Task 1.")


if __name__ == "__main__":
    main()