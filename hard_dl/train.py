import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hard_dl.augment import augment_xrd
from hard_dl.model import XRDCNN
from shared.config import load_config
from shared.plotting import (
    save_loss_curve,
    save_split_metric_plot,
    save_xrd_seen_unseen_plot,
)
from shared.seed import set_seed
from shared.splits import train_val_test_split
from shared.torch_utils import (
    EarlyStopping,
    count_parameters,
    make_regression_loader,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hard_dl/config.yaml")
    return parser.parse_args()


def load_xrd_dataset(path, normalise_intensity: bool):
    data = np.load(path, allow_pickle=True)

    X = data["X"].astype("float32")
    y = data["y"]

    if normalise_intensity:
        X = X / (X.max(axis=1, keepdims=True) + 1e-8)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y).astype("int64")

    theta = data["theta"] if "theta" in data.files else None

    return X, y_encoded, encoder.classes_, theta


def make_classification_loader(X, y, batch_size: int, shuffle: bool):
    X = torch.tensor(X[:, None, :], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_classification(model, loader, loss_fn):
    model.eval()

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            losses.append(loss.item() * len(X_batch))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)

    loss = sum(losses) / len(loader.dataset)
    accuracy = float((y_true == y_pred).mean())

    return loss, accuracy, y_true, y_pred


def main():
    cfg = load_config(parse_args().config)
    set_seed(cfg["experiment"]["seed"])

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("TASK 3: DEEP LEARNING FROM XRD PATTERNS")
    print("============================================================")

    print("\nThis task explores:")
    print("- raw diffraction signals")
    print("- learned representations")
    print("- convolutional neural networks")
    print("- robustness and generalisation")

    print("\n[1/6] Loading XRD dataset")

    X, y, class_names, theta = load_xrd_dataset(
        path=cfg["data"]["path"],
        normalise_intensity=cfg["data"]["normalise_intensity"],
    )

    print(f"Dataset path: {cfg['data']['path']}")
    print(f"X shape: {X.shape}")
    print(f"Classes: {list(class_names)}")
    print(f"Number of classes: {len(class_names)}")

    print("\nBridge from Tasks 1 and 2:")
    print("Tasks 1 and 2 used human-designed molecular descriptors.")
    print("Task 3 gives the model a diffraction signal directly.")
    print("The CNN must learn useful peak-pattern representations itself.")

    print("\n[2/6] Creating train / validation / test splits")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
        cfg["experiment"]["seed"],
    )

    print(f"Train={len(X_train)} | Validation={len(X_val)} | Test={len(X_test)}")

    print("\nExample XRD pattern")
    print("------------------------------------------------------------")
    print("The model sees an intensity vector, not a descriptor table.")
    print(f"One pattern has {X_train.shape[1]} intensity values.")
    print(f"Class label: {class_names[y_train[0]]}")

    example = X_train[0]
    top_peak_indices = np.argsort(example)[-8:][::-1]

    if theta is not None:
        top_peaks = pd.DataFrame(
            {
                "two_theta": theta[top_peak_indices],
                "intensity": example[top_peak_indices],
            }
        )
        print("\nStrongest peaks in this example:")
        print(top_peaks.round(3).to_string(index=False))
    else:
        top_peaks = pd.DataFrame(
            {
                "grid_index": top_peak_indices,
                "intensity": example[top_peak_indices],
            }
        )
        print("\nStrongest peaks in this example:")
        print(top_peaks.round(3).to_string(index=False))

    print("\n[3/6] Building data loaders")

    train_loader = make_classification_loader(
        X_train,
        y_train,
        cfg["data"]["batch_size"],
        shuffle=True,
    )

    val_loader = make_classification_loader(
        X_val,
        y_val,
        cfg["data"]["batch_size"],
        shuffle=False,
    )

    test_loader = make_classification_loader(
        X_test,
        y_test,
        cfg["data"]["batch_size"],
        shuffle=False,
    )

    print(f"Batch size: {cfg['data']['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")

    print("\n[4/6] Configuring CNN")

    model = XRDCNN(
        n_points=X.shape[1],
        n_classes=len(class_names),
        channels=cfg["model"]["channels"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        dropout_p=cfg["model"]["dropout_p"],
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    loss_fn = nn.CrossEntropyLoss()

    scheduler = None
    if cfg["training"]["reduce_lr_on_plateau"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=cfg["training"]["lr_factor"],
            patience=cfg["training"]["lr_patience"],
        )

    stopper = EarlyStopping(cfg["training"]["patience"])

    print(f"Channels: {cfg['model']['channels']}")
    print(f"Kernel size: {cfg['model']['kernel_size']}")
    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Dropout: {'ON' if cfg['model']['dropout'] else 'OFF'}")
    print(f"Augmentation: {'ON' if cfg['augmentation']['enabled'] else 'OFF'}")
    print(f"ReduceLROnPlateau: {'ON' if scheduler else 'OFF'}")

    print("\nParameter guide:")
    print("- channels controls how many learned filters each convolution layer has.")
    print("  More channels can learn more peak motifs, but increase capacity and runtime.")
    print("- kernel_size controls the local 2θ window each filter sees.")
    print("  Small kernels focus on narrow local patterns; larger kernels see broader peak regions.")
    print("- dropout randomly switches off part of the classifier during training.")
    print("  This can reduce overfitting in small scientific datasets.")
    print("- augmentation adds noise or small shifts to training patterns.")
    print("  This tests whether the CNN learns robust diffraction features.")

    print("\n[5/6] Training CNN")

    history = []
    start_total = time.perf_counter()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        epoch_start = time.perf_counter()

        for X_batch, y_batch in train_loader:
            if cfg["augmentation"]["enabled"]:
                X_batch = augment_xrd(
                    X_batch,
                    noise_std=cfg["augmentation"]["noise_std"],
                    shift_max=cfg["augmentation"]["shift_max"],
                )

            optimiser.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimiser.step()

            train_losses.append(loss.item() * len(X_batch))

        train_loss = sum(train_losses) / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate_classification(model, val_loader, loss_fn)

        if scheduler:
            scheduler.step(val_loss)

        lr = optimiser.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": lr,
                "epoch_time_s": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.3f} | "
            f"lr={lr:.5f} | "
            f"time={epoch_time:.2f}s"
        )

        should_stop = stopper.step(val_loss, model)

        if cfg["training"]["early_stopping"] and should_stop:
            print("Early stopping triggered.")
            break

    stopper.restore(model)
    total_time = time.perf_counter() - start_total

    print("\n[6/6] Evaluating final model")

    train_loss, train_acc, y_train_true, y_train_pred = evaluate_classification(
        model,
        train_loader,
        loss_fn,
    )

    val_loss, val_acc, y_val_true, y_val_pred = evaluate_classification(
        model,
        val_loader,
        loss_fn,
    )

    test_loss, test_acc, y_test_true, y_test_pred = evaluate_classification(
        model,
        test_loader,
        loss_fn,
    )

    metrics = {
        "train": {"loss": train_loss, "accuracy": train_acc},
        "validation": {"loss": val_loss, "accuracy": val_acc},
        "test": {"loss": test_loss, "accuracy": test_acc},
    }

    # Choose a correct and uncorrect prediction to visualise the CNN's performance on seen vs unseen patterns.

    train_correct = np.where(y_train_true == y_train_pred)[0]
    test_correct = np.where(y_test_true == y_test_pred)[0]
    test_incorrect = np.where(y_test_true != y_test_pred)[0]

    def find_train_example_for_class(target_class):
        matches = train_correct[y_train_true[train_correct] == target_class]
        return matches[0] if len(matches) else train_correct[0] if len(train_correct) else 0

    def find_test_correct_pair():
        for idx in test_correct:
            target_class = y_test_true[idx]
            train_idx = find_train_example_for_class(target_class)
            return train_idx, idx
        return 0, 0

    def find_test_incorrect_pair():
        for idx in test_incorrect:
            target_class = y_test_true[idx]
            train_idx = find_train_example_for_class(target_class)
            return train_idx, idx
        return 0, 0

    train_idx_correct, test_idx_correct = find_test_correct_pair()
    train_idx_incorrect, test_idx_incorrect = find_test_incorrect_pair()

    save_xrd_seen_unseen_plot(
        theta=theta if theta is not None else np.arange(X.shape[1]),
        train_pattern=X_train[train_idx_correct],
        train_true=class_names[y_train_true[train_idx_correct]],
        train_pred=class_names[y_train_pred[train_idx_correct]],
        test_pattern=X_test[test_idx_correct],
        test_true=class_names[y_test_true[test_idx_correct]],
        test_pred=class_names[y_test_pred[test_idx_correct]],
        path=output_dir / "seen_vs_unseen_correct_xrd_prediction.png",
        title="Task 3: Same-class seen vs correct unseen XRD prediction",
    )

    save_xrd_seen_unseen_plot(
        theta=theta if theta is not None else np.arange(X.shape[1]),
        train_pattern=X_train[train_idx_incorrect],
        train_true=class_names[y_train_true[train_idx_incorrect]],
        train_pred=class_names[y_train_pred[train_idx_incorrect]],
        test_pattern=X_test[test_idx_incorrect],
        test_true=class_names[y_test_true[test_idx_incorrect]],
        test_pred=class_names[y_test_pred[test_idx_incorrect]],
        path=output_dir / "seen_vs_unseen_incorrect_xrd_prediction.png",
        title="Task 3: Same-class seen vs incorrect unseen XRD prediction",
    )

    print("\nMetrics")
    print("------------------------------------------------------------")

    for split, values in metrics.items():
        print(
            f"{split:10s} | "
            f"loss={values['loss']:.4f} | "
            f"accuracy={values['accuracy']:.3f}"
        )

    print("\nBaseline comparison:")
    print(f"Random guessing across {len(class_names)} classes gives ~{1 / len(class_names):.2f} accuracy.")
    print("The CNN should significantly exceed this if it is learning useful diffraction structure.")

    print(f"\nTotal training time: {total_time:.2f}s")

    print("\nInterpretation:")
    print("- High train accuracy but lower validation accuracy suggests overfitting.")
    print("- Similar train and validation accuracy suggests more stable generalisation.")
    print("- Test accuracy should be treated as final evaluation, not a tuning signal.")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(output_dir / "metrics.csv")

    pd.DataFrame(
        {
            "true_label": class_names[y_test_true],
            "predicted_label": class_names[y_test_pred],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    save_loss_curve(
        history,
        output_dir / "loss_curve.png",
        title="Task 3: Train vs validation loss",
        ylabel="Cross-entropy loss",
    )

    save_split_metric_plot(
        metrics,
        output_dir / "split_accuracy.png",
        metric_name="accuracy",
        title="Task 3: Train vs validation vs test accuracy",
        ylabel="Accuracy",
    )

    print(f"\nOutputs saved to: {output_dir}")

    print("\nNext steps:")
    print("- Turn augmentation on and compare validation accuracy and runtime.")
    print("- Increase channels and inspect whether train accuracy improves faster than validation accuracy.")
    print("- Change kernel_size and consider what size of peak region the CNN should see.")
    print("- Disable learning-rate scheduling and compare training stability.")


if __name__ == "__main__":
    main()