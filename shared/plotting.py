from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def save_parity_plot(
    y_true,
    y_pred,
    path: str | Path,
    title: str,
    xlabel: str = "True value",
    ylabel: str = "Predicted value",
    label: str = "Predictions",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.75, label=label)
    ax.plot([lo, hi], [lo, hi], linestyle="--", label="Ideal prediction")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


#  1. Task 1 Plotting

def save_split_metric_plot(
    metrics: dict,
    path: str | Path,
    metric_name: str = "rmse",
    title: str = "Model performance by split",
    ylabel: str = "RMSE",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(metrics.keys())
    values = [metrics[split][metric_name] for split in splits]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, values)

    ax.set_title(title)
    ax.set_xlabel("Dataset split")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_feature_importance_plot(
    feature_importances,
    path: str | Path,
    title: str = "Feature importance",
    xlabel: str = "Importance",
    ylabel: str = "Feature",
    top_n: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = feature_importances.copy().sort_values("importance", ascending=True)

    if top_n is not None:
        df = df.tail(top_n)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(df))))
    ax.barh(df["feature"], df["importance"])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

#  2. Task 2 & Task 3 Plotting

def save_loss_curve(
    history: list[dict],
    path: str | Path,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    train_key: str = "train_loss",
    val_key: str = "val_loss",
    train_label: str = "Training loss",
    val_label: str = "Validation loss",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs, [row[train_key] for row in history], label=train_label)
    ax.plot(epochs, [row[val_key] for row in history], label=val_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

#  3. Task 3 Plotting

def save_xrd_seen_unseen_plot(
    theta,
    train_pattern,
    train_true,
    train_pred,
    test_pattern,
    test_true,
    test_pred,
    path: str | Path,
    title: str = "Seen vs unseen XRD predictions",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(theta, train_pattern)
    axes[0].set_title(f"Seen during training | true: {train_true} | predicted: {train_pred}")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(alpha=0.25)

    axes[1].plot(theta, test_pattern)
    axes[1].set_title(f"Unseen test sample | true: {test_true} | predicted: {test_pred}")
    axes[1].set_xlabel("2-theta")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def save_confusion_matrix_plot(
    y_true,
    y_pred,
    class_names,
    path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    # Configure axes
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Add text labels inside the matrix cells
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm_display[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
            )

    ax.set_title(title)
    ax.set_ylabel("True crystal system")
    ax.set_xlabel("Predicted crystal system")
    
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
