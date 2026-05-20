from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_regression_loader(X, y, batch_size: int, shuffle: bool):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_regression(model, loader, loss_fn):
    model.eval()

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            losses.append(loss.item() * len(X_batch))
            predictions.append(y_pred.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())

    mean_loss = sum(losses) / len(loader.dataset)

    return (
        mean_loss,
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state = None
        self.counter = 0

    def step(self, val_loss: float, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)