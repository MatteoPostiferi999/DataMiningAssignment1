import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_classification(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob)
    return metrics


def evaluate_regression(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Train / test split — per-user temporal
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().sort_values(["id", "date"]).reset_index(drop=True)
    train_idx, test_idx = [], []
    for uid, grp in df.groupby("id"):
        n = len(grp)
        cutoff = int(n * train_frac)
        train_idx.extend(grp.index[:cutoff].tolist())
        test_idx.extend(grp.index[cutoff:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def fix_user_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    stats = train.groupby("id")["mood"].agg(
        user_mood_mean="mean", user_mood_std="std"
    )
    for col in ["user_mood_mean", "user_mood_std"]:
        train[col] = train["id"].map(stats[col])
        test[col] = test["id"].map(stats[col])
    return train, test


# ---------------------------------------------------------------------------
# 1D CNN for classification
# ---------------------------------------------------------------------------

class MoodCNN(nn.Module):
    def __init__(self, n_channels: int, seq_len: int = 7,
                 n_filters_1: int = 32, n_filters_2: int = 64,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, n_filters_1, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters_2, 1)

    def forward(self, x):
        # x: (batch, seq_len, channels) → transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.drop(torch.relu(self.conv1(x)))
        x = self.drop(torch.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class MoodCNNRegressor(nn.Module):
    def __init__(self, n_channels: int, seq_len: int = 7,
                 n_filters_1: int = 32, n_filters_2: int = 64,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, n_filters_1, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters_2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.drop(torch.relu(self.conv1(x)))
        x = self.drop(torch.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def train_cnn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 150,
    patience: int = 15,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    model = model.to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_vl = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_vl = torch.tensor(y_val, dtype=torch.float32, device=device)

    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    n = len(X_tr)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            out = model(X_tr[idx])
            loss = criterion(out, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_out = model(X_vl)
            val_loss = criterion(val_out, y_vl).item()

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = len(history["val_loss"]) - patience
    history["epochs_run"] = len(history["val_loss"])
    return history


def predict_cnn(model: nn.Module, X: np.ndarray, task: str = "classification"):
    model.eval()
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    if task == "classification":
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs >= 0.5).astype(int)
        return preds, probs
    else:
        return logits
