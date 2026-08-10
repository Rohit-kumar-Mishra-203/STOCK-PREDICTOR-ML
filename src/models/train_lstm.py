"""
Train an LSTM to predict next-day price direction, as a deep-learning
comparison against the XGBoost baseline (see train_baseline.py).

Why compare against XGBoost at all: XGBoost treats each day as an
independent row. An LSTM processes a SEQUENCE of days and can, in
principle, learn temporal patterns (building momentum, multi-day trends)
that a row-at-a-time model can't directly see. Whether that theoretical
advantage shows up in practice - on this data, at this scale - is
exactly what this comparison is for. It's entirely possible (and a
legitimate, reportable finding either way) that XGBoost wins anyway.

=== TWO LSTM-SPECIFIC LOOKAHEAD/LEAKAGE RISKS NOT PRESENT IN XGBOOST ===

1. SEQUENCE WINDOWS: a sequence for predicting day t's target must only
   contain features from days <= t. Building a window means slicing
   [t - seq_len + 1 : t + 1] - never reaching past t. This mirrors the
   row-level rule from build_dataset.py, just applied to a whole window.

2. FEATURE SCALING: LSTMs need normalized inputs to train properly
   (unlike XGBoost, which is scale-invariant). The scaler's mean/std
   MUST be computed from the TRAINING set only, then applied to both
   train and test. Fitting the scaler on the combined dataset would let
   the model's inputs be quietly informed by the test period's price
   statistics - a subtle, easy-to-miss form of lookahead bias.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger
from src.models.train_baseline import load_ticker_dataset, chronological_split, get_feature_columns, compute_naive_baseline

SEQ_LEN = 20  # how many past trading days the LSTM looks at per prediction


class SequenceDataset(Dataset):
    """
    Wraps scaled features + targets into fixed-length sequences.

    LOOKAHEAD CHECK: __getitem__ returns X[idx : idx + seq_len] as the
    input window and y[idx + seq_len - 1] as the target - i.e. the
    window's LAST day is the day the target corresponds to. The window
    never includes any row beyond that day. This is the sequence
    equivalent of "features only use data at or before t".
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        window = self.X[idx: idx + self.seq_len]
        target = self.y[idx + self.seq_len - 1]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class LSTMClassifier(nn.Module):
    """
    A small, deliberately simple LSTM - given our dataset has under 1000
    training rows, a large/deep network would overfit almost immediately.
    Starting small and simple is a defensible choice, not a limitation
    to apologize for.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the final hidden state - it's a summary of the whole sequence
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)  # raw logits, sigmoid applied in loss


def prepare_sequences(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    """
    Scale features (fit on train only) and build sequence datasets.

    LOOKAHEAD CHECK: scaler.fit() is called ONLY on X_train. X_test is
    transformed using train's fitted mean/std via scaler.transform() -
    never scaler.fit_transform() on test. This is the scaling
    equivalent of the chronological train/test split rule.
    """
    X_train_raw = np.asarray(train_df[feature_cols].values, dtype=np.float64)
    y_train_raw = np.asarray(train_df["target"].values, dtype=np.float64)
    X_test_raw = np.asarray(test_df[feature_cols].values, dtype=np.float64)
    y_test_raw = np.asarray(test_df["target"].values, dtype=np.float64)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)   # fit + transform on train
    X_test_scaled = scaler.transform(X_test_raw)          # transform ONLY on test, using train's stats

    train_ds = SequenceDataset(X_train_scaled, y_train_raw, SEQ_LEN)
    test_ds = SequenceDataset(X_test_scaled, y_test_raw, SEQ_LEN)

    return train_ds, test_ds, scaler


def train_lstm_model(train_ds: SequenceDataset, input_size: int, epochs: int = 30, lr: float = 1e-3) -> LSTMClassifier:
    """
    Train the LSTM with a simple held-out validation split (last 15% of
    training sequences, chronologically - NOT randomly, same principle
    as everywhere else) for early stopping.

    Why early stopping: with limited data and a model that can overfit
    within a handful of epochs, training for a fixed large number of
    epochs regardless of validation performance risks just memorizing
    the training set. Stopping when validation loss stops improving is
    a standard, defensible regularization technique.
    """
    n_val = max(1, int(len(train_ds) * 0.15))
    n_train = len(train_ds) - n_val

    # Chronological split of the training sequences themselves - the
    # validation slice is the LAST n_val sequences, not a random sample.
    train_subset = torch.utils.data.Subset(train_ds, range(0, n_train))
    val_subset = torch.utils.data.Subset(train_ds, range(n_train, len(train_ds)))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = LSTMClassifier(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= n_val

        logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best checkpoint, not the last (possibly overfit) one

    return model


def evaluate_lstm(model: LSTMClassifier, test_ds: SequenceDataset) -> dict:
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.int().tolist())

    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
    }
    cm = confusion_matrix(all_targets, all_preds)
    logger.info(f"LSTM confusion matrix:\n{cm}")

    return metrics


def run_for_ticker(ticker: str, split_date: str) -> dict:
    logger.info(f"=== Training LSTM for {ticker} ===")

    df = load_ticker_dataset(ticker)
    train_df, test_df = chronological_split(df, split_date)
    feature_cols = get_feature_columns(df)

    train_ds, test_ds, scaler = prepare_sequences(train_df, test_df, feature_cols)
    logger.info(f"Train sequences: {len(train_ds)}, Test sequences: {len(test_ds)}, seq_len={SEQ_LEN}")

    model = train_lstm_model(train_ds, input_size=len(feature_cols))
    metrics = evaluate_lstm(model, test_ds)

    # Naive baseline computed over the same rows the LSTM actually predicted
    # on (test_df loses its first SEQ_LEN-1 rows to sequence construction)
    naive_test_df = test_df.iloc[SEQ_LEN - 1:]
    naive = compute_naive_baseline(naive_test_df)

    edge = metrics["accuracy"] - naive["accuracy"]
    logger.info(f"{ticker} - LSTM accuracy: {metrics['accuracy']:.4f} | Naive: {naive['accuracy']:.4f} | Edge: {edge:+.4f}")

    return {"ticker": ticker, "metrics": metrics, "naive_baseline": naive}


def main():
    cfg = get_config()
    tickers: list[str] = cfg["data"]["tickers"]
    split_date: str = str(cfg["model"]["train_test_split_date"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = []
    for ticker in tickers:
        with mlflow.start_run(run_name=f"lstm_{ticker}"):
            result = run_for_ticker(ticker, split_date)
            results.append(result)

            mlflow.log_param("ticker", ticker)
            mlflow.log_param("seq_len", SEQ_LEN)
            mlflow.log_metrics(result["metrics"])
            mlflow.log_metric("naive_baseline_accuracy", result["naive_baseline"]["accuracy"])
            mlflow.log_metric("edge_over_naive", result["metrics"]["accuracy"] - result["naive_baseline"]["accuracy"])

    logger.info("=== LSTM Summary ===")
    for r in results:
        edge = r["metrics"]["accuracy"] - r["naive_baseline"]["accuracy"]
        logger.info(f"{r['ticker']}: accuracy={r['metrics']['accuracy']:.4f}, edge_over_naive={edge:+.4f}")


if __name__ == "__main__":
    main()