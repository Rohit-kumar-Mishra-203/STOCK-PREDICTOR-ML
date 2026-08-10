"""
Hyperparameter tuning and feature importance analysis for the XGBoost
baseline.

=== WHY TimeSeriesSplit, NOT STANDARD KFold ===
Standard KFold cross-validation randomly shuffles rows into folds - this
would let hyperparameter tuning "cheat" the exact same way a random
train/test split would (see train_baseline.py), just one level deeper:
a fold could contain 2022 data used to validate performance on 2021 data.
TimeSeriesSplit instead creates folds where each validation fold comes
strictly AFTER all the training folds before it - preserving the same
"only the past predicts the future" principle throughout the entire
tuning process, not just the final train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we're saving to file
import matplotlib.pyplot as plt

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger
from src.models.train_baseline import (
    load_ticker_dataset,
    chronological_split,
    get_feature_columns,
    evaluate_model,
    compute_naive_baseline,
)


def analyze_feature_importance(model: xgb.XGBClassifier, feature_cols: list[str], ticker: str, top_n: int = 15) -> pd.DataFrame:
    """
    Extract and save feature importances from a trained model.

    Why this matters beyond curiosity: if sentiment features rank near
    the bottom, that's a real, honest finding worth reporting - it would
    mean the NLP half of this project adds less signal than the technical
    indicators, which is a legitimate (if less flattering) result to be
    upfront about rather than assuming NLP must be helping.
    """
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info(f"Top {top_n} features for {ticker}:\n{imp_df.head(top_n).to_string(index=False)}")

    # Save a plot - useful for README/report, not just console output
    cfg = get_config()
    plot_dir = resolve_path(cfg["paths"]["models"]) / "feature_importance"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top = imp_df.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"{ticker} - Top {top_n} Feature Importances")
    plt.tight_layout()
    plot_path = plot_dir / f"{ticker}_feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved feature importance plot to {plot_path}")

    return imp_df


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 30) -> xgb.XGBClassifier:
    """
    Randomized hyperparameter search using TimeSeriesSplit CV.

    Why RandomizedSearchCV over exhaustive GridSearchCV: with 6 parameters
    each having several candidate values, a full grid would mean hundreds
    of combinations - expensive for marginal gain. Randomized search
    samples a fixed budget (n_iter) of combinations, which research shows
    finds near-optimal results with a fraction of the compute in practice.

    Why n_splits=5 for TimeSeriesSplit: gives 5 sequential train/validate
    folds within the training data only (never touching the held-out test
    set) - balances having enough folds to trust the average, without
    each fold's training set becoming too small to be meaningful.
    """
    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    base_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,  # use all CPU cores - tuning is embarrassingly parallel
        verbose=1,
    )

    logger.info(f"Starting hyperparameter search ({n_iter} candidates x 5 folds = {n_iter * 5} fits)")
    search.fit(X_train, y_train)

    logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_estimator_


def run_tuning_for_ticker(ticker: str, split_date: str) -> dict:
    logger.info(f"=== Tuning baseline for {ticker} ===")

    df = load_ticker_dataset(ticker)
    train_df, test_df = chronological_split(df, split_date)

    feature_cols = get_feature_columns(df)
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    tuned_model = tune_hyperparameters(X_train, y_train)

    metrics = evaluate_model(tuned_model, X_test, y_test)
    naive = compute_naive_baseline(test_df)

    logger.info(f"{ticker} - Tuned accuracy: {metrics['accuracy']:.4f} | Naive: {naive['accuracy']:.4f} | Edge: {metrics['accuracy'] - naive['accuracy']:+.4f}")

    importance_df = analyze_feature_importance(tuned_model, feature_cols, ticker)

    return {
        "ticker": ticker,
        "model": tuned_model,
        "metrics": metrics,
        "naive_baseline": naive,
        "importance": importance_df,
    }


def main():
    cfg = get_config()
    tickers: list[str] = cfg["data"]["tickers"]
    split_date: str = str(cfg["model"]["train_test_split_date"])

    results = []
    for ticker in tickers:
        results.append(run_tuning_for_ticker(ticker, split_date))

    logger.info("=== Tuned vs Untuned Summary ===")
    for r in results:
        edge = r["metrics"]["accuracy"] - r["naive_baseline"]["accuracy"]
        logger.info(f"{r['ticker']}: tuned_accuracy={r['metrics']['accuracy']:.4f}, edge_over_naive={edge:+.4f}")


if __name__ == "__main__":
    main()