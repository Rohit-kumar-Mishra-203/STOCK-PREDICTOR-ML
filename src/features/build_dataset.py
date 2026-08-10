"""
Combine technical indicators, sentiment features, and lag features into
one final model-ready dataset per ticker, and construct the target
variable (next-day price direction).

=== LOOKAHEAD BIAS: THE CENTRAL CONCERN OF THIS FILE ===

Lookahead bias happens when a model is trained on information that
would not actually have been available at prediction time. It's the
single most common way stock prediction projects silently cheat -
the model looks great in backtesting and is useless in production,
because during training it was secretly allowed to see the future.

The rule this file enforces everywhere:
    For a row representing trading day t:
    - FEATURES may only use data from t or earlier
    - TARGET is explicitly computed from t+1 (the one deliberate,
      clearly-labeled exception, and only for the label - never a feature)

Every function below is annotated with which side of that line it's on.
"""

import pandas as pd

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger
from src.features.technical_indicators import add_technical_indicators
from src.features.sentiment import add_sentiment_features


def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    """
    Add lagged versions of Close price and sentiment as features.

    LOOKAHEAD CHECK: df['Close'].shift(n) with POSITIVE n pulls a value
    from n rows EARLIER into the current row - i.e. "what was the price
    n days ago". This looks backward in time, which is exactly what a
    feature is allowed to do. This is the opposite direction from the
    target's shift(-1) used later in this file - that distinction (shift
    positive = safe feature, shift negative = must only be used for target)
    is the single most important thing to get right in this whole file.
    """
    df = df.copy()

    for lag in lag_days:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

        if "sentiment_positive" in df.columns:
            df[f"sentiment_pos_lag_{lag}"] = df["sentiment_positive"].shift(lag)

    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily and rolling returns as features.

    LOOKAHEAD CHECK: pct_change() by default compares row t to row t-1
    (backward-looking) - this is the current day's return relative to
    the PREVIOUS day, computable using only data known by the close of
    day t. Safe. We are NOT computing t to t+1 here (that would be the
    target, not a feature).
    """
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=5).std()
    return df


def fill_no_news_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill sentiment columns with neutral defaults on days with no news,
    rather than leaving NaN.

    WHY THIS IS CORRECT, NOT DATA FABRICATION: a NaN here doesn't mean
    "unknown value we're missing" - it means "no news was published,"
    which is a real, common, meaningful state. Representing "no news"
    as sentiment_neutral=1.0 (fully neutral) and positive/negative=0.0
    is an honest encoding of that fact, not a guess. Contrast this with
    something like filling a missing Close price with 0 - that WOULD be
    fabrication, because there's no such thing as "no price today."

    We explicitly cast to float64 via pd.to_numeric before filling -
    the sentiment columns can end up as 'object' dtype upstream (from
    pd.NA assignment when there's zero news at all), and pandas warns
    about silently downcasting object -> float on fillna. Casting first
    makes the dtype explicit and removes the ambiguity/warning entirely.
    """
    df = df.copy()

    sentiment_defaults = {
        "sentiment_positive": 0.0,
        "sentiment_negative": 0.0,
        "sentiment_neutral": 1.0,
    }

    for col, default in sentiment_defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["news_article_count"] = pd.to_numeric(df["news_article_count"], errors="coerce").fillna(0)

    # Lag features on sentiment inherit the same pattern - fill those too
    for col in df.columns:
        if col.startswith("sentiment_pos_lag_"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def build_target(df: pd.DataFrame, horizon_days: int, target_type: str) -> pd.DataFrame:
    """
    Construct the target variable - the ONLY place in this entire file
    where we deliberately look into the future, and only to build the
    label the model will learn to predict, never a feature it learns from.

    LOOKAHEAD CHECK (read this carefully - this is the critical line):
        df['Close'].shift(-horizon_days)
    The NEGATIVE sign is what pulls a FUTURE value backward into the
    current row. Row t's 'future_close' column now holds day t+horizon's
    actual closing price. This is intentional and necessary - it's the
    answer key. The model will never see this column as an input feature;
    it exists ONLY to become the target/label.

    Rows at the end of the dataset (Nth-to-last horizon_days rows) will
    have NaN future_close, since there's no real future data for them yet
    - these get dropped, since we cannot honestly evaluate a prediction
    we don't have a real outcome for.
    """
    df = df.copy()
    df["future_close"] = df["Close"].shift(-horizon_days)

    if target_type == "direction":
        # 1 if price goes up, 0 if it goes down or stays flat
        df["target"] = (df["future_close"] > df["Close"]).astype(int)
    elif target_type == "return":
        df["target"] = (df["future_close"] - df["Close"]) / df["Close"]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Drop rows with no real future outcome to compare against -
    # keeping them with a fabricated/default target would silently
    # corrupt evaluation with fake labels.
    n_before = len(df)
    df = df.dropna(subset=["future_close"])
    n_dropped = n_before - len(df)
    logger.info(f"Dropped {n_dropped} rows with no future outcome yet (end of dataset)")

    # future_close itself is now discarded - it was only scaffolding to
    # build 'target'. Leaving it in the dataframe would be a live
    # lookahead landmine if anyone ever mistakenly used it as a feature.
    df = df.drop(columns=["future_close"])

    return df

def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute price-level features into relative/normalized ones.

    WHY THIS MATTERS FOR TREE-BASED MODELS: XGBoost splits on absolute
    thresholds (e.g. "Close > 150"). If training data covers one price
    range (e.g. AAPL $60-$200) and test data covers a different range
    (e.g. $250-$340), the model is effectively extrapolating beyond
    anything it learned from - trees handle this very poorly, unlike
    linear models which can at least project a trend. Ratios and
    percentage differences (e.g. Close/SMA_50) stay in a stable,
    comparable range regardless of the stock's absolute price level
    or how much it's grown over time.

    LOOKAHEAD CHECK: every ratio here divides same-day or past values
    by same-day or past values (Close_t / SMA_50_t, Close_t / Close_{t-lag}).
    No future information enters any of these - same safety guarantee
    as the raw features they're derived from.
    """
    df = df.copy()

    df["close_to_sma10"] = df["Close"] / df["sma_10"]
    df["close_to_sma50"] = df["Close"] / df["sma_50"]
    df["sma10_to_sma50"] = df["sma_10"] / df["sma_50"]

    # Position within the Bollinger Bands, 0 = at lower band, 1 = at upper band
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_pct"] = (df["Close"] - df["Open"]) / df["Open"]

    # MACD is already a difference of EMAs, but its absolute magnitude still
    # scales with the stock's price level - normalize by dividing by Close
    df["macd_line_norm"] = df["macd_line"] / df["Close"]
    df["macd_signal_norm"] = df["macd_signal"] / df["Close"]
    df["macd_histogram_norm"] = df["macd_histogram"] / df["Close"]

    # Volume also trends over years (more shares outstanding, more interest) -
    # compare to its own recent rolling average instead of raw share count
    df["volume_to_avg20"] = df["Volume"] / df["Volume"].rolling(window=20, min_periods=20).mean()

    # Replace raw lagged prices with a ratio (equivalent to an N-day return),
    # which is naturally scale-invariant
    lag_cols = [c for c in df.columns if c.startswith("close_lag_")]
    for col in lag_cols:
        lag = col.split("_")[-1]
        df[f"close_ratio_lag_{lag}"] = df["Close"] / df[col]

    return df


def build_ticker_features(ticker: str) -> pd.DataFrame:
    """
    Full feature pipeline for one ticker: load merged data, add technical
    indicators, add sentiment, add lags/returns, then build the target.

    ORDER MATTERS: target is built LAST, after all features. This isn't
    just style - it makes the lookahead boundary visually obvious in the
    code (everything above this call is "past-safe", the one call below
    is where we deliberately reach into the future for the label only).
    """
    cfg = get_config()
    processed_path = resolve_path(cfg["paths"]["processed"]) / f"{ticker}_merged.csv"

    if not processed_path.exists():
        raise FileNotFoundError(f"No merged dataset found for {ticker} at {processed_path}. Run clean_data.py first.")

    df = pd.read_csv(processed_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # 'trading_date' is a leftover artifact from the price/news merge in
    # clean_data.py - it duplicates 'Date' when news matched, and is NaT
    # otherwise. It serves no purpose downstream and was silently causing
    # dropna() to wipe out nearly the whole dataset. Drop it here.
    if "trading_date" in df.columns:
        df = df.drop(columns=["trading_date"])

    logger.info(f"Building features for {ticker}")

    df = add_technical_indicators(df, indicators=cfg["features"]["technical_indicators"])
    df = add_sentiment_features(df)
    df = add_return_features(df)
    df = add_lag_features(df, lag_days=cfg["features"]["lag_days"])
    df = add_normalized_features(df)
    df = fill_no_news_defaults(df)

    # ---- everything above this line is past-safe feature engineering ----
    # ---- everything below deliberately touches the future, for the target only ----

    df = build_target(
        df,
        horizon_days=cfg["target"]["horizon_days"],
        target_type=cfg["target"]["type"],
    )

    # Rolling/lag features produce NaN for the first several rows (not
    # enough history yet, e.g. a 50-day SMA needs 50 real days first).
    # Drop them rather than filling with fabricated values - an honest
    # gap in early data is better than a misleading filled-in number.
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    logger.info(f"Dropped {n_dropped} rows with incomplete feature history (start of dataset)")

    return df


def build_all_datasets(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        results[ticker] = build_ticker_features(ticker)
    return results


def save_final_datasets(data: dict[str, pd.DataFrame]) -> None:
    cfg = get_config()
    save_dir = resolve_path(cfg["paths"]["processed"])

    for ticker, df in data.items():
        filepath = save_dir / f"{ticker}_final.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved final dataset for {ticker} to {filepath} ({len(df)} rows, {len(df.columns)} columns)")


def main():
    cfg = get_config()
    tickers: list[str] = cfg["data"]["tickers"]

    logger.info("Building final feature datasets")
    data = build_all_datasets(tickers)
    save_final_datasets(data)
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()