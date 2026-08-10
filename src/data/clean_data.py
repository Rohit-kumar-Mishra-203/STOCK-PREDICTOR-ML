"""
Clean and merge raw price and news data into a single, model-ready dataset.

Why this step is separate from fetching:
Fetching hits external APIs (slow, rate-limited, can fail). Cleaning is
pure computation on already-saved data (fast, repeatable, no network
dependency). Separating them means we can re-run cleaning logic as many
times as we want while debugging, without re-hitting APIs.

Key data quality problem this file solves:
Prices only exist on trading days (no weekends/holidays). News can be
published any day, any time. Without deliberate handling, weekend news
either gets silently dropped or misaligned to the wrong trading day.
Our rule: news published on a non-trading day is attributed to the NEXT
trading day, since that's the first day the market could realistically
react to it.
"""

import numpy as np
import pandas as pd

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger


def load_raw_prices(ticker: str) -> pd.DataFrame:
    """Load a single ticker's raw price CSV, saved by fetch_prices.py."""
    cfg = get_config()
    filepath = resolve_path(cfg["paths"]["raw_prices"]) / f"{ticker}.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"No raw price file found for {ticker} at {filepath}. Run fetch_prices.py first.")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  # strip timezone for clean date matching
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_raw_news() -> pd.DataFrame:
    """
    Load the most recent raw news CSV.

    Why 'most recent': fetch_news.py saves a new dated file each run
    (news_YYYYMMDD.csv). We always want the latest pull, not an old one.
    """
    cfg = get_config()
    news_dir = resolve_path(cfg["paths"]["raw_news"])
    news_files = sorted(news_dir.glob("news_*.csv"))

    if not news_files:
        raise FileNotFoundError(f"No raw news files found in {news_dir}. Run fetch_news.py first.")

    latest_file = news_files[-1]
    logger.info(f"Loading news from {latest_file.name}")

    df = pd.read_csv(latest_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def align_news_to_trading_days(news_df: pd.DataFrame, trading_days: pd.Series) -> pd.DataFrame:
    """
    Shift any news date that falls on a non-trading day forward to the
    next available trading day.

    Why this approach specifically (vs. dropping weekend news, or
    backdating it): dropping loses real information (a bad weekend
    headline can absolutely move Monday's price). Backdating to the
    previous trading day is wrong - the market couldn't have reacted
    to news it didn't know about yet. Forward-filling to the next
    trading day is the only choice consistent with cause-and-effect.

    Implemented as a single vectorized np.searchsorted call across the
    whole news_df at once, rather than .apply() row-by-row - faster for
    large datasets (one call vs. thousands), and avoids ambiguous return
    types that trip up static type checkers with .apply().
    """
    # Explicit cast to a concrete datetime64 ndarray - .unique() alone
    # returns a broad Unknown|ExtensionArray|Categorical type that both
    # np.sort and np.searchsorted refuse to accept under strict type
    # checking, even though it works fine at runtime.
    trading_days_sorted = np.sort(
        np.asarray(trading_days.unique(), dtype="datetime64[ns]")
    )

    news_df = news_df.copy()
    news_dates = np.asarray(news_df["date"].values, dtype="datetime64[ns]")

    # searchsorted returns, for each news date, the index of the first
    # trading day >= that date - exactly the "next available" trading day.
    indices = np.searchsorted(trading_days_sorted, news_dates)

    # Any index beyond the array length means the news is too recent -
    # no future trading day exists yet in our data.
    valid_mask = indices < len(trading_days_sorted)

    trading_dates = np.full(len(news_df), np.datetime64("NaT"), dtype="datetime64[ns]")
    trading_dates[valid_mask] = trading_days_sorted[indices[valid_mask]]

    news_df["trading_date"] = trading_dates

    dropped = int((~valid_mask).sum())
    if dropped > 0:
        logger.warning(f"Dropped {dropped} news rows with no future trading day available yet")

    return news_df.dropna(subset=["trading_date"])


def clean_ticker_dataset(ticker: str, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one clean, merged dataset for a single ticker: price data
    with aligned news rows attached (still one row per article at this
    stage - aggregation into daily sentiment happens later in feature
    engineering, kept separate since that's a modeling decision, not
    a cleaning one).
    """
    prices = load_raw_prices(ticker)

    price_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Basic price data quality checks - fail loudly rather than silently
    # proceeding with bad data.
    if prices[price_cols].isna().any().any():
        n_missing = int(prices[price_cols].isna().sum().sum())
        logger.warning(f"{ticker}: found {n_missing} missing price values, forward-filling")
        prices[price_cols] = prices[price_cols].ffill()

    ticker_news = news_df[news_df["ticker"] == ticker].copy()

    if ticker_news.empty:
        logger.warning(f"No news found for {ticker} - proceeding with price-only data")
        prices["trading_date"] = prices["Date"]
        return prices

    aligned_news = align_news_to_trading_days(ticker_news, prices["Date"])

    # Merge news onto price data by trading_date. This is a left join
    # from price's perspective - every trading day is kept even if no
    # news exists for it (that's a valid, common case, not missing data).
    merged = prices.merge(
        aligned_news[["trading_date", "title", "description", "source"]],
        left_on="Date",
        right_on="trading_date",
        how="left",
    )

    return merged


def clean_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    news_df = load_raw_news()
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        logger.info(f"Cleaning dataset for {ticker}")
        results[ticker] = clean_ticker_dataset(ticker, news_df)

    return results


def save_processed(data: dict[str, pd.DataFrame]) -> None:
    cfg = get_config()
    save_dir = resolve_path(cfg["paths"]["processed"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in data.items():
        filepath = save_dir / f"{ticker}_merged.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved cleaned dataset for {ticker} to {filepath} ({len(df)} rows)")


def main():
    cfg = get_config()
    tickers: list[str] = cfg["data"]["tickers"]

    logger.info("Starting data cleaning and merging")
    data = clean_all(tickers)
    save_processed(data)
    logger.info("Cleaning complete.")


if __name__ == "__main__":
    main()