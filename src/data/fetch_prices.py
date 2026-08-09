"""
Fetch historical OHLCV price data for configured tickers using yfinance.

Why yfinance:
Free, no API key required, reliable enough for a portfolio project. In a
real production system you'd likely pay for a more robust data vendor
(Polygon, Alpha Vantage premium, IEX Cloud) since yfinance scrapes Yahoo
Finance and can break if Yahoo changes their site — worth mentioning this
limitation explicitly if asked, it shows awareness of production tradeoffs.

Why we save raw data to disk before processing:
Separating "fetch" from "process" means if a later step in the pipeline
fails or a bug is found in feature engineering, we don't need to re-hit
the API (which is rate-limited and slow) — we just re-run processing on
the already-saved raw data. This raw/processed separation is a standard
data engineering pattern.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger
from typing import Optional



from typing import Optional


def fetch_ticker_data(ticker: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date. Raises if no data is returned, since a silent
    empty result is worse than a loud failure — we want the pipeline
    to stop and tell us, not proceed with missing data.
    """
    logger.info(f"Fetching price data for {ticker} from {start_date} to {end_date or 'today'}")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=True,  # adjusts for splits/dividends — important for accurate returns
    )

    # yfinance's type stubs say this can return None — narrow the type here
    # so the rest of the function (and Pylance) knows it's a real DataFrame.
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check ticker symbol or date range.")

    # yfinance sometimes returns MultiIndex columns for a single ticker download
    # depending on version — flatten to be safe and consistent downstream.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["ticker"] = ticker
    logger.info(f"Fetched {len(df)} rows for {ticker}")
    return df


def fetch_all_tickers(tickers: list[str], start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Fetch data for all configured tickers.

    Returns a dict keyed by ticker rather than one giant combined DataFrame,
    because saving per-ticker files makes it trivial to add/remove a single
    ticker later without touching or re-fetching the others.
    """
    results: dict[str, pd.DataFrame] = {}
    failed = []

    for ticker in tickers:
        try:
            results[ticker] = fetch_ticker_data(ticker, start_date, end_date, interval)
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed to fetch data for: {failed}")

    return results

def save_raw_prices(data: dict[str, pd.DataFrame]) -> None:
    """Save each ticker's raw data as a separate CSV under data/raw/prices/."""
    cfg = get_config()
    save_dir = resolve_path(cfg["paths"]["raw_prices"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in data.items():
        filepath = save_dir / f"{ticker}.csv"
        df.to_csv(filepath)
        logger.info(f"Saved {ticker} data to {filepath}")


def main():
    cfg = get_config()
    tickers = cfg["data"]["tickers"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")
    interval = cfg["data"]["price_interval"]

    logger.info(f"Starting price fetch for {len(tickers)} tickers: {tickers}")

    data = fetch_all_tickers(tickers, start_date, end_date, interval)
    save_raw_prices(data)

    logger.info("Price fetch complete.")


if __name__ == "__main__":
    main()