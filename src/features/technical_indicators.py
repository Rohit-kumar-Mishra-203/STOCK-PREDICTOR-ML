"""
Compute technical indicators from OHLCV price data.

Why implement these manually instead of using a library (e.g., pandas-ta):
pandas-ta's last working release dropped Python 3.10 support, and the
underlying formulas for these indicators are simple enough to implement
directly with pandas/numpy. This also means zero dependency on an
unmaintained package, and demonstrates actual understanding of what each
indicator measures rather than treating it as a black-box function call.
"""

import pandas as pd
import numpy as np

from src.utils.logger import logger


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index - measures momentum: how strongly a stock
    has been rising vs falling over the recent window. Ranges 0-100.
    Above 70 = commonly considered 'overbought', below 30 = 'oversold'.

    Formula: RSI = 100 - (100 / (1 + RS)), where RS = avg gain / avg loss
    over the period.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # avoid divide-by-zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence - measures trend direction
    and momentum by comparing a fast and slow exponential moving average.

    Returns a DataFrame with macd_line, signal_line, and macd_histogram
    (the difference between them - a common signal for trend reversals).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average - smooths price over a window to reveal trend."""
    return close.rolling(window=window, min_periods=window).mean()


def compute_bbands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands - a moving average with upper/lower bands at N standard
    deviations, capturing volatility. Price near the upper band suggests
    relatively high/expensive; near the lower band suggests relatively low.
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": sma,
        "bb_lower": lower,
        # bandwidth is a useful derived feature - how "squeezed" or
        # "wide" the bands currently are, often a precursor to breakouts
        "bb_bandwidth": (upper - lower) / sma,
    })


def add_technical_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Add the configured set of technical indicators as new columns.

    Why config-driven (not hardcoded which indicators to add): lets us
    toggle indicators on/off from config.yaml for experimentation without
    touching code - e.g. testing whether MACD actually improves accuracy
    or is just noise, without commenting code in and out.
    """
    df = df.copy()

    if "rsi" in indicators:
        df["rsi_14"] = compute_rsi(df["Close"])

    if "macd" in indicators:
        macd_df = compute_macd(df["Close"])
        df = pd.concat([df, macd_df], axis=1)

    if "sma_10" in indicators:
        df["sma_10"] = compute_sma(df["Close"], window=10)

    if "sma_50" in indicators:
        df["sma_50"] = compute_sma(df["Close"], window=50)

    if "bbands" in indicators:
        bb_df = compute_bbands(df["Close"])
        df = pd.concat([df, bb_df], axis=1)

    n_indicator_cols = len(df.columns) - len(["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "trading_date", "title", "description", "source"])
    logger.info(f"Added technical indicators, dataset now has {len(df.columns)} columns")

    return df