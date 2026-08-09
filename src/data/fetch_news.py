"""
Fetch financial news headlines for configured tickers using NewsAPI.

Why NewsAPI:
Free tier available, simple REST API, good enough for a portfolio project.
Limitation worth stating explicitly: free tier only returns articles from
the last month and caps at 100 requests/day — a real production system
would need a paid tier or a different provider (e.g., Alpha Vantage News,
Benzinga, or a dedicated financial news API) for full historical coverage.

Why we fetch news separately from prices:
Different API, different rate limits, different failure modes. Keeping
this isolated means a news API outage doesn't block price data collection,
and vice versa — each piece of the pipeline can fail independently without
taking down the whole system.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger

load_dotenv()

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"


def fetch_news_for_ticker(ticker: str, company_name: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    """
    Fetch news headlines mentioning a ticker/company over the lookback window.

    We search by company name rather than ticker symbol alone, since
    searching "TSLA" as a raw string misses most articles that refer to
    "Tesla" by name — a subtle but important detail for search quality.
    """
    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.info(f"Fetching news for {ticker} ({company_name}) since {from_date}")

    params = {
        "q": company_name,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,  # max allowed on free tier
        "apiKey": api_key,
    }

    response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)

    if response.status_code != 200:
        # Fail loudly with the actual API error message, not a generic exception -
        # makes debugging rate limits vs bad requests vs auth failures much faster.
        raise RuntimeError(f"NewsAPI request failed for {ticker}: {response.status_code} - {response.text}")

    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        logger.warning(f"No news articles found for {ticker}")
        return pd.DataFrame(columns=["date", "ticker", "title", "description", "source"])

    rows = []
    for article in articles:
        rows.append({
            "date": article["publishedAt"][:10],  # keep just YYYY-MM-DD, drop time
            "ticker": ticker,
            "title": article.get("title", ""),
            "description": article.get("description", "") or "",
            "source": article.get("source", {}).get("name", "unknown"),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Fetched {len(df)} articles for {ticker}")
    return df


def fetch_all_news(ticker_names: dict[str, str], lookback_days: int, api_key: str) -> pd.DataFrame:
    """
    Fetch news for all configured tickers and combine into one DataFrame.

    Why one combined DataFrame here (unlike prices, which we kept separate
    per ticker): news volume is small enough that one file is manageable,
    and downstream sentiment scoring processes all headlines together
    in batches anyway - a combined file avoids unnecessary complexity.
    """
    all_dfs = []

    for ticker, company_name in ticker_names.items():
        try:
            df = fetch_news_for_ticker(ticker, company_name, lookback_days, api_key)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")

        # NewsAPI free tier rate limit courtesy delay - avoids hammering
        # the API back-to-back across multiple tickers in the same run.
        time.sleep(1)

    if not all_dfs:
        raise RuntimeError("No news data was fetched for any ticker.")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def save_raw_news(df: pd.DataFrame) -> None:
    cfg = get_config()
    save_dir = resolve_path(cfg["paths"]["raw_news"])
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / f"news_{datetime.today().strftime('%Y%m%d')}.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} news rows to {filepath}")


def main():
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise EnvironmentError(
            "NEWSAPI_KEY not found. Did you create a .env file from .env.example and add your key?"
        )

    cfg = get_config()

    # Map tickers to full company names for better search results.
    # Hardcoded here since it's small and stable - if the ticker list grows
    # significantly, this would move into config.yaml instead.
    ticker_names = {
        "AAPL": "Apple",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
    }

    tickers: list[str] = cfg["data"]["tickers"]
    ticker_names: dict[str, str]= {t: ticker_names.get(t, t) for t in tickers}

    lookback_days = cfg["data"]["news_lookback_days"]

    logger.info(f"Starting news fetch for {len(tickers)} tickers")

    df = fetch_all_news(ticker_names, lookback_days, api_key)
    save_raw_news(df)

    logger.info("News fetch complete.")


if __name__ == "__main__":
    main()