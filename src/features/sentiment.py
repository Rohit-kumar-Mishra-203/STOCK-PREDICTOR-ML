"""
Score news headlines for sentiment using FinBERT (ProsusAI/finbert),
then aggregate to a daily per-ticker sentiment feature.

Why FinBERT specifically, not a generic sentiment model:
General-purpose sentiment models (e.g., a standard BERT sentiment
classifier) are trained on product reviews or social media - they
misread financial language. "Shares tumbled" or "beat estimates" carry
specific meaning in finance that generic models don't capture well.
FinBERT is pretrained specifically on financial text (analyst reports,
earnings calls) for this reason.

Why we use the model directly via a HuggingFace pipeline instead of
training our own classifier:
Training a sentiment classifier from scratch would need thousands of
hand-labeled financial headlines - expensive and unnecessary. Using a
well-established pretrained model is the correct engineering choice
here: reuse an existing, validated model where one exists.

Design decision: we keep 3 separate probability columns (positive,
negative, neutral) per day rather than collapsing to one compound
score. This preserves more information - e.g. a day with sharply
conflicting headlines looks different from a day with no news at all,
which a single averaged score would hide.
"""

from typing import Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.config import get_config
from src.utils.logger import logger

SENTIMENT_COLS = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]


def load_finbert(model_name: str) -> Tuple:
    """
    Load FinBERT tokenizer and model once.

    Why loaded as a separate function (not inline in the scoring loop):
    model loading is slow (downloads/loads weights) and should happen
    exactly once per run, not repeatedly.
    """
    logger.info(f"Loading sentiment model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # inference mode - disables dropout etc, we're not training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Sentiment model loaded on device: {device}")

    return tokenizer, model, device


def score_headlines_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    """
    Run FinBERT over a list of headlines in batches, returning probability
    scores for positive/negative/neutral per headline.

    Why batching: each forward pass has fixed overhead - batching
    amortizes that across many headlines, substantially faster.

    Why torch.no_grad(): inference only, not training - skips gradient
    tracking we don't need, saving memory and compute.
    """
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu())

    all_probs_tensor = torch.cat(all_probs, dim=0)

    # FinBERT's label order (from its model config) is:
    # 0 = positive, 1 = negative, 2 = neutral
    return pd.DataFrame(all_probs_tensor.numpy(), columns=SENTIMENT_COLS)


def score_news_dataframe(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every headline in a merged ticker DataFrame that has actual
    news text, leaving rows with no news untouched (NaN sentiment).

    Why title only, not title+description: headlines are the concentrated
    signal - descriptions mostly restate them, adding noise/length
    without proportionally more information.
    """
    cfg = get_config()
    sent_cfg = cfg["sentiment"]

    model_name: str = str(sent_cfg["model_name"])
    batch_size: int = int(sent_cfg["batch_size"])
    max_length: int = int(sent_cfg["max_length"])

    tokenizer, model, device = load_finbert(model_name)

    has_news_mask = news_df["title"].notna()
    title_values=news_df.loc[has_news_mask,"title"].astype(str).tolist()
    texts: list[str] = [str(t) for t in title_values]

    news_df = news_df.copy()

    if not texts:
        logger.warning("No headlines to score in this dataset")
        for col in SENTIMENT_COLS:
            news_df[col] = pd.NA
        return news_df

    logger.info(f"Scoring {len(texts)} headlines with FinBERT")

    scores = score_headlines_batch(
        texts, tokenizer, model, device,
        batch_size=batch_size, max_length=max_length,
    )

    for col in SENTIMENT_COLS:
        news_df.loc[has_news_mask, col] = scores[col].values

    return news_df


def aggregate_daily_sentiment(df: pd.DataFrame, agg_method: str = "mean") -> pd.DataFrame:
    """
    Collapse multiple headline-level sentiment rows per trading day into
    one row per day.

    Why groupby + agg rather than dropping duplicate dates: a trading day
    with 5 headlines carries more information than a day with 1 - just
    keeping one row per day would silently discard real signal.
    """
    excluded = set(SENTIMENT_COLS + ["title", "description", "source"])
    all_cols: list[str] = [str(c) for c in df.columns.tolist()]
    non_sentiment_cols: list[str] = [c for c in df.columns if c not in excluded]

    daily_sentiment = df.groupby("Date")[SENTIMENT_COLS].agg(agg_method).reset_index()

    article_counts = df.groupby("Date")["title"].apply(lambda x: x.notna().sum()).reset_index()
    article_counts.columns = ["Date", "news_article_count"]

    base = df[non_sentiment_cols].drop_duplicates(subset="Date")

    result = base.merge(daily_sentiment, on="Date", how="left").merge(article_counts, on="Date", how="left")
    return result


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline step: score headlines, then aggregate to one row per trading day."""
    cfg = get_config()
    agg_method: str = str(cfg["features"]["sentiment_agg"])

    scored = score_news_dataframe(df)
    daily = aggregate_daily_sentiment(scored, agg_method=agg_method)
    logger.info(f"Aggregated to {len(daily)} daily rows with sentiment features")
    return daily