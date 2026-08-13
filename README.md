# Stock Movement Prediction: ML + NLP + Deep Learning Pipeline

Predicts next-day stock price direction (up/down) by combining technical
indicators, FinBERT-based news sentiment analysis, and a comparison
between XGBoost and LSTM models — served through a FastAPI backend and
a Streamlit dashboard.

**This project prioritizes honest, defensible evaluation over inflated
accuracy numbers.** Every result below is reported alongside a naive
baseline for fair context. See [Results](#results--honest-evaluation)
below.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  yfinance   │     │   NewsAPI    │     │                    │
│ (OHLCV data)│     │ (headlines)  │     │                    │
└──────┬──────┘     └──────┬───────┘     │                    │
       │                   │             │                    │
       ▼                   ▼             │                    │
┌─────────────────────────────────┐      │   Data Pipeline    │
│   clean_data.py                 │      │                    │
│   - Aligns news to next trading │      │                    │
│     day (weekends/holidays)     │      │                    │
│   - Left join preserves all     │      │                    │
│     trading days                │      │                    │
└──────────────┬───────────────────┘     │                    │
               ▼                          │                    │
┌─────────────────────────────────┐      │                    │
│   build_dataset.py               │      │                    │
│   - Technical indicators (RSI,   │      │                    │
│     MACD, Bollinger Bands)       │      │                    │
│   - FinBERT sentiment scoring    │      │                    │
│   - Lookahead-safe target        │      │                    │
│     (see Methodology below)      │      │                    │
└──────────────┬───────────────────┘     └────────────────────┘
               ▼
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────┐         ┌──────────┐
│ XGBoost │         │   LSTM   │      Modeling & Comparison
│ baseline│         │(PyTorch) │
└────┬────┘         └────┬─────┘
     └──────────┬─────────┘
                ▼
        ┌───────────────┐
        │   MLflow       │      Experiment tracking
        │   tracking     │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  FastAPI       │      Serving layer
        │  (main.py)     │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  Streamlit     │      Dashboard (thin client)
        │  dashboard     │
        └───────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | `yfinance`, NewsAPI |
| Data processing | `pandas`, `numpy` |
| NLP / Sentiment | HuggingFace `transformers`, FinBERT (`ProsusAI/finbert`) |
| Classical ML | `XGBoost`, `scikit-learn` |
| Deep Learning | `PyTorch` (LSTM) |
| Experiment tracking | `MLflow` |
| API | `FastAPI`, `uvicorn` |
| Dashboard | `Streamlit`, `Plotly` |
| Config/Logging | `PyYAML`, `loguru` |

---

## Methodology & Key Design Decisions

This section is the "why," not just the "what" — the reasoning behind
choices that could otherwise look arbitrary.

### 1. Chronological train/test split, never random

A random split would let the model train on future data relative to
what it's tested on — impossible in real trading, where you only ever
have the past to predict the future. `train_test_split_date` in
`config.yaml` defines a hard date boundary; everything before is train,
everything after is test. This same principle is enforced through
hyperparameter tuning too, via `TimeSeriesSplit` instead of standard
k-fold CV.

### 2. Explicit lookahead-bias prevention in target construction

The target (`build_target()` in `build_dataset.py`) is the only place
in the entire pipeline allowed to look into the future — via
`Close.shift(-horizon_days)`. Every feature, by contrast, only ever
looks backward (`shift(positive_n)`, `pct_change()`). The scaffolding
column used to build the target (`future_close`) is dropped immediately
after use, so it can never accidentally leak into the model as a feature.

### 3. News-to-trading-day alignment

News can be published any day; markets only trade on weekdays
(excluding holidays). Weekend/holiday news is attributed to the **next**
available trading day — since that's the first day the market could
realistically react to it. Backdating would violate cause-and-effect;
dropping would discard real signal.

### 4. Normalized features, not raw price levels

Initial models trained on raw OHLC prices performed *worse* than a
naive baseline for AAPL. Root cause: tree-based models split on
absolute thresholds, and a stock's price range shifts permanently over
years (AAPL traded in a completely different range in 2020 vs 2026).
Replacing raw prices with ratios (`Close/SMA_50`, Bollinger Band
position, MACD normalized by price) fixed this — see [Results](#results--honest-evaluation)
for the before/after numbers.

### 5. Train-only feature scaling for the LSTM

The `StandardScaler` used to normalize LSTM inputs is fit exclusively
on training data, then applied (not re-fit) to test data. Fitting on
combined data would let the model's inputs be quietly informed by the
test period's price statistics — a subtle form of lookahead bias.

### 6. Every model is evaluated against a naive baseline

"52% accuracy" means nothing in isolation. A naive baseline (predict
tomorrow repeats today's direction) is computed for every model and
reported side-by-side, so the model's actual, incremental "edge" is
always visible.

---

## Results & Honest Evaluation

### XGBoost baseline: before vs. after feature normalization

| Ticker | Raw prices (accuracy) | Normalized features (accuracy) | Naive baseline |
|---|---|---|---|
| AAPL | 45.6% (below naive) | 51.6% | 51.6% |
| TSLA | 50.1% | 51.9% | 50.1% |
| NVDA | 50.2% | 52.4% | 47.6% |

### XGBoost (tuned) vs. LSTM comparison

| Ticker | XGBoost (untuned) edge | XGBoost (tuned) edge | LSTM edge |
|---|---|---|---|
| AAPL | +0.0% | -2.0% | +0.95% |
| TSLA | +1.8% | +4.6% | +0.47% |
| NVDA | +4.8% | +5.8% | +6.8% |

**Interpretation:** XGBoost generally outperforms the LSTM on this
dataset, most likely because the training set (~957 rows per ticker)
is small — LSTMs typically need substantially more data to reliably
out-learn tree-based models. NVDA is the interesting exception, where
the LSTM's sequence-modeling edge outperformed tuned XGBoost, possibly
reflecting more genuine multi-day momentum structure in NVDA's price
action during this period versus day-to-day noise dominating for
AAPL/TSLA. This is a hypothesis, not a proven causal claim.

**Hyperparameter tuning is not a free win:** tuning improved TSLA and
NVDA but made AAPL *worse* (0.0% → -2.0% edge) — most likely because
with only ~190 rows per validation fold, the search overfit to
specific patterns in AAPL's training window that didn't generalize to
the test period. This is reported as-is rather than cherry-picking the
best-looking number per ticker.

### Feature importance finding: sentiment underperforms

Across all three tickers, **no `sentiment_*` feature appears in the top
15 most important features** — technical/price-derived features
dominate entirely. This is most likely explained by a real limitation:
NewsAPI's free tier only returns the last 30 days of articles, while
the training window spans 2020–2023. Sentiment was effectively `NaN`
(filled as neutral) for the overwhelming majority of training rows,
giving the model little real signal to learn from. **With a paid news
API providing full historical coverage, sentiment's actual contribution
could be properly evaluated** — this is a clearly identified path for
future improvement, not a dead end.

---

## Known Limitations

- **News data coverage**: NewsAPI free tier limits historical news to
  the last 30 days, meaning sentiment features have minimal signal
  across most of the training period (see finding above).
- **Data source reliability**: `yfinance` scrapes an unofficial Yahoo
  Finance API and has broken in production before due to upstream
  changes (encountered and fixed during this project — see commit
  history), a real risk of any free, unofficial data source.
- **`/predict-latest` staleness**: this endpoint serves predictions
  based on whatever the last pipeline run produced — it is not live
  real-time data. A fully live system would need a scheduled pipeline
  run (e.g., via Airflow or cron) to keep data fresh.
- **Small dataset for deep learning**: ~957 training rows per ticker
  is modest for an LSTM; results might favor deep learning more with a
  larger dataset (more tickers, longer history).
- **This is not investment advice**: predicting next-day direction
  with a small, defensible edge over chance is very different from a
  profitable trading strategy once transaction costs, slippage, and
  risk are accounted for.

---

## Project Structure

```
stock-predictor-ml/
├── src/
│   ├── data/           # fetch_prices.py, fetch_news.py, clean_data.py
│   ├── features/        # technical_indicators.py, sentiment.py, build_dataset.py
│   ├── models/           # train_baseline.py, tune_baseline.py, train_lstm.py
│   ├── api/               # main.py (FastAPI)
│   ├── dashboard/          # app.py (Streamlit)
│   └── utils/               # config.py, logger.py
├── config/config.yaml    # all tunable settings, nothing hardcoded
├── tests/                # pytest suite
├── data/                 # raw/ and processed/ (gitignored, regenerable)
└── models_store/         # trained models + feature importance plots (gitignored)
```

---

## Running This Project

**1. Setup**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then add your NEWSAPI_KEY
```

**2. Run the full pipeline**
```bash
python -m src.data.fetch_prices
python -m src.data.fetch_news
python -m src.data.clean_data
python -m src.features.build_dataset
python -m src.models.train_baseline
```

**3. Serve predictions**
```bash
uvicorn src.api.main:app --reload
# API docs at http://127.0.0.1:8000/docs
```

**4. Launch the dashboard** (in a second terminal, with the API running)
```bash
streamlit run src/dashboard/app.py
```

---

## What I'd Do With More Time

- Integrate a paid news API for full historical coverage, and properly
  re-evaluate whether sentiment adds real signal once it has enough
  history to learn from
- Add a scheduled orchestration layer (Airflow/Prefect) to keep data
  and predictions genuinely live
- Expand to more tickers to give the LSTM a fairer, larger-data
  comparison against XGBoost
- Backtest with realistic transaction costs and position sizing to
  evaluate this as an actual trading signal, not just directional
  accuracy in isolation