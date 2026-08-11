"""
Streamlit dashboard - a thin client over the FastAPI serving layer.

Why this calls the API rather than loading models/data directly:
The API is the single source of truth for predictions and data access.
If this dashboard loaded models independently, we'd have two paths that
could silently drift apart (different feature ordering, different model
versions) - going through the same API the rest of the world would use
keeps this dashboard honest about actually testing the real served
system, not a separate shortcut.

Run with: streamlit run src/dashboard/app.py
(Requires the API to be running separately: uvicorn src.api.main:app)
"""

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock Movement Predictor", layout="wide")

st.title("Stock Movement Prediction Dashboard")
st.caption(
    "Predicts next-day price direction using XGBoost trained on technical "
    "indicators + FinBERT sentiment. See project README for honest "
    "evaluation and known limitations."
)


@st.cache_data(ttl=60)
def fetch_prediction(ticker: str):
    """
    Cached for 60 seconds - avoids hammering the API if the user
    re-selects the same ticker repeatedly within a short window, while
    still refreshing reasonably often if the underlying pipeline reruns.
    """
    response = requests.get(f"{API_BASE_URL}/predict-latest/{ticker}", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def fetch_history(ticker: str, days: int = 90):
    response = requests.get(f"{API_BASE_URL}/history/{ticker}", params={"days": days}, timeout=10)
    response.raise_for_status()
    return pd.DataFrame(response.json())


@st.cache_data(ttl=300)
def fetch_available_tickers():
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json().get("models_loaded", [])


# --- Sidebar ---
st.sidebar.header("Settings")

try:
    available_tickers = fetch_available_tickers()
except requests.exceptions.ConnectionError:
    st.error(
        "Cannot reach the prediction API. Make sure it's running: "
        "`uvicorn src.api.main:app --reload`"
    )
    st.stop()

if not available_tickers:
    st.warning("API is running but no models are loaded. Run train_baseline.py first.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Select ticker", available_tickers)
history_days = st.sidebar.slider("History window (days)", min_value=30, max_value=250, value=90)

# --- Main content ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"{selected_ticker} — Next-Day Prediction")

    try:
        prediction = fetch_prediction(selected_ticker)
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    direction = prediction["predicted_direction"]
    prob_up = prediction["probability_up"]

    if direction == "up":
        st.metric("Predicted Direction", "▲ UP", delta=f"{prob_up:.1%} confidence")
    else:
        st.metric("Predicted Direction", "▼ DOWN", delta=f"{1 - prob_up:.1%} confidence", delta_color="inverse")

    st.progress(prob_up, text=f"P(up) = {prob_up:.1%}")

    st.caption(
        "This reflects a modest statistical edge over a naive baseline "
        "(see project evaluation) - not a guaranteed outcome. Markets "
        "are inherently close to unpredictable at daily granularity."
    )

with col2:
    st.subheader(f"{selected_ticker} — Price History ({history_days}d)")

    hist_df = fetch_history(selected_ticker, history_days)
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["Close"],
        mode="lines", name="Close Price", line=dict(color="#1f77b4"),
    ))
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None, yaxis_title="Price ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sentiment section ---
st.subheader(f"{selected_ticker} — Sentiment Signal (last {history_days}d)")

sent_fig = go.Figure()
sent_fig.add_trace(go.Bar(
    x=hist_df["Date"], y=hist_df["sentiment_positive"],
    name="Positive", marker_color="#2ca02c",
))
sent_fig.add_trace(go.Bar(
    x=hist_df["Date"], y=-hist_df["sentiment_negative"],
    name="Negative", marker_color="#d62728",
))
sent_fig.update_layout(
    height=250, margin=dict(l=20, r=20, t=20, b=20),
    barmode="relative", yaxis_title="Sentiment score",
)
st.plotly_chart(sent_fig, use_container_width=True)

st.caption(
    "Note: sentiment ranked low in feature importance during model "
    "evaluation, likely due to limited historical news coverage from "
    "the free-tier API (see README for details)."
)