import numpy as np
import pandas as pd
import polars as pl
import logging
from pathlib import Path

from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)
import eda.eda_starter_template as eda_basics

PRICE_COL = "PriceUSD"
DYNAMIC_STRENGTH = 4.0 
MIN_W = 1e-6

# Markov Thresholds
STR_THR = 0.03
NEU_THR = 0.005

FEATS = [
    "dist_sma_20",
    "macd",
    "rsi_14",
    "stochastic_14",
    "markov_bull_prob",
    "polymarket_sentiment"
]

def compute_markov_bull_probs(df: pl.DataFrame) -> pl.Series:
    """
    Computes daily probabilities of a bullish move using a 5x5 transition matrix.
    Logic derived from build_markov_likelihood in EDA.ipynb.
    """
    # 1. Define states based on daily returns
    df = df.with_columns([
        ((pl.col(PRICE_COL) / pl.col(PRICE_COL).shift(1)) - 1).alias("daily_ret")
    ])
    
    df = df.with_columns(
        pl.when(pl.col("daily_ret") < -STR_THR).then(pl.lit(0))     # Str Bear
        .when(pl.col("daily_ret") < -NEU_THR).then(pl.lit(1))       # Bear
        .when(pl.col("daily_ret") <= NEU_THR).then(pl.lit(2))       # Neutral
        .when(pl.col("daily_ret") <= STR_THR).then(pl.lit(3))       # Bull
        .otherwise(pl.lit(4))                                       # Str Bull
        .alias("state")
    )

    # 2. Build 5x5 Transition Matrix
    states = df.select("state").to_series().to_list()
    matrix = np.zeros((5, 5))
    for i in range(len(states) - 1):
        curr, next_s = states[i], states[i+1]
        if curr is not None and next_s is not None:
            matrix[int(curr)][int(next_s)] += 1
            
    # Normalize rows to get probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    transition_probs = matrix / row_sums
    
    # 3. Probability of (Bull + Str Bull) for each state
    bull_probs_by_state = transition_probs[:, 3:].sum(axis=1)
    
    return df.select(
        pl.col("state").map_elements(lambda s: bull_probs_by_state[int(s)] if s is not None else 0.5, return_dtype=pl.Float64)
    ).to_series()

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw CoinMetrics data into model features.
    Utilizes Polars for efficient computation as seen in EDA.ipynb.
    """
    ldf = pl.from_pandas(df.reset_index())
    
    # 1. Technical Indicators
    ldf = ldf.with_columns([
        pl.col(PRICE_COL).rolling_mean(window_size=20).alias("sma_20"),
        (pl.col(PRICE_COL).ewm_mean(span=12) - pl.col(PRICE_COL).ewm_mean(span=26)).alias("macd")
    ])
    
    ldf = ldf.with_columns([
        ((pl.col(PRICE_COL) / pl.col("sma_20")) - 1).alias("dist_sma_20")
    ])
    
    # RSI 14
    diff = pl.col(PRICE_COL).diff()
    gain = diff.clip(lower_bound=0).rolling_mean(14)
    loss = diff.clip(upper_bound=0).abs().rolling_mean(14)
    ldf = ldf.with_columns((100 - (100 / (1 + (gain / loss)))).alias("rsi_14"))
    
    # Stochastic 14
    low_14 = pl.col(PRICE_COL).rolling_min(14)
    high_14 = pl.col(PRICE_COL).rolling_max(14)
    ldf = ldf.with_columns(((pl.col(PRICE_COL) - low_14) / (high_14 - low_14) * 100).alias("stochastic_14"))
    
    # 2. Markov Probabilities
    ldf = ldf.with_columns(compute_markov_bull_probs(ldf).alias("markov_bull_prob"))
    
    # 3. Polymarket Sentiment
    ldf = ldf.with_columns(pl.lit(0.5).alias("polymarket_sentiment"))
    
    feature_cols = ["dist_sma_20", "macd", "rsi_14", "stochastic_14", "markov_bull_prob", "polymarket_sentiment"]
    ldf = ldf.with_columns([pl.col(c).shift(1).fill_null(strategy="forward").alias(c) for c in feature_cols])
    
    return ldf.to_pandas().set_index("time")


def compute_dynamic_multiplier(features_df: pd.DataFrame) -> np.ndarray:
    """
    Combines features into a single weight multiplier.
    Uses exponential scaling to adjust DCA amounts.
    """

    sig_rsi = (50 - features_df["rsi_14"]) / 50.0 
    sig_sma = -features_df["dist_sma_20"] * 5.0  
    sig_markov = (features_df["markov_bull_prob"] - 0.4) * 5.0 
    
    # Weighted Signal Combination
    combined = (sig_rsi * 0.3) + (sig_sma * 0.4) + (sig_markov * 0.3)
    
    multiplier = np.exp(combined * DYNAMIC_STRENGTH)
    return np.clip(multiplier, 0.1, 10.0) 


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Main API for the backtester to compute weights for a 1-year window.
    Ensures summation to 1.0 and past-weight locking.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = features_df.reindex(full_range).fillna(method="ffill").fillna(0)
    
    n = len(df)
    base_pdf = np.ones(n) / n
    
    dyn_multiplier = compute_dynamic_multiplier(df)
    raw_weights = base_pdf * dyn_multiplier
    
    # Determine the split between past (locked) and future (uniform)
    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    
    weights = allocate_sequential_stable(raw_weights, n_past, locked_weights)
    
    return pd.Series(weights, index=full_range)