import numpy as np
import pandas as pd
import polars as pl
from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)

PRICE_COL = "PriceUSD"
MVRV_COL = "CapMVRVCur"
NVT_COL = "NVTAdj"

DYNAMIC_STRENGTH = 3.0
MIN_W = 1e-6

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactored for Value-Based Accumulation.
    Combines MVRV, NVT, and Technical indicators with zero look-ahead bias.
    """
    if "CapMrktCurUSD" in df.columns and "TxTfrValAdjUSD" in df.columns:
        df["NVTAdj"] = df["CapMrktCurUSD"] / df["TxTfrValAdjUSD"]
    else:
        print("Required columns for NVT calculation missing.")
        df["NVTAdj"] = 0

    ldf = pl.from_pandas(df.reset_index())
    
    # 1. Technical Indicators
    ldf = ldf.with_columns([
        pl.col(PRICE_COL).rolling_mean(window_size=200).alias("sma_200")
    ])
    ldf = ldf.with_columns([
        ((pl.col(PRICE_COL) / pl.col("sma_200")) - 1).alias("dist_sma_200")
    ])
    
    # RSI 14 (Short-term exhaustion)
    diff = pl.col(PRICE_COL).diff()
    gain = diff.clip(lower_bound=0).rolling_mean(14)
    loss = diff.clip(upper_bound=0).abs().rolling_mean(14)
    ldf = ldf.with_columns((100 - (100 / (1 + (gain / loss)))).alias("rsi_14"))
    
    # 2. On-Chain Normalization
    ldf = ldf.with_columns([
        pl.col(NVT_COL).rolling_median(window_size=90).alias("nvt_median")
    ])
    
    # 3. Create Signals (Positive = Undervalued / Buy More)
    ldf = ldf.with_columns([
        (1.5 - pl.col(MVRV_COL)).alias("mvrv_signal"),
        ((pl.col("nvt_median") - pl.col(NVT_COL)) / pl.col("nvt_median")).alias("nvt_signal"),
        (-pl.col("dist_sma_200")).alias("sma_200_signal"),
        ((50 - pl.col("rsi_14")) / 50.0).alias("rsi_signal")
    ])
    
    feature_cols = ["mvrv_signal", "nvt_signal", "sma_200_signal", "rsi_signal", PRICE_COL]
    ldf = ldf.with_columns([
        pl.col(c).shift(1).fill_null(strategy="backward").alias(c) for c in feature_cols
    ])
    
    return ldf.to_pandas().set_index("time").fillna(0)

def compute_dynamic_multiplier(features_df: pd.DataFrame) -> np.ndarray:
    """
    Computes a single weighted multiplier from multiple value-based signals.
    """

    w_mvrv = 0.40 
    w_nvt = 0.20 
    w_sma = 0.20  
    w_rsi = 0.20 
    
    combined = (
        features_df["mvrv_signal"] * w_mvrv +
        features_df["nvt_signal"] * w_nvt +
        features_df["sma_200_signal"] * w_sma +
        features_df["rsi_signal"] * w_rsi
    )
    
    multiplier = np.exp(combined * DYNAMIC_STRENGTH)
    
    return np.clip(multiplier, 0.1, 8.0) 

def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Main API for the backtester.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = features_df.loc[:current_date].reindex(full_range).ffill().fillna(0)
    
    n = len(df)
    base_pdf = np.ones(n) / n
    
    dyn_multiplier = compute_dynamic_multiplier(df)
    raw_weights = base_pdf * dyn_multiplier
    
    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    
    weights = allocate_sequential_stable(raw_weights, n_past, locked_weights)
    
    return pd.Series(weights, index=full_range)