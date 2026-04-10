import logging
import sys
import pandas as pd
from pathlib import Path

root_path = str(Path(__file__).parent.parent)

if root_path not in sys.path:
    sys.path.append(root_path)

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

from model_development import precompute_features, compute_window_weights

_FEATURES_DF = None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for the Technical/Markov model.
    
    Adapts the specific model function to the interface expected 
    by the template backtest engine.
    """
    global _FEATURES_DF
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    
    current_date = end_date
    
    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)

def main():
    global _FEATURES_DF
    
    # 1. Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Analysis - Technical & Markov Model")
    
    # 2. Load Data
    btc_df = load_data()
    
    # 3. Precompute Features
    logging.info("Precomputing Technical indicators and Markov 5x5 state probabilities...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 4. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output_technical_markov"
    
    # 5. Run Full Analysis Pipeline
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Technical + Markov DCA",
    )

if __name__ == "__main__":
    main()