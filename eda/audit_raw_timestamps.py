import sys
from pathlib import Path

import pandas as pd

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from template.prelude_template import load_polymarket_data


def audit_timestamps():
    """Audit Polymarket timestamps after loader fixes are applied.
    
    Uses load_polymarket_data() to verify timestamps are correct
    from the application's perspective.
    """
    data = load_polymarket_data()
    
    if not data:
        print("No Polymarket data found.")
        return
    
    # Map loader keys to original filenames for display
    key_to_filename = {
        "markets": "finance_politics_markets.parquet",
        "odds_history": "finance_politics_odds_history.parquet",
        "summary": "finance_politics_summary.parquet",
        "tokens": "finance_politics_tokens.parquet",
        "trades": "finance_politics_trades.parquet",
        "event_stats": "finance_politics_event_stats.parquet",
    }
    
    print(f"\n{'File':<40} {'Column':<15} {'Pre-2020 Rows':<15} {'Valid Rows':<15}")
    print("-" * 85)
    
    threshold = pd.Timestamp("2020-01-01")
    
    for key, df in data.items():
        filename = key_to_filename.get(key, f"{key}.parquet")
        
        for col in df.columns:
            if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                # Ensure it's a datetime type for comparison
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                
                # Count non-NaT values and pre-2020 values
                non_nat = df[col].dropna()
                valid_count = len(non_nat)
                pre_2020 = (non_nat < threshold).sum()
                
                print(f"{filename:<40} {col:<15} {pre_2020:<15} {valid_count:<15}")


if __name__ == "__main__":
    audit_timestamps()
