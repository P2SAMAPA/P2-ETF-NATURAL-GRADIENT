"""
Main training script for natural gradient allocation.
Computes both Global (full history) and Last 252 Days portfolios.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from natural_gradient import NaturalGradientAllocator
import push_results

def compute_allocation(returns_matrix, tickers, mode_name):
    """Run natural gradient optimisation on a returns matrix."""
    n_assets = len(tickers)
    ng = NaturalGradientAllocator(
        n_assets=n_assets,
        learning_rate=config.LEARNING_RATE,
        fisher_damp=config.FISHER_DAMP,
        max_iter=config.MAX_ITER,
        tol=config.TOLERANCE,
        risk_free_rate=config.RISK_FREE_RATE,
        transaction_cost=config.TRANSACTION_COST
    )
    ng.fit(returns_matrix)
    raw_weights = ng.predict_weights()

    # Keep only top 5 ETFs (optional, you can comment out)
    top5_idx = np.argsort(raw_weights)[-5:]
    top5_raw = raw_weights[top5_idx]
    # Smoothing power (0.3 gives gradient; set to 1.0 for raw)
    smoothing_power = 0.1
    smoothed = top5_raw ** smoothing_power
    final_weights = smoothed / smoothed.sum()
    weights = np.zeros(n_assets)
    weights[top5_idx] = final_weights

    top_picks = []
    for i in np.argsort(weights)[::-1]:
        if weights[i] > 0:
            top_picks.append({"ticker": tickers[i], "weight": float(weights[i])})
    top_picks = top_picks[:3]

    return {
        "weights": {ticker: float(weights[i]) for i, ticker in enumerate(tickers)},
        "top_picks": top_picks,
        "mode": mode_name,
        "n_assets": n_assets,
        "lookback_days": len(returns_matrix) if mode_name == "Global" else config.LOOKBACK_WINDOW,
        "smoothing_power": smoothing_power,
        "forced_top5": True
    }

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        # --- Global allocation (full history) ---
        if len(returns) < 2:
            print(f"  Not enough data for global, skipping.")
            global_res = None
        else:
            # Use all available returns from 2008 onwards
            full_returns = returns.values
            global_res = compute_allocation(full_returns, tickers, "Global")
            print(f"  Global: top pick {global_res['top_picks'][0]['ticker']}")

        # --- Last 252 days allocation ---
        if len(returns) < config.LOOKBACK_WINDOW:
            print(f"  Insufficient data for last {config.LOOKBACK_WINDOW} days, skipping.")
            last_res = None
        else:
            recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:].values
            last_res = compute_allocation(recent_returns, tickers, "Last 252 Days")
            print(f"  Last 252: top pick {last_res['top_picks'][0]['ticker']}")

        universe_entry = {}
        if global_res:
            universe_entry["global"] = global_res
        if last_res:
            universe_entry["last_252"] = last_res
        if universe_entry:
            all_results[universe_name] = universe_entry

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/natural_gradient_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Natural Gradient allocation complete (Global + Last 252 Days) ===")

if __name__ == "__main__":
    main()
