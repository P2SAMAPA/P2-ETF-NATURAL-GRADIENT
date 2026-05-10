"""
Main training script for natural gradient allocation with top‑5 forced concentration.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from natural_gradient import NaturalGradientAllocator
import push_results

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

        if len(returns) < config.LOOKBACK_WINDOW:
            print(f"  Insufficient data for {universe_name}, skipping.")
            continue

        train_returns = returns.iloc[-config.LOOKBACK_WINDOW:].values
        n_assets = len(tickers)

        ng = NaturalGradientAllocator(
            n_assets=n_assets,
            learning_rate=config.LEARNING_RATE,
            fisher_damp=config.FISHER_DAMP,
            max_iter=config.MAX_ITER,
            tol=config.TOLERANCE,
            risk_free_rate=config.RISK_FREE_RATE,
            downside_threshold=config.DOWNSIDE_THRESHOLD,
            transaction_cost=config.TRANSACTION_COST
        )
        ng.fit(train_returns)
        weights = ng.predict_weights()

        # --- Force allocation into top 5 ETFs only ---
        top5_idx = np.argsort(weights)[-5:]      # indices of 5 largest weights
        new_weights = np.zeros_like(weights)
        new_weights[top5_idx] = weights[top5_idx]
        new_weights = new_weights / new_weights.sum()   # renormalise to sum to 1
        weights = new_weights

        # Get top picks (now only non‑zero weights)
        top_picks = []
        for i in np.argsort(weights)[::-1]:
            if weights[i] > 0:
                top_picks.append({"ticker": tickers[i], "weight": float(weights[i])})
        top_picks = top_picks[:3]   # show top 3 in dashboard

        universe_results = {
            "weights": {ticker: float(weights[i]) for i, ticker in enumerate(tickers)},
            "top_picks": top_picks,
            "training_end_date": str(returns.index[-1].date()),
            "n_assets": n_assets,
            "lookback_days": config.LOOKBACK_WINDOW,
            "forced_top5": True
        }
        all_results[universe_name] = universe_results

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/natural_gradient_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Natural Gradient allocation complete (top‑5 forced) ===")

if __name__ == "__main__":
    main()
