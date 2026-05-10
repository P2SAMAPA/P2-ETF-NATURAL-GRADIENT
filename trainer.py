"""
Main training script for natural gradient allocation.
For each universe, optimise portfolio weights over rolling windows and save recommendations.
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

    # Load data
    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        # Ensure enough data
        if len(returns) < config.LOOKBACK_WINDOW:
            print(f"  Insufficient data for {universe_name}, skipping.")
            continue

        # Use only the last LOOKBACK_WINDOW days for training
        train_returns = returns.iloc[-config.LOOKBACK_WINDOW:].values
        n_assets = len(tickers)

        # Train natural gradient allocator
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
        top_picks = ng.top_etfs(tickers, top_n=3)

        # Prepare output
        universe_results = {
            "weights": {ticker: float(weights[i]) for i, ticker in enumerate(tickers)},
            "top_picks": [{"ticker": t, "weight": w} for t, w in top_picks],
            "training_end_date": str(returns.index[-1].date()),
            "n_assets": n_assets,
            "lookback_days": config.LOOKBACK_WINDOW
        }
        all_results[universe_name] = universe_results

    # Save results locally
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/natural_gradient_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    # Push to Hugging Face
    push_results.push_daily_result(local_path)

    print("\n=== Natural Gradient allocation complete ===")

if __name__ == "__main__":
    main()
