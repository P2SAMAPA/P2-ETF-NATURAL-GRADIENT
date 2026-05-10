"""
Configuration for P2-ETF-NATURAL-GRADIENT engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-natural-gradient-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

# --- Macro features (compatibility) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Natural Gradient hyperparameters ---
LOOKBACK_WINDOW = 252
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.0
DOWNSIDE_THRESHOLD = 0.0

LEARNING_RATE = 0.1
FISHER_DAMP = 1e-4
MAX_ITER = 500
TOLERANCE = 1e-6

# Return emphasis: very high (e.g., 50 or 100) virtually eliminates downside penalty
RETURN_EMPHASIS = 50.0    # try 50, 100, or 200 for aggressive return‑seeking

# --- Training schedule ---
REBALANCE_FREQ = 1
TRAIN_START_DATE = "2008-01-01"

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
