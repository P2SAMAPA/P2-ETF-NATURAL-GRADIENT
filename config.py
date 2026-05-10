"""
Configuration for P2-ETF-NATURAL-GRADIENT engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-natural-gradient-results"

# --- Universe definitions (same as working repo) ---
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

# --- Macro features (not used directly, but kept for compatibility) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Natural Gradient hyperparameters ---
LOOKBACK_WINDOW = 252      # days used to compute objective and Fisher
TRANSACTION_COST = 0.001   # 10 bps
RISK_FREE_RATE = 0.0       # annualised, used for Sortino (set to 0 for absolute return)
DOWNSIDE_THRESHOLD = 0.0   # returns below this are considered downside

LEARNING_RATE = 0.1        # natural gradient step size
FISHER_DAMP = 1e-4         # damping for Fisher matrix inversion
MAX_ITER = 500             # max iterations for natural gradient optimisation
TOLERANCE = 1e-6           # early stopping tol

# New: higher value → more focus on return, less on downside risk
# Start with 1.5, increase to 2.0, 3.0, or even 10.0 for aggressive return‑seeking
RETURN_EMPHASIS = 1.5

# --- Training schedule ---
REBALANCE_FREQ = 1         # daily rebalancing
TRAIN_START_DATE = "2008-01-01"

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
