"""
Configuration for P2-ETF-NATURAL-GRADIENT engine (Sharpe version).
"""

import os
from datetime import datetime

DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-natural-gradient-results"

FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))
UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

LOOKBACK_WINDOW = 252
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.0

LEARNING_RATE = 0.5          # larger step
FISHER_DAMP = 1e-6           # almost no damping
MAX_ITER = 1000              # more iterations
TOLERANCE = 1e-8

RETURN_EMPHASIS = 1.0        # not used in Sharpe version, kept for compatibility

REBALANCE_FREQ = 1
TRAIN_START_DATE = "2008-01-01"

TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
