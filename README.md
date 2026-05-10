# P2-ETF-NATURAL-GRADIENT

**Natural Gradient portfolio allocation** using the Fisher information metric to optimise Sortino ratio.

## Features

- Information geometry: natural gradient descent on the simplex of portfolio weights.
- Sortino ratio objective (downside deviation).
- Daily rebalancing, transaction costs.
- Works on three ETF universes (FI_COMMODITIES, EQUITY_SECTORS, COMBINED).
- Data from `P2SAMAPA/fi-etf-macro-signal-master-data` (2008–2026 YTD).
- Results pushed to `P2SAMAPA/p2-etf-natural-gradient-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-NATURAL-GRADIENT.git
cd P2-ETF-NATURAL-GRADIENT
pip install -r requirements.txt
