# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polish stock market (GPW - Giełda Papierów Wartościowych) analysis toolkit. The codebase and all identifiers (functions, variables, log messages) are written in **Polish**.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run technical analysis for a single stock (outputs <TICKER>_stock_analysis.png)
python analizator.py CDR.WA --period 5y
python analizator.py AAPL --period 1y --output my_chart.png

# Run ML backtester (RandomForest strategy vs. buy-and-hold)
python backtester.py CDR.WA --kapital 10000

# Run WIG20 market scanner (outputs skan_<YYYY-MM-DD_HHMM>.csv)
python skaner_gpw.py

# Lint (CI uses these exact commands)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run tests
pytest
```

## Architecture

`indicators.py` is the **shared core library** imported by all other scripts. It exposes two kinds of functions:
- Individual calculators: `oblicz_sma`, `oblicz_rsi`, `oblicz_macd`, `oblicz_bollinger`
- `dodaj_wszystkie_wskazniki(data)` — adds every indicator column to a DataFrame at once

**Data flow across modules:**

| Script | Imports from indicators.py | Output |
|---|---|---|
| `analizator.py` | `dodaj_wszystkie_wskazniki` | 4-panel PNG chart (price+BB, volume, RSI, MACD) |
| `backtester.py` | `dodaj_wszystkie_wskazniki` | Console metrics + ML vs. buy-and-hold comparison |
| `skaner_gpw.py` | `oblicz_sma`, `oblicz_rsi` only | timestamped CSV of signals |

**backtester.py ML details:**
- Model: `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')`
- Target: whether price rises >3% within 30 days (`future_window=30`, `target_percentage=1.03`)
- Features (10): `RSI`, `SMA_diff`, `MACD_diff`, `Volatility`, `BB_Position`, `SMA50_dist`, `SMA200_dist`, `ROC_10`, `ROC_30`, `Returns`
- Split: 80% train / 20% test + 5-fold `TimeSeriesSplit` cross-validation
- Buy threshold: probability ≥ 0.4 (intentionally below 0.5 to bias toward buy signals)
- Commission constant: `PROWIZJA = 0.003` (0.3% per trade)

**skaner_gpw.py** scans a hardcoded list of WIG20 tickers (20 stocks). Signals detected: Golden Cross / Death Cross (SMA50 vs SMA200 crossover in last 5 days) and RSI oversold (<30) / overbought (>70).

## Key Conventions

- Polish stock tickers use `.WA` suffix (e.g., `CDR.WA`, `PKO.WA`); international stocks use standard symbols.
- yfinance sometimes returns MultiIndex columns for a single ticker — all scripts flatten these with `data.columns = data.columns.get_level_values(0)`.
- Output files (`*.png`, `*.csv`) are gitignored and saved to the script's directory via `os.path.dirname(os.path.abspath(__file__))`.
- `_jako_float()` in `backtester.py` is a utility to safely unwrap pandas/numpy scalar types — use it when extracting single values from DataFrames in that module.
- CI (`.github/workflows/python-app.yml`) runs on Python 3.10, uses `flake8` + `pytest`, and triggers on pushes/PRs to `master`.
