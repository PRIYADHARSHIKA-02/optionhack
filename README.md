# Introductory Options Hackathon(IIT Guwahati)

## Overview
This repository contains my submission for the **Introductory Options Hackathon** (IIT Guwahati).

The project builds a transparent, educational workflow to:
- estimate a **fair value** for a live listed option using a **Black–Scholes** model,
- compute and interpret key **Greeks** (Delta, Theta, Vega),
- highlight the **primary risks** of holding an option position,
- propose a **basic hedge** (Delta hedging with the underlying),
- run **scenario analyses** (volatility shocks, time decay),
- show **call vs put price comparison** at the same strike,
- generate plots to support an analyst-style explanation.

The analysis uses **12 months of historical price data** (via `yfinance`) to estimate annualized historical volatility.

---

## Problem Statement (Hackathon Brief)
You are a junior analyst at an equity-focused hedge fund that uses options to manage risk and generate returns.
Option prices change due to:
- movements in the underlying price,
- time decay,
- changes in market uncertainty (volatility).

The goal is to provide a clear framework to:
- estimate a fair option value,
- understand key risks (Greeks),
- hedge or adjust positions as conditions change,
- communicate a consulting-style recommendation.

Universe of allowed underlyings:
- `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `META`, `GOOGL`

---

## What This Project Does
Given a selected stock and a specific live option contract (strike + expiry + call/put), the tool:

- **Fetches underlying data** (last 12 months) and computes **historical volatility**.
- **Pulls the live option chain** for a selected expiry (from `yfinance`).
- **Prices the option** using **Black–Scholes**.
- **Compares model price vs market price** (last traded price + bid/ask snapshot).
- **Calculates Greeks**:
  - **Delta (Δ)**: sensitivity to $1 move in underlying
  - **Theta (Θ)**: time decay per day
  - **Vega (ν)**: sensitivity to a 1% change in volatility
  - (also computes **Gamma** and **Rho** for completeness)
### New Features Added
- **Volatility Sensitivity**: Shows how the option price changes if volatility moves ±10%.
- **Time Decay Scenario**: Projects the option price after 7 days (if expiry allows).
- **Call vs Put Comparison**: Shows the price of the opposite option type at the same strike.
- **Model Limitations Section**: Explicitly lists assumptions and real-world gaps.

- **Identifies primary risk** and prints a **risk summary**.
- **Suggests a basic delta hedge** using underlying shares.
- **Generates 4 plots** (saved as `TICKER_option_analysis.png`).

---

## Methodology (High-Level)
- **Volatility estimate**: annualized standard deviation of daily returns over the most recent 12 months.
- **Option pricing**: Black–Scholes (European-style intuition).
- **Market vs model differences** can occur because:
  - implied volatility differs from historical volatility,
  - bid/ask spreads and liquidity effects,
  - supply/demand and positioning,
  - dividends, rates, early exercise / American feature (model mismatch),
  - event risk (earnings, macro),
  - stale last traded price.

---

## Outputs / Deliverables
- **Console analysis**
  - Black–Scholes fair value vs market price
  - Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Volatility shock scenarios (+/-10%)
  - 7-day time decay projection
  - Call vs put price comparison
  - Risk interpretation + delta hedging suggestion
  - Final recommendation summary
  - Model limitations disclaimer

- **Visualization image**
  - Saved as: `TICKER_option_analysis.png`

Example charts already generated in this repo:
- `AAPL_option_analysis.png`
- `MSFT_option_analysis.png`
- `NVDA_option_analysis.png`

---

## Repository Structure
- `option_analysis.py` — main script (interactive CLI)
- `AAPL_option_analysis.png` — sample output chart
- `MSFT_option_analysis.png` — sample output chart
- `NVDA_option_analysis.png` — sample output chart

---

## How To Run
### 1) Install dependencies
This script uses:
- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

Install with:
```bash
pip install yfinance numpy pandas matplotlib scipy
```

### 2) Run the analysis
```bash
python option_analysis.py
```

### 3) Follow the prompts
You will be prompted for:
- stock ticker (e.g., `AAPL`)
- expiration date (`YYYY-MM-DD`)
- strike price
- option type (`call` / `put`)

The tool prints the analysis and saves a plot as:
- `TICKER_option_analysis.png`

---

## Notes / Assumptions
- Black–Scholes is a simplified model; real listed equity options are typically American-style and may embed dividends.
- Volatility used here is **historical**, not implied.
- Option chain data is sourced via `yfinance` and depends on availability/latency.
- The tool now explicitly lists model limitations in the output.

---

## Acknowledgements
- Market data: `yfinance`
- Model: Black–Scholes (educational / transparent baseline)

---
