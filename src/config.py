"""
Central configuration for project 3: do options overprice macro news?

Every setting lives here: change it once and every script picks it up.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Folders (relative to this file, so the project works on any computer)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CBOE_DIR = RAW / "cboe"                      # daily volatility indices
FRED_DIR = RAW / "fred"                      # daily S&P 500 close and T-bill rate
OPTIONS_DIR = RAW / "options"
CANDLES_DIR = OPTIONS_DIR / "candles"        # 1-minute candles, one file per option contract
GREEKS_DIR = OPTIONS_DIR / "greeks"          # minute implied volatility and Greeks, release eves only
CHAIN_DIR = OPTIONS_DIR / "chain"            # snapshots of today's listed contracts
OPTIONS_COVERAGE = OPTIONS_DIR / "coverage.csv"
DATA_LSE = RAW / "lse"                       # 1-minute SPY bars, one parquet file per year
DATA_LSE_1S = RAW / "lse_1s"                 # (not used; kept so src/lse.py works unchanged)
LSE_CATALOG = RAW / "lse_catalog.csv"
ECON_CALENDAR = RAW / "economic_calendar_us.parquet"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

for _folder in (CBOE_DIR, FRED_DIR, CANDLES_DIR, GREEKS_DIR, CHAIN_DIR, DATA_LSE, FIGURES, RESULTS):
    _folder.mkdir(parents=True, exist_ok=True)

TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
START = "2022-05-01"            # VIX1D (the options market's expected move for the next day) starts 13 May 2022
EVENT_KINDS = ("CPI", "Payrolls", "FOMC")

# ---------------------------------------------------------------------------
# CBOE volatility indices (free daily CSV files, annualised %)
# ---------------------------------------------------------------------------
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
CBOE_INDICES = {
    "VIX1D": "S&P 500 expected volatility, next trading day",
    "VIX9D": "next 9 days",
    "VIX": "next 30 days",
    "VIX3M": "next 3 months",
}

# ---------------------------------------------------------------------------
# FRED (free daily series)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "SP500": "S&P 500 index close",
    "DTB3": "3-month Treasury bill (risk-free rate for option pricing)",
}

# ---------------------------------------------------------------------------
# London Strategic Edge: SPY 1-minute bars, the macro calendar, SPY options
# ---------------------------------------------------------------------------
LSE_START = START
LSE_SYMBOLS = {"SPY": "SPY"}    # our name -> LSE symbol
LSE_DATASETS = {"SPY": "us_stocks"}   # SPY is also listed under "nasdaq", a paid feed (HTTP 403): use us_stocks
SECONDS_START = "2026-07-01"    # (not used here; kept so src/lse.py works unchanged)

OPTIONS_UNDERLYING = "SPY"
OPTIONS_START = "2025-09-15"    # 1-minute option candles exist from about October 2025 (step 0c);
                                # starting a little earlier records exactly where they begin
STRIKE_OFFSETS = (-1, 0, 1)     # strikes around the eve's SPY close (SPY strikes are $1 apart)
CHAIN_MAX_DTE = 45              # today's chain snapshot: contracts expiring within 45 days

# SPY's quarterly dividends in the option sample: ex-dividend date -> $ per share.
# Source: dividendranks.com/etfs/SPY/dividend-history, cross-checked on chartrow.com (Nov 2026).
# Holders at the close BEFORE the ex-date get it; SPY's price drops by about this much on the ex-date.
SPY_DIVIDENDS = {
    "2025-09-19": 1.8311,
    "2025-12-19": 1.9934,
    "2026-03-20": 1.7970,
    "2026-06-18": 1.9040,
    "2026-09-18": 1.8888,
}
