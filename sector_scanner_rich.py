"""
Sector ETF · RSI / MACD Scanner  (optimised 3-phase)
======================================================
Scans all 11 SPDR sector ETFs and their top holdings for:
  BUY  setups: ETF trending up  + stock recently oversold  (<30 RSI in last 15d)
               + current RSI < 45 + daily RSI ↑ + daily MACD ↑ + weekly RSI ↑
  SELL setups: ETF trending down + stock recently overbought (>70 RSI in last 15d)
               + current RSI > 55 + daily RSI ↓ + daily MACD ↓ + weekly RSI ↓

3-phase pipeline (minimises expensive Twelve Data calls):
  Phase 1 — yfinance (parallel) : determine trend for all 11 sector ETFs
                                   (SMA9 derivative: up if last 2 daily Δ > 0, down if < 0)
  Phase 2 — yfinance (parallel) : screen ~165 stocks; keep only those with a recent
                                   RSI extreme that matches the ETF trend direction
  Phase 3 — Twelve Data (sequential, throttled) : precise RSI/MACD direction check
                                   on the small candidate shortlist only

Without --tdkey the script runs entirely on yfinance (phases 1+2 only for
the coarse screen, then yfinance again for phase 3 fine-check).

Requirements:
    pip install yfinance pandas requests rich

Get a FREE Twelve Data API key at: https://twelvedata.com  (800 req/day, no credit card)

Usage:
    python sector_scanner_rich.py                           # yfinance only
    python sector_scanner_rich.py --tdkey YOUR_KEY          # Twelve Data for fine-check
    python sector_scanner_rich.py --tdkey YOUR_KEY --workers 12 --out results.csv
"""

import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH = True
except ImportError:
    RICH = False

# ── Sector ETF universe ───────────────────────────────────────────────────────

SECTOR_ETFS = {
    # Full S&P 500 GICS sector membership (sourced from ETF fact sheets, Mar 2026)
    # ~500 holdings across all 11 sectors vs ~275 in the original list

    "XLE":  {"name": "Energy",            "holdings": [
                # Integrated / E&P
                "XOM","CVX","COP","EOG","OXY","DVN","FANG","CTRA","APA","CHRD",
                # Midstream / Pipelines
                "WMB","KMI","OKE","TRGP","HESM","LNG","DTM",
                # Refining
                "VLO","PSX","MPC","DINO",
                # Oilfield Services
                "SLB","HAL","BKR","NOV",
                # Other
                "EQT","RRC","TPL","EXE","HES"]},

    "XLF":  {"name": "Financials",        "holdings": [
                # Large-cap banks
                "JPM","BAC","WFC","C","USB","PNC","TFC","BK","STT","NTRS",
                # Regional banks
                "FITB","HBAN","MTB","CFG","RF","KEY","ZION","CMA","FCNCA","WAL",
                # Capital markets / asset managers
                "GS","MS","SCHW","BLK","BX","APO","KKR","ARES","AMG","IVZ",
                "TROW","BEN","AMP","RJF","LPLA","BR","SEIC","MKTX",
                # Diversified financials & payments
                "V","MA","AXP","COF","SYF","ALLY","BRK-B",
                # Exchanges & data
                "CME","ICE","CBOE","NDAQ","SPGI","MCO","FDS",
                # Insurance
                "PGR","CB","TRV","AFL","MET","PRU","ALL","HIG","AIG",
                "GL","CINF","UNM","WRB","ACGL","EG","MMC","AON",
                # Fintech (data processing — GICS IT but commonly screened here)
                "GPN","FIS","FI"]},

    "XLK":  {"name": "Technology",        "holdings": [
                # Semiconductors
                "NVDA","AVGO","AMD","MU","INTC","QCOM","TXN","ADI","AMAT","LRCX",
                "KLAC","ON","MCHP","NXPI","SWKS","MPWR","TER","SMCI",
                # Software — infrastructure & cloud
                "MSFT","ORCL","CRM","INTU","NOW","PLTR","APP","PANW","FTNT","CRWD",
                "ZS","SNPS","CDNS","ANSS","PTC","VRSN","GEN","AKAM",
                # IT services & consulting
                "ACN","IBM","CTSH","EPAM","IT","DXC","JKHY",
                # Hardware & storage
                "AAPL","DELL","HPQ","HPE","WDC","STX","NTAP","APH","TEL","KEYS",
                # Networking
                "CSCO","ANET","FFIV","MSI",
                # Payments (GICS Data Processing)
                "PYPL",
                # Other software
                "GDDY"]},

    "XLV":  {"name": "Health Care",       "holdings": [
                # Large-cap pharma
                "LLY","JNJ","ABBV","MRK","PFE","BMY","AMGN","GILD","VRTX","REGN",
                # Biotech
                "MRNA","BIIB","ALNY","NBIX","UTHR","INCY","EXAS","NTRA","VTRS",
                # Life sciences & diagnostics
                "TMO","DHR","A","RVTY","IQV","WAT","MTD","ILMN","BRKR","TECH",
                # Medical devices
                "ABT","ISRG","MDT","BSX","SYK","ZBH","BDX","DXCM","PODD","RMD",
                "HOLX","COO","TFX","HSIC","STE","GEHC","VEEV",
                # Managed care & insurance
                "UNH","CI","ELV","HUM","MOH","CNC",
                # Health systems & distribution
                "HCA","CVS","MCK","COR","ABC","CAH",
                # Smaller biotech
                "JAZZ"]},

    "XLI":  {"name": "Industrials",       "holdings": [
                # Aerospace & defence
                "GE","RTX","LMT","NOC","GD","BA","HWM","GEV","TDG","LHX",
                "TDY","HII","AXON","TXT",
                # Air freight & logistics
                "UPS","FDX","EXPD","JBHT","ODFL","CHRW",
                # Airlines
                "DAL","UAL","LUV","AAL","ALK",
                # Building products & HVAC
                "CARR","OTIS","JCI","AOS","MAS","ALLE",
                # Business services
                "CTAS","ADP","VRSK","CPRT","RSG","WM",
                # Electrical equipment & machinery
                "ETN","EMR","PH","AME","DOV","ITW","XYL","ROP","IR","IDEX",
                "GNRC","FTV",
                # Construction & engineering
                "CAT","DE","PWR","EME",
                # Railroads
                "UNP","NSC","CSX","WAB",
                # Road transport & ground
                "UBER",
                # Trading companies & distributors
                "GWW","FAST","SNA",
                # Professional & government services
                "HON","MMM","CMI","LDOS","SAIC","BAH",
                # Ground transport
                "CHRW"]},

    "XLB":  {"name": "Materials",         "holdings": [
                # Specialty & industrial gases
                "LIN","APD","ECL","IFF","ALB",
                # Mining & metals
                "NEM","FCX","NUE","STLD","CLF","AA","ATI","MP",
                # Construction materials
                "MLM","VMC","CRH","SHW","PPG","RPM",
                # Chemicals
                "DOW","LYB","CE","EMN","WLK","HUN","OLN","FMC","MOS","CF","CTVA",
                # Containers & packaging
                "PKG","IP","AVY","BALL","AMCR","OI","SEE","SON","GEF",
                # Other
                "SW"]},

    "XLY":  {"name": "Consumer Discret.", "holdings": [
                # E-commerce & broadline retail
                "AMZN","EBAY","ETSY","W",
                # Specialty retail
                "HD","LOW","TJX","ROST","AZO","ORLY","BBWI","ULTA","WSM","RH",
                "POOL","MHK","SCI",
                # Autos
                "TSLA","GM","F","CVNA","KMX","AN",
                # Restaurants & fast food
                "MCD","SBUX","CMG","YUM","DKNG",
                # Hotels, resorts & cruise lines
                "MAR","HLT","H","RCL","NCLH","CCL","MGM","LVS","WYNN","MTN",
                # Travel & booking
                "BKNG","EXPE","ABNB",
                # Homebuilders
                "DHI","LEN","PHM","NVR","TOL",
                # Apparel & luxury
                "NKE","RL","PVH","TPR","VFC","LULU",
                # Auto parts
                "GPC","DASH",
                # Leisure products
                "GRMN"]},

    "XLP":  {"name": "Consumer Staples",  "holdings": [
                # Hypermarkets & superstores
                "WMT","COST","TGT","KR","SFM","GO","CASY",
                # Food & beverages
                "PEP","KO","MNST","STZ","TAP","BF-B",
                "GIS","KHC","MDLZ","HSY","MKC","HRL","SJM","CPB","CAG","TSN",
                "LW","BG","POST","ADM",
                # Personal care & household
                "PG","CL","KMB","KVUE","CHD","EL",
                # Tobacco
                "PM","MO",
                # Food distribution
                "SYY","USFD",
                # Drug retail
                "DG","DLTR",
                # Other staples
                "KDP"]},

    "XLRE": {"name": "Real Estate",       "holdings": [
                # Industrial REITs
                "PLD","REXR","EGP",
                # Data centre REITs
                "EQIX","DLR",
                # Cell tower REITs
                "AMT","CCI","SBA",
                # Healthcare REITs
                "WELL","VTR",
                # Retail REITs
                "SPG","KIM","REG","FRT",
                # Residential REITs — apartments
                "AVB","EQR","MAA","ESS","UDR","CPT","AIRC",
                # Residential REITs — single family
                "INVH","AMH",
                # Office REITs
                "BXP","VNO","ARE",
                # Diversified / net-lease REITs
                "O","NNN","VICI","GLPI",
                # Self-storage REITs
                "PSA","EXR","CUBE",
                # Speciality REITs
                "IRM","COLD",
                # Hotels
                "HST","PK","RHP",
                # Timber REITs
                "WY","RYN","PCH",
                # RE services (non-REIT)
                "CBRE"]},

    "XLU":  {"name": "Utilities",         "holdings": [
                # Electric — large cap
                "NEE","SO","DUK","AEP","EXC","SRE","D","XEL","PEG","ETR",
                "ED","WEC","DTE","AEE","EIX","PPL","FE","ES","CNP",
                # Nuclear / merchant power
                "CEG","VST","NRG",
                # Water
                "AWK",
                # Multi-state electric
                "LNT","PNW","EVRG","NI","OGE",
                # Natural gas distribution
                "ATO","NFG","UGI","SWX","NI",
                # Other
                "PCG","IDA"]},

    "XLC":  {"name": "Comm. Services",    "holdings": [
                # Interactive media & social
                "META","GOOGL","GOOG","SNAP","PINS","MTCH","IAC",
                # Entertainment & streaming
                "NFLX","DIS","WBD","PARA","SIRI","LYV","TKO",
                # Video games
                "EA","TTWO",
                # Telecom
                "VZ","T","TMUS","CHTR",
                # Media & publishing
                "CMCSA","FOXA","FOX","NWSA","NWS","NYT","IPG","OMC",
                # Digital advertising & ad tech
                "TTD",
                # Streaming / other
                "ROKU"]},
}

# ── Twelve Data throttle (free tier: 8 req/min) ───────────────────────────────

TD_API_KEY    = None
TD_MIN_DELAY  = 7.6          # seconds between calls
_last_td_call = 0.0

def _td_throttle():
    global _last_td_call
    wait = _last_td_call + TD_MIN_DELAY - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_td_call = time.monotonic()

# ── Technical indicators ──────────────────────────────────────────────────────
#
#  Two implementations are provided:
#    calc_rsi_fast / calc_macd_fast  — pandas ewm-based; fast, used for coarse
#                                      yfinance screening (phase 2)
#    calc_rsi / calc_macd            — exact Pine Script / TradingView match;
#                                      used for the Twelve Data fine-check (phase 3)

# -- Fast variants (yfinance coarse screen) -----------------------------------

def calc_rsi_fast(closes: pd.Series, period: int = 14) -> pd.Series:
    delta    = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd_fast(closes: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow   = closes.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# -- Accurate variants (TradingView / Pine Script exact match) ----------------

def _rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA — identical to Pine Script ta.rma(). Seed = SMA of first period values."""
    result  = np.full(len(values), np.nan)
    alpha   = 1.0 / period
    buf     = []
    rma_val = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if len(buf) < period:
            buf.append(v)
            if len(buf) == period:
                rma_val   = float(np.mean(buf))
                result[i] = rma_val
        else:
            rma_val   = rma_val * (1 - alpha) + v * alpha
            result[i] = rma_val
    return result


def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI matching Pine Script ta.rsi() exactly."""
    c    = closes.to_numpy(dtype=float)
    diff = np.diff(c)
    u    = np.where(diff > 0,  diff, 0.0)
    d    = np.where(diff < 0, -diff, 0.0)
    avg_u = np.concatenate([[np.nan], _rma(u, period)])
    avg_d = np.concatenate([[np.nan], _rma(d, period)])
    with np.errstate(invalid="ignore", divide="ignore"):
        rs  = avg_u / avg_d
        rsi = np.where(avg_d == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    return pd.Series(rsi, index=closes.index)


def _ema_tv(values: np.ndarray, period: int) -> np.ndarray:
    """EMA matching Pine Script ta.ema(). Seed = first non-NaN value."""
    result  = np.full(len(values), np.nan)
    alpha   = 2.0 / (period + 1)
    ema_val = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        ema_val   = v if np.isnan(ema_val) else ema_val + alpha * (v - ema_val)
        result[i] = ema_val
    return result


def calc_macd(closes: pd.Series, fast=12, slow=26, signal=9):
    """MACD matching Pine Script ta.macd() exactly."""
    c    = closes.to_numpy(dtype=float)
    macd = _ema_tv(c, fast) - _ema_tv(c, slow)
    sig  = _ema_tv(macd, signal)
    return pd.Series(macd, index=closes.index), pd.Series(sig, index=closes.index)


def to_weekly(daily: pd.Series) -> pd.Series:
    return daily.resample("W").last().dropna()


def determine_trend(closes: pd.Series) -> str:
    """SMA9 derivative method:
    Compute daily SMA9; if both the last 2 daily changes are positive -> up,
    both negative -> down, otherwise -> unknown.
    """
    if len(closes) < 11:
        return "unknown"
    sma9 = closes.rolling(9).mean().dropna()
    if len(sma9) < 3:
        return "unknown"
    d1 = float(sma9.iloc[-1] - sma9.iloc[-2])  # today's derivative
    d2 = float(sma9.iloc[-2] - sma9.iloc[-3])  # yesterday's derivative
    if d1 > 0 and d2 > 0:
        return "up"
    if d1 < 0 and d2 < 0:
        return "down"
    return "unknown"

# ── Data fetching ─────────────────────────────────────────────────────────────

def _to_series(raw) -> "pd.Series | None":
    """Flatten yfinance's sometimes-MultiIndex output into a plain 1-D Series."""
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    raw = raw.squeeze().dropna()
    if not isinstance(raw, pd.Series) or len(raw) < 40:
        return None
    # Drop rows where the adjusted close implies a single-day move > 20%.
    # yfinance auto_adjust=True occasionally produces corrupt dividend-adjustment
    # artifacts that look like extreme 1-day spikes — these send RSI to 0/100.
    pct = raw.pct_change().abs()
    bad = pct > 0.20
    if bad.any():
        raw = raw[~bad]
    return raw if len(raw) >= 40 else None


def fetch_yfinance(symbol: str, period: str = "9mo") -> "pd.Series | None":
    try:
        df = yf.download(symbol, period=period, interval="1d",
                         progress=False, auto_adjust=True, threads=False)
        return _to_series(df["Close"])
    except Exception:
        return None


def fetch_twelvedata(symbol: str, outputsize: int = 200) -> "pd.Series | None":
    if not TD_API_KEY:
        return None
    _td_throttle()
    try:
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params=dict(symbol=symbol, interval="1day", outputsize=outputsize,
                        apikey=TD_API_KEY, format="JSON", order="ASC"),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            return None
        records = data["values"]
        idx  = pd.to_datetime([r["datetime"] for r in records])
        vals = np.array([float(r["close"]) for r in records])
        s    = pd.Series(vals, index=idx).sort_index().dropna()
        return s if len(s) >= 40 else None
    except Exception:
        return None

# ── Phase 2: coarse RSI screen (yfinance) ─────────────────────────────────────

def coarse_screen(symbol: str, etf_key: str, etf_trend: str) -> "dict | None":
    """
    Fast yfinance check: did this stock recently hit the RSI extreme
    that matches the ETF trend?
      ETF up   → was stock oversold   (RSI < 30) in the last 15 days?
      ETF down → was stock overbought (RSI > 70) in the last 15 days?
    Returns a small dict with the symbol metadata if it passes, else None.
    """
    closes = fetch_yfinance(symbol)
    if closes is None:
        return None

    rsi_vals  = calc_rsi(closes)      # Wilder's RMA — matches Phase 3 / TradingView
    last15    = rsi_vals.iloc[-15:].dropna()
    if len(last15) < 5:
        return None

    if etf_trend == "up"   and bool((last15 < 30).any()):
        return dict(symbol=symbol, etf_key=etf_key, etf_trend=etf_trend,
                    extreme="oversold",   extreme_val=round(float(last15.min()), 1))
    if etf_trend == "down" and bool((last15 > 70).any()):
        return dict(symbol=symbol, etf_key=etf_key, etf_trend=etf_trend,
                    extreme="overbought", extreme_val=round(float(last15.max()), 1))
    return None

# ── Phase 3: fine-check with accurate data (Twelve Data or yfinance fallback) ─

def fine_check(candidate: dict, debug: bool = False, relax: bool = False) -> "dict | None":
    """
    Re-fetch the candidate with Twelve Data (or yfinance if no key) and
    verify all remaining conditions with TradingView-accurate indicators.

    debug=True  — print a per-condition breakdown regardless of outcome.
    relax=True  — require 2/3 momentum indicators instead of all 3.
    """
    symbol    = candidate["symbol"]
    etf_key   = candidate["etf_key"]
    etf_trend = candidate["etf_trend"]

    # Use Twelve Data if key provided, else fall back to yfinance
    closes = fetch_twelvedata(symbol) if TD_API_KEY else fetch_yfinance(symbol)
    if closes is None or len(closes) < 40:
        if debug:
            print(f"    {symbol:6s}  ✗ insufficient price data")
        return None

    # Daily RSI (accurate)
    daily_rsi  = calc_rsi(closes)
    last15_rsi = daily_rsi.iloc[-15:].dropna()
    if len(last15_rsi) < 5:
        if debug:
            print(f"    {symbol:6s}  ✗ insufficient RSI history")
        return None

    cur_rsi  = float(daily_rsi.iloc[-1])
    prev_rsi = float(daily_rsi.iloc[-2])
    rsi_up   = cur_rsi > prev_rsi
    rsi_down = cur_rsi < prev_rsi

    # Daily MACD (accurate)
    macd_line, _ = calc_macd(closes)
    cur_macd  = float(macd_line.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    macd_up   = cur_macd > prev_macd
    macd_down = cur_macd < prev_macd

    # Weekly RSI (accurate)
    weekly_rsi = calc_rsi(to_weekly(closes))
    if len(weekly_rsi.dropna()) < 5:
        if debug:
            print(f"    {symbol:6s}  ✗ insufficient weekly RSI history")
        return None
    cur_wrsi  = float(weekly_rsi.iloc[-1])
    prev_wrsi = float(weekly_rsi.iloc[-2])
    wrsi_up   = cur_wrsi > prev_wrsi
    wrsi_down = cur_wrsi < prev_wrsi

    # ── Condition evaluation ──────────────────────────────────────────────
    if etf_trend == "up":
        thresh_ok  = cur_rsi < 45
        conditions = [rsi_up, macd_up, wrsi_up]
        labels     = ["D-RSI↑", "MACD↑", "W-RSI↑"]
        needed_dir = "BUY"
    else:
        thresh_ok  = cur_rsi > 55
        conditions = [rsi_down, macd_down, wrsi_down]
        labels     = ["D-RSI↓", "MACD↓", "W-RSI↓"]
        needed_dir = "SELL"

    passing = sum(conditions)
    required = 2 if relax else 3

    if debug:
        thresh_label = f"RSI {'<45' if etf_trend == 'up' else '>55'}"
        values = [
            f"{prev_rsi:.1f}→{cur_rsi:.1f}",
            f"{prev_macd:.3f}→{cur_macd:.3f}",
            f"{prev_wrsi:.1f}→{cur_wrsi:.1f}",
        ]
        cond_str = "  ".join(
            f"{'✓' if ok else '✗'} {lbl} [{v}]"
            for ok, lbl, v in zip(conditions, labels, values)
        )
        mode_str = "[relax 2/3]" if relax else "[strict 3/3]"
        print(
            f"    {symbol:6s}  {mode_str}  "
            f"{'✓' if thresh_ok else '✗'} {thresh_label} ({cur_rsi:.1f})  "
            f"{cond_str}  "
            f"→ {passing}/{len(conditions)} pass  "
            f"{'✅ SIGNAL' if thresh_ok and passing >= required else '✗ filtered'}"
        )

    signal = None
    if thresh_ok and passing >= required:
        signal = needed_dir

    if signal is None:
        return None

    return {
        "Signal":          signal,
        "Symbol":          symbol,
        "ETF":             etf_key,
        "Sector":          SECTOR_ETFS[etf_key]["name"],
        "Price":           round(float(closes.iloc[-1]), 2),
        "D-RSI":           round(cur_rsi, 1),
        "RSI Dir":         "↑" if rsi_up   else "↓",
        "MACD Dir":        "↑" if macd_up  else "↓",
        "W-RSI":           round(cur_wrsi, 1),
        "W-RSI Dir":       "↑" if wrsi_up  else "↓",
        "Min/Max RSI 15d": candidate["extreme_val"],
        "RSI Thresh":      "<45 ✓" if signal == "BUY" else ">55 ✓",
    }

# ── Output helpers ────────────────────────────────────────────────────────────

def print_plain(label: str, rows: list):
    if not rows:
        print(f"\n  No {label} signals found.\n"); return
    cols  = list(rows[0].keys())
    col_w = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print(f"\n{'─'*80}\n  {label}  ({len(rows)} signals)\n{'─'*80}")
    print("  " + "  ".join(c.ljust(col_w[c]) for c in cols))
    print("  " + "  ".join("-" * col_w[c]    for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r[c]).ljust(col_w[c]) for c in cols))
    print()


def print_rich_table(label: str, rows: list, color: str):
    console = Console()
    if not rows:
        console.print(f"\n  [dim]No {label} signals found.[/dim]\n"); return
    t = Table(title=f"{label}  ({len(rows)})", box=box.SIMPLE_HEAVY, border_style="dim")
    cols = list(rows[0].keys())
    for c in cols:
        t.add_column(c, style=f"bold {color}" if c == "Signal" else "default")
    for r in rows:
        t.add_row(*[str(r[c]) for c in cols])
    console.print(t)

# ── Progress helper ───────────────────────────────────────────────────────────

def _run_parallel(jobs, fn, workers, prog_label, console):
    """Run fn(job) for each job in parallel; return (results, failed) lists."""
    results, failed = [], []
    total = len(jobs)
    done  = 0

    if RICH:
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(),
                      console=console) as prog:
            task = prog.add_task(prog_label, total=total)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(fn, j): j for j in jobs}
                for fut in as_completed(futures):
                    sym = futures[fut][0] if isinstance(futures[fut], tuple) else futures[fut].get("symbol", "")
                    prog.advance(task)
                    prog.update(task, description=f"[cyan]{sym:6s}[/cyan]")
                    try:
                        r = fut.result()
                        if r: results.append(r)
                    except Exception:
                        failed.append(sym)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, j): j for j in jobs}
            for fut in as_completed(futures):
                sym = futures[fut][0] if isinstance(futures[fut], tuple) else futures[fut].get("symbol", "")
                done += 1
                bar  = "█" * int(done/total*40) + "░" * (40 - int(done/total*40))
                print(f"\r  [{bar}] {done}/{total}  {sym:6s}", end="", flush=True)
                try:
                    r = fut.result()
                    if r: results.append(r)
                except Exception:
                    failed.append(sym)
        print()

    return results, failed

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TD_API_KEY

    parser = argparse.ArgumentParser(description="Sector ETF RSI/MACD Scanner")
    parser.add_argument("--tdkey",   type=str, default=None,
                        help="Twelve Data API key (free at twelvedata.com). "
                             "Used only for the final fine-check on candidates.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for yfinance phases (default 10).")
    parser.add_argument("--out",     type=str, default=None,
                        help="Optional CSV output path, e.g. results.csv")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-condition breakdown for every Phase 3 candidate.")
    parser.add_argument("--relax",   action="store_true",
                        help="Require 2/3 momentum indicators instead of all 3 (catches "
                             "more signals in choppy markets).")
    args = parser.parse_args()

    if args.tdkey:
        TD_API_KEY = args.tdkey

    console = Console() if RICH else None
    started = datetime.now()
    td_label = "Twelve Data" if TD_API_KEY else "yfinance (no --tdkey)"

    print("\n📡  Sector ETF · RSI / MACD Scanner  (3-phase optimised)")
    print(f"    {started.strftime('%Y-%m-%d %H:%M')}")
    print(f"    Phase 1+2: yfinance  (parallel, workers={args.workers})")
    print(f"    Phase 3:   {td_label}  (accurate indicators, sequential)\n")

    # ── Phase 1: ETF trends via yfinance ──────────────────────────────────
    print("Phase 1 — Sector ETF trends (yfinance) …\n")

    etf_trends = {}
    for etf in SECTOR_ETFS:
        closes = fetch_yfinance(etf)
        if closes is None:
            trend, price = "unknown", "–"
        else:
            trend = determine_trend(closes)
            price = f"${float(closes.iloc[-1]):.2f}"
        etf_trends[etf] = trend
        arrow = "📈" if trend == "up" else "📉" if trend == "down" else "➖"
        print(f"  {etf:5s}  {arrow}  {trend.upper():8s}  {price:>8s}   {SECTOR_ETFS[etf]['name']}")

    # ── Phase 2: Coarse RSI screen via yfinance (parallel) ────────────────
    print("\nPhase 2 — Coarse RSI screen (yfinance, parallel) …\n")

    screen_jobs = [
        (sym, etf, etf_trends[etf])
        for etf, data in SECTOR_ETFS.items()
        for sym in data["holdings"]
        if etf_trends.get(etf) in ("up", "down")
    ]

    def _coarse(job):
        return coarse_screen(*job)

    candidates, screen_failed = _run_parallel(
        screen_jobs, _coarse, args.workers, "Screening…", console)

    print(f"\n  → {len(candidates)} candidate(s) passed coarse RSI screen "
          f"(from {len(screen_jobs)} stocks)\n")
    if candidates:
        for c in candidates:
            direction = "oversold"   if c["etf_trend"] == "up" else "overbought"
            print(f"     {c['symbol']:6s}  [{c['etf_key']}]  {direction}  "
                  f"RSI extreme={c['extreme_val']}")
    print()

    if not candidates:
        print("  No candidates — nothing to fine-check. Exiting.\n")
        return

    # ── Phase 3: Fine-check candidates via Twelve Data (sequential) ───────
    src = "Twelve Data" if TD_API_KEY else "yfinance"
    mode_note = ""
    if args.relax:
        mode_note += "  [--relax: 2/3 indicators required]"
    if args.debug:
        mode_note += "  [--debug: showing condition breakdown]"
    print(f"Phase 3 — Fine-check {len(candidates)} candidate(s) via {src} …{mode_note}\n")

    results, fine_failed = [], []
    total = len(candidates)

    for i, cand in enumerate(candidates, 1):
        sym = cand["symbol"]
        if not args.debug:
            bar = "█" * int(i/total*30) + "░" * (30 - int(i/total*30))
            print(f"\r  [{bar}] {i}/{total}  {sym:6s}", end="", flush=True)
        else:
            print(f"  [{i}/{total}]  {sym}", flush=True)
        try:
            r = fine_check(cand, debug=args.debug, relax=args.relax)
            if r:
                results.append(r)
        except Exception as e:
            fine_failed.append(sym)
            if args.debug:
                print(f"    {sym:6s}  ✗ exception: {e}")

    print()

    # ── Results ────────────────────────────────────────────────────────────
    buys  = [r for r in results if r["Signal"] == "BUY"]
    sells = [r for r in results if r["Signal"] == "SELL"]
    all_failed = screen_failed + fine_failed

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'═'*80}")
    print(f"  Scan complete in {elapsed}s  |  "
          f"Screened: {len(screen_jobs)}  Candidates: {len(candidates)}  "
          f"BUY: {len(buys)}  SELL: {len(sells)}  Failed: {len(all_failed)}")
    print(f"{'═'*80}")

    if RICH:
        print_rich_table("🟢  BUY  Setups", buys,  "green")
        print_rich_table("🔴  SELL Setups", sells, "red")
    else:
        print_plain("BUY  Setups", buys)
        print_plain("SELL Setups", sells)

    if all_failed:
        print(f"  ⚠️  Could not fetch ({len(all_failed)}): {', '.join(all_failed)}\n")

    if args.out and results:
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"  💾  Results saved to {args.out}\n")
    elif args.out:
        print("  ℹ️   No signals to save.\n")


if __name__ == "__main__":
    main()
