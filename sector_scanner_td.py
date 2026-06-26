"""
Sector ETF · RSI / MACD Scanner  (Twelve Data only)
====================================================
Same scan logic as sector_scanner_rich.py but uses Twelve Data for
ALL three phases — no yfinance — so RSI/MACD values are fully consistent
across the coarse screen (Phase 2) and fine-check (Phase 3).

Key design choices vs the original:
  • Single data source  : Twelve Data throughout → no cross-source divergence
  • In-memory cache     : each symbol is fetched exactly once; Phase 3 reuses
                          Phase 2 data at zero extra API cost
  • Wilder RSI only     : calc_rsi (Pine Script-accurate) used everywhere
  • Sequential fetching : free-tier rate limit (8 req/min) enforced globally
  • Time estimate       : progress bar shows requests remaining and ETA

Rate limits (Twelve Data free tier):
  800 credits/day · 8 req/min
  Full scan (11 ETFs + ~275 stocks) = ~286 requests ≈ 36 minutes

Get a FREE Twelve Data API key at: https://twelvedata.com  (no credit card)

Usage:
    python sector_scanner_td.py --tdkey YOUR_KEY
    python sector_scanner_td.py --tdkey YOUR_KEY --out results.csv
    python sector_scanner_td.py --tdkey YOUR_KEY --debug
    python sector_scanner_td.py --tdkey YOUR_KEY --relax
    python sector_scanner_td.py --tdkey YOUR_KEY --debug --relax --out results.csv

Requirements:
    pip install pandas numpy requests rich
"""

import argparse
import time
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
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

# ── Twelve Data client ────────────────────────────────────────────────────────

TD_MIN_DELAY  = 7.6          # seconds between calls (free tier: 8 req/min)
_last_td_call = 0.0
_fetch_cache: dict = {}      # symbol → pd.Series; avoids double-fetching
_req_count    = 0


def _throttle():
    global _last_td_call
    wait = _last_td_call + TD_MIN_DELAY - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_td_call = time.monotonic()


def fetch(symbol: str, api_key: str, outputsize: int = 200) -> "pd.Series | None":
    """Fetch daily closes from Twelve Data, with in-memory caching."""
    global _req_count
    if symbol in _fetch_cache:
        return _fetch_cache[symbol]

    _throttle()
    _req_count += 1
    try:
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params=dict(symbol=symbol, interval="1day", outputsize=outputsize,
                        apikey=api_key, format="JSON", order="ASC"),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            _fetch_cache[symbol] = None
            return None
        records = data["values"]
        idx  = pd.to_datetime([r["datetime"] for r in records])
        vals = np.array([float(r["close"]) for r in records])
        s    = pd.Series(vals, index=idx).sort_index().dropna()
        result = s if len(s) >= 40 else None
    except Exception:
        result = None

    _fetch_cache[symbol] = result
    return result

# ── Technical indicators (Wilder / TradingView-accurate) ─────────────────────

def _rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA — identical to Pine Script ta.rma()."""
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
    """EMA matching Pine Script ta.ema()."""
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
    """SMA9 derivative: up if last 2 daily Δ > 0, down if < 0, else unknown."""
    if len(closes) < 11:
        return "unknown"
    sma9 = closes.rolling(9).mean().dropna()
    if len(sma9) < 3:
        return "unknown"
    d1 = float(sma9.iloc[-1] - sma9.iloc[-2])
    d2 = float(sma9.iloc[-2] - sma9.iloc[-3])
    if d1 > 0 and d2 > 0:
        return "up"
    if d1 < 0 and d2 < 0:
        return "down"
    return "unknown"

# ── Phase 2: coarse RSI screen ────────────────────────────────────────────────

def coarse_screen(symbol: str, etf_key: str, etf_trend: str,
                  api_key: str) -> "dict | None":
    """
    Fetch closes via Twelve Data and check whether this stock recently hit
    the RSI extreme matching the ETF trend direction.
    Data is cached — Phase 3 reuses it at no extra API cost.
    """
    closes = fetch(symbol, api_key)
    if closes is None:
        return None

    rsi_vals = calc_rsi(closes)
    last15   = rsi_vals.iloc[-15:].dropna()
    if len(last15) < 5:
        return None

    if etf_trend == "up"   and bool((last15 < 30).any()):
        return dict(symbol=symbol, etf_key=etf_key, etf_trend=etf_trend,
                    extreme="oversold",   extreme_val=round(float(last15.min()), 1))
    if etf_trend == "down" and bool((last15 > 70).any()):
        return dict(symbol=symbol, etf_key=etf_key, etf_trend=etf_trend,
                    extreme="overbought", extreme_val=round(float(last15.max()), 1))
    return None

# ── Phase 3: fine-check (reuses cached data — no new API calls) ───────────────

def fine_check(candidate: dict, debug: bool = False,
               relax: bool = False) -> "dict | None":
    """
    Verify all momentum conditions using the closes already in cache.
    No Twelve Data call is made here.

    debug=True  — print per-condition breakdown.
    relax=True  — require 2/3 momentum indicators instead of all 3.
    """
    symbol    = candidate["symbol"]
    etf_key   = candidate["etf_key"]
    etf_trend = candidate["etf_trend"]

    closes = _fetch_cache.get(symbol)
    if closes is None or len(closes) < 40:
        if debug:
            print(f"    {symbol:6s}  ✗ no cached data")
        return None

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

    macd_line, _ = calc_macd(closes)
    cur_macd  = float(macd_line.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    macd_up   = cur_macd > prev_macd
    macd_down = cur_macd < prev_macd

    weekly_rsi = calc_rsi(to_weekly(closes))
    if len(weekly_rsi.dropna()) < 5:
        if debug:
            print(f"    {symbol:6s}  ✗ insufficient weekly RSI history")
        return None
    cur_wrsi  = float(weekly_rsi.iloc[-1])
    prev_wrsi = float(weekly_rsi.iloc[-2])
    wrsi_up   = cur_wrsi > prev_wrsi
    wrsi_down = cur_wrsi < prev_wrsi

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

    passing  = sum(conditions)
    required = 2 if relax else 3

    if debug:
        thresh_label = f"RSI {'<45' if etf_trend == 'up' else '>55'}"
        cond_str = "  ".join(
            f"{'✓' if ok else '✗'} {lbl}" for ok, lbl in zip(conditions, labels)
        )
        mode_str = "[relax 2/3]" if relax else "[strict 3/3]"
        print(
            f"    {symbol:6s}  {mode_str}  "
            f"{'✓' if thresh_ok else '✗'} {thresh_label} ({cur_rsi:.1f})  "
            f"{cond_str}  "
            f"→ {passing}/{len(conditions)} pass  "
            f"{'✅ SIGNAL' if thresh_ok and passing >= required else '✗ filtered'}"
        )

    if not (thresh_ok and passing >= required):
        return None

    signal = needed_dir
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

# ── Progress line ─────────────────────────────────────────────────────────────

def _progress(done: int, total: int, symbol: str, started: float):
    elapsed = time.monotonic() - started
    eta_s   = int((elapsed / done) * (total - done)) if done else 0
    eta     = str(timedelta(seconds=eta_s))
    bar     = "█" * int(done / total * 35) + "░" * (35 - int(done / total * 35))
    print(f"\r  [{bar}] {done}/{total}  {symbol:6s}  ETA {eta}  ", end="", flush=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sector ETF RSI/MACD Scanner — Twelve Data only")
    parser.add_argument("--tdkey",   required=True,
                        help="Twelve Data API key (free at twelvedata.com).")
    parser.add_argument("--out",     default=None,
                        help="Optional CSV output path, e.g. results.csv")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-condition breakdown for every Phase 3 candidate.")
    parser.add_argument("--relax",   action="store_true",
                        help="Require 2/3 momentum indicators instead of all 3.")
    parser.add_argument("--outputsize", type=int, default=200,
                        help="Bars to fetch per symbol (default 200 ≈ 9 months).")
    args = parser.parse_args()

    # Pre-flight estimate
    all_symbols = [etf for etf in SECTOR_ETFS] + [
        sym for data in SECTOR_ETFS.values() for sym in data["holdings"]
    ]
    total_requests = len(all_symbols)
    est_minutes    = int(total_requests * TD_MIN_DELAY / 60) + 1

    started_wall = datetime.now()
    started_mono = time.monotonic()

    print("\n📡  Sector ETF · RSI / MACD Scanner  (Twelve Data only)")
    print(f"    {started_wall.strftime('%Y-%m-%d %H:%M')}")
    print(f"    Data source : Twelve Data  (all phases)")
    print(f"    Indicators  : Wilder RSI + Pine Script MACD  (TradingView-accurate)")
    print(f"    Symbols     : {len(SECTOR_ETFS)} ETFs + ~{total_requests - len(SECTOR_ETFS)} holdings")
    print(f"    API calls   : ~{total_requests} total  (Phase 3 reuses cache → 0 extra calls)")
    print(f"    Est. time   : ~{est_minutes} min  (free-tier rate limit: 8 req/min)\n")

    # ── Phase 1: ETF trends ───────────────────────────────────────────────
    print("Phase 1 — Sector ETF trends (Twelve Data) …\n")

    etf_trends = {}
    for i, etf in enumerate(SECTOR_ETFS, 1):
        _progress(i, len(SECTOR_ETFS), etf, started_mono)
        closes = fetch(etf, args.tdkey, args.outputsize)
        if closes is None:
            trend, price = "unknown", "–"
        else:
            trend = determine_trend(closes)
            price = f"${float(closes.iloc[-1]):.2f}"
        etf_trends[etf] = trend

    print()  # newline after progress bar
    for etf, trend in etf_trends.items():
        closes = _fetch_cache.get(etf)
        price  = f"${float(closes.iloc[-1]):.2f}" if closes is not None else "–"
        arrow  = "📈" if trend == "up" else "📉" if trend == "down" else "➖"
        print(f"  {etf:5s}  {arrow}  {trend.upper():8s}  {price:>8s}   {SECTOR_ETFS[etf]['name']}")

    # ── Phase 2: Coarse RSI screen (Twelve Data, sequential) ─────────────
    print("\nPhase 2 — Coarse RSI screen (Twelve Data, sequential) …\n")

    screen_jobs = [
        (sym, etf, etf_trends[etf])
        for etf, data in SECTOR_ETFS.items()
        for sym in data["holdings"]
        if etf_trends.get(etf) in ("up", "down")
    ]

    candidates, failed = [], []
    total_p2 = len(screen_jobs)
    p2_start = time.monotonic()

    for i, (sym, etf, trend) in enumerate(screen_jobs, 1):
        _progress(i, total_p2, sym, p2_start)
        try:
            result = coarse_screen(sym, etf, trend, args.tdkey)
            if result:
                candidates.append(result)
        except Exception:
            failed.append(sym)

    print()
    print(f"\n  → {len(candidates)} candidate(s) passed coarse RSI screen "
          f"(from {total_p2} stocks)\n")
    if candidates:
        for c in candidates:
            direction = "oversold" if c["etf_trend"] == "up" else "overbought"
            print(f"     {c['symbol']:6s}  [{c['etf_key']}]  {direction}  "
                  f"RSI extreme={c['extreme_val']}")
    print()

    if not candidates:
        print("  No candidates — nothing to fine-check. Exiting.\n")
        return

    # ── Phase 3: Fine-check (no API calls — uses cache) ───────────────────
    mode_note = ""
    if args.relax:
        mode_note += "  [--relax: 2/3 indicators required]"
    if args.debug:
        mode_note += "  [--debug: showing condition breakdown]"

    print(f"Phase 3 — Fine-check {len(candidates)} candidate(s)  "
          f"[cache hit — 0 new API calls]{mode_note}\n")

    results = []
    for i, cand in enumerate(candidates, 1):
        sym = cand["symbol"]
        if args.debug:
            print(f"  [{i}/{len(candidates)}]  {sym}")
        try:
            r = fine_check(cand, debug=args.debug, relax=args.relax)
            if r:
                results.append(r)
        except Exception as e:
            failed.append(sym)
            if args.debug:
                print(f"    {sym:6s}  ✗ exception: {e}")

    # ── Results ────────────────────────────────────────────────────────────
    buys  = [r for r in results if r["Signal"] == "BUY"]
    sells = [r for r in results if r["Signal"] == "SELL"]

    elapsed = int(time.monotonic() - started_mono)
    print(f"\n{'═'*80}")
    print(f"  Scan complete in {elapsed}s  |  "
          f"API calls: {_req_count}  Candidates: {len(candidates)}  "
          f"BUY: {len(buys)}  SELL: {len(sells)}  Failed: {len(failed)}")
    print(f"{'═'*80}")

    if RICH:
        print_rich_table("🟢  BUY  Setups", buys,  "green")
        print_rich_table("🔴  SELL Setups", sells, "red")
    else:
        print_plain("BUY  Setups", buys)
        print_plain("SELL Setups", sells)

    if failed:
        print(f"  ⚠️  Could not fetch ({len(failed)}): {', '.join(failed)}\n")

    if args.out and results:
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"  💾  Results saved to {args.out}\n")
    elif args.out:
        print("  ℹ️   No signals to save.\n")


if __name__ == "__main__":
    main()
