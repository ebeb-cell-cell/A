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
    # Holdings sourced from stockanalysis.com (top 25 per ETF, as of Mar 2026)
    "XLE":  {"name": "Energy",            "holdings": [
                "XOM","CVX","COP","WMB","SLB","EOG","KMI","VLO","PSX","MPC",
                "BKR","OKE","TRGP","OXY","EQT","FANG","TPL","HAL","DVN","EXE",
                "CTRA","APA","HESM"]},
    "XLF":  {"name": "Financials",        "holdings": [
                "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","C","AXP",
                "SCHW","BLK","SPGI","COF","PGR","CB","CME","ICE","MMC","BX",
                "USB","PNC","BK","AON","MCO"]},
    "XLK":  {"name": "Technology",        "holdings": [
                "NVDA","AAPL","MSFT","AVGO","MU","PLTR","AMD","CSCO","AMAT","LRCX",
                "ORCL","IBM","INTC","KLAC","TXN","CRM","ADI","APH","QCOM","ANET",
                "APP","ACN","PANW","INTU","NOW"]},
    "XLV":  {"name": "Health Care",       "holdings": [
                "LLY","JNJ","ABBV","MRK","UNH","AMGN","TMO","ABT","GILD","ISRG",
                "PFE","SYK","DHR","BMY","MDT","VRTX","MCK","BSX","CVS","HCA",
                "REGN","CI","COR","ELV","ZTS"]},
    "XLI":  {"name": "Industrials",       "holdings": [
                "GE","CAT","RTX","GEV","BA","UBER","UNP","HON","DE","ETN",
                "LMT","PH","HWM","NOC","TT","GD","WM","ADP","JCI","MMM",
                "PWR","FDX","EMR","UPS","CMI"]},
    "XLB":  {"name": "Materials",         "holdings": [
                "LIN","NEM","FCX","SHW","CTVA","APD","ECL","CRH","NUE","MLM",
                "VMC","STLD","DOW","PPG","SW","IP","ALB","MOS","CF","IFF",
                "PKG","AVY","EMN","CE","RPM"]},
    "XLY":  {"name": "Consumer Discret.", "holdings": [
                "AMZN","TSLA","HD","MCD","TJX","LOW","BKNG","SBUX","ORLY","MAR",
                "GM","RCL","HLT","NKE","ROST","DASH","AZO","ABNB","F","CMG",
                "CVNA","YUM","DHI","EBAY","GRMN"]},
    "XLP":  {"name": "Consumer Staples",  "holdings": [
                "WMT","COST","PG","KO","PM","PEP","CL","MDLZ","MO","MNST",
                "TGT","SYY","KR","KDP","KMB","KVUE","HSY","ADM","DG","EL",
                "GIS","DLTR","CHD","STZ","KHC"]},
    "XLRE": {"name": "Real Estate",       "holdings": [
                "WELL","PLD","EQIX","AMT","O","PSA","DLR","SPG","VTR","CCI",
                "CBRE","IRM","VICI","EXR","AVB","EQR","ARE","HST","KIM","MAA",
                "NNN","SBA","ESS","UDR","WY"]},
    "XLU":  {"name": "Utilities",         "holdings": [
                "NEE","SO","DUK","CEG","AEP","SRE","D","VST","EXC","XEL",
                "ETR","PEG","PCG","ED","WEC","NRG","DTE","AEE","ATO","EIX",
                "CNP","PPL","ES","AWK","FE"]},
    "XLC":  {"name": "Comm. Services",    "holdings": [
                "META","GOOGL","GOOG","WBD","EA","NFLX","DIS","TTWO","OMC","VZ",
                "CMCSA","T","TMUS","LYV","CHTR","TTD","FOXA","TKO","NWSA","FOX",
                "MTCH","PSKY","NWS"]},
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
    return raw if isinstance(raw, pd.Series) and len(raw) >= 40 else None


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

    rsi_vals  = calc_rsi_fast(closes)
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
