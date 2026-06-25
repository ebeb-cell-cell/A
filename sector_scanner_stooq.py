"""
Sector ETF · RSI / MACD Scanner  (Stooq → yfinance fallback + Twelve Data)
===========================================================================
Phase 1+2 — Stooq (preferred) or yfinance Adj Close (auto-fallback)
Phase 3   — Twelve Data (accurate fine-check) or same fallback source

At startup the script probes Stooq with a single test request.
  • Stooq reachable  → use Stooq (free, no API key, clean CSV, no rate limit)
  • Stooq blocked    → fall back to yfinance with auto_adjust=False + Adj Close
                       This column is pre-computed by Yahoo separately and does
                       not suffer from the auto_adjust=True dividend-artifact bug
                       that caused false RSI extremes in the original script.

Twelve Data is still used for Phase 3 (fine-check) because it provides
TradingView-accurate adjusted prices and is the authoritative source.
--tdkey is optional; without it Phase 3 uses the same fallback source.

Requirements:
    pip install pandas numpy requests rich yfinance

Usage:
    python sector_scanner_stooq.py                            # auto-detect source
    python sector_scanner_stooq.py --tdkey YOUR_KEY           # + TD fine-check
    python sector_scanner_stooq.py --tdkey YOUR_KEY --out results.csv
    python sector_scanner_stooq.py --tdkey YOUR_KEY --debug
    python sector_scanner_stooq.py --tdkey YOUR_KEY --relax
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

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
TD_MIN_DELAY  = 7.6
_last_td_call = 0.0

def _td_throttle():
    global _last_td_call
    wait = _last_td_call + TD_MIN_DELAY - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_td_call = time.monotonic()

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
    """RSI matching Pine Script ta.rsi() exactly (Wilder's smoothing)."""
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

# ── Data source: Stooq with yfinance Adj Close fallback ──────────────────────

_USE_STOOQ: bool = False   # set in main() after connectivity probe


def _probe_stooq(timeout: float = 5.0) -> bool:
    """Return True if Stooq is reachable and returning valid CSV."""
    try:
        r = requests.get(
            "https://stooq.com/q/d/l/?s=xle.us&i=d",
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return r.status_code == 200 and "Close" in r.text
    except Exception:
        return False


def _clean(s: pd.Series) -> "pd.Series | None":
    """Drop NaNs and rows with single-day moves > 20% (split/dividend artifacts)."""
    s = s.dropna()
    bad = s.pct_change().abs() > 0.20
    if bad.any():
        s = s[~bad]
    return s if isinstance(s, pd.Series) and len(s) >= 40 else None


def _stooq_ticker(symbol: str) -> str:
    """BRK-B → brk.b.us  (Stooq uses lowercase dot-separated tickers)"""
    return symbol.replace("-", ".").lower() + ".us"


def _fetch_stooq(symbol: str, months: int = 9) -> "pd.Series | None":
    end   = datetime.now()
    start = end - timedelta(days=months * 31)
    url   = (
        "https://stooq.com/q/d/l/"
        f"?s={_stooq_ticker(symbol)}"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
        "&i=d"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
        if df.empty or "Close" not in df.columns:
            return None
        return _clean(df.set_index("Date")["Close"].sort_index())
    except Exception:
        return None


def _fetch_yfinance_adj(symbol: str, period: str = "9mo") -> "pd.Series | None":
    """
    yfinance with auto_adjust=False + Adj Close column.
    'Adj Close' is pre-computed by Yahoo and does not suffer from the
    in-place auto_adjust rewriting that can produce fake price spikes.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval="1d",
                         progress=False, auto_adjust=False, threads=False)
        if df.empty:
            return None
        # Handle both flat and MultiIndex column layouts
        if isinstance(df.columns, pd.MultiIndex):
            try:
                s = df["Adj Close"][symbol]
            except KeyError:
                s = df["Adj Close"].iloc[:, 0]
        else:
            s = df["Adj Close"]
        return _clean(s.squeeze())
    except Exception:
        return None


def fetch_p12(symbol: str) -> "pd.Series | None":
    """Phase 1+2 fetch — Stooq if available, yfinance Adj Close otherwise."""
    return _fetch_stooq(symbol) if _USE_STOOQ else _fetch_yfinance_adj(symbol)


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

# ── Phase 2: coarse RSI screen ───────────────────────────────────────────────

def coarse_screen(symbol: str, etf_key: str, etf_trend: str) -> "dict | None":
    closes   = fetch_p12(symbol)
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

# ── Phase 3: fine-check (Twelve Data or Stooq fallback) ──────────────────────

def fine_check(candidate: dict, debug: bool = False,
               relax: bool = False) -> "dict | None":
    """
    Verify all momentum conditions.
    Uses Twelve Data if --tdkey was provided, else falls back to Stooq.
    debug=True  — print per-condition breakdown.
    relax=True  — require 2/3 momentum indicators instead of all 3.
    """
    symbol    = candidate["symbol"]
    etf_key   = candidate["etf_key"]
    etf_trend = candidate["etf_trend"]

    closes = fetch_twelvedata(symbol) if TD_API_KEY else fetch_p12(symbol)
    if closes is None or len(closes) < 40:
        if debug:
            print(f"    {symbol:6s}  ✗ no data")
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
        src_str  = "TD" if TD_API_KEY else "Stooq"
        print(
            f"    {symbol:6s}  {mode_str}  [{src_str}]  "
            f"{'✓' if thresh_ok else '✗'} {thresh_label} ({cur_rsi:.1f})  "
            f"{cond_str}  "
            f"→ {passing}/{len(conditions)} pass  "
            f"{'✅ SIGNAL' if thresh_ok and passing >= required else '✗ filtered'}"
        )

    if not (thresh_ok and passing >= required):
        return None

    return {
        "Signal":          needed_dir,
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
        "RSI Thresh":      "<45 ✓" if needed_dir == "BUY" else ">55 ✓",
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

# ── Parallel runner ───────────────────────────────────────────────────────────

def _run_parallel(jobs, fn, workers, prog_label, console):
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

    parser = argparse.ArgumentParser(
        description="Sector ETF RSI/MACD Scanner — Stooq (P1+2) + Twelve Data (P3)")
    parser.add_argument("--tdkey",   default=None,
                        help="Twelve Data API key for Phase 3 fine-check. "
                             "Without it Phase 3 also uses Stooq.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for Stooq phases (default 10).")
    parser.add_argument("--out",     default=None,
                        help="Optional CSV output path, e.g. results.csv")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-condition breakdown for every Phase 3 candidate.")
    parser.add_argument("--relax",   action="store_true",
                        help="Require 2/3 momentum indicators instead of all 3.")
    args = parser.parse_args()

    global TD_API_KEY, _USE_STOOQ

    if args.tdkey:
        TD_API_KEY = args.tdkey

    console = Console() if RICH else None
    started = datetime.now()

    # ── Probe data source for Phase 1+2 ───────────────────────────────────
    print("\n📡  Sector ETF · RSI / MACD Scanner")
    print(f"    {started.strftime('%Y-%m-%d %H:%M')}")
    print("    Probing Phase 1+2 data source … ", end="", flush=True)
    _USE_STOOQ = _probe_stooq()
    if _USE_STOOQ:
        p12_src = "Stooq (no API key)"
        print("✓  Stooq reachable")
    else:
        p12_src = "yfinance Adj Close (auto_adjust=False)"
        print("✗  Stooq blocked — using yfinance Adj Close")

    p3_src = "Twelve Data" if TD_API_KEY else p12_src
    print(f"    Phase 1+2 : {p12_src}  (parallel, workers={args.workers})")
    print(f"    Phase 3   : {p3_src}  (accurate indicators, sequential)\n")

    # ── Phase 1: ETF trends via Stooq ─────────────────────────────────────
    print(f"Phase 1 — Sector ETF trends ({p12_src}) …\n")

    etf_trends = {}
    for etf in SECTOR_ETFS:
        closes = fetch_p12(etf)
        if closes is None:
            trend, price = "unknown", "–"
        else:
            trend = determine_trend(closes)
            price = f"${float(closes.iloc[-1]):.2f}"
        etf_trends[etf] = trend
        arrow = "📈" if trend == "up" else "📉" if trend == "down" else "➖"
        print(f"  {etf:5s}  {arrow}  {trend.upper():8s}  {price:>8s}   {SECTOR_ETFS[etf]['name']}")

    # ── Phase 2: Coarse RSI screen via Stooq (parallel) ───────────────────
    print(f"\nPhase 2 — Coarse RSI screen ({p12_src}, parallel) …\n")

    screen_jobs = [
        (sym, etf, etf_trends[etf])
        for etf, data in SECTOR_ETFS.items()
        for sym in data["holdings"]
        if etf_trends.get(etf) in ("up", "down")
    ]

    candidates, screen_failed = _run_parallel(
        screen_jobs, lambda j: coarse_screen(*j),
        args.workers, "Screening…", console)

    print(f"\n  → {len(candidates)} candidate(s) passed coarse RSI screen "
          f"(from {len(screen_jobs)} stocks)\n")
    if candidates:
        for c in candidates:
            direction = "oversold" if c["etf_trend"] == "up" else "overbought"
            print(f"     {c['symbol']:6s}  [{c['etf_key']}]  {direction}  "
                  f"RSI extreme={c['extreme_val']}")
    print()

    if not candidates:
        print("  No candidates — nothing to fine-check. Exiting.\n")
        return

    # ── Phase 3: Fine-check (Twelve Data sequential, or Stooq) ────────────
    mode_note = ""
    if args.relax:
        mode_note += "  [--relax: 2/3 indicators]"
    if args.debug:
        mode_note += "  [--debug]"
    print(f"Phase 3 — Fine-check {len(candidates)} candidate(s) via {p3_src} …{mode_note}\n")

    results, fine_failed = [], []
    total = len(candidates)

    for i, cand in enumerate(candidates, 1):
        sym = cand["symbol"]
        if not args.debug:
            bar = "█" * int(i/total*30) + "░" * (30 - int(i/total*30))
            print(f"\r  [{bar}] {i}/{total}  {sym:6s}", end="", flush=True)
        else:
            print(f"  [{i}/{total}]  {sym}")
        try:
            r = fine_check(cand, debug=args.debug, relax=args.relax)
            if r:
                results.append(r)
        except Exception as e:
            fine_failed.append(sym)
            if args.debug:
                print(f"    {sym:6s}  ✗ exception: {e}")

    if not args.debug:
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
