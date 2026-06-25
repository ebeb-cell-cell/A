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
