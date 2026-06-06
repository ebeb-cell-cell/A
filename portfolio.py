#!/usr/bin/env python3
"""
AutoPilot Portfolio Inspector — powered by yfinance

Portfolio holdings are defined in a JSON file (see portfolio.json.example).
yfinance fetches live prices, company names, and ISINs from Yahoo Finance.

Usage:
    python portfolio.py                    # reads portfolio.json
    python portfolio.py my_portfolio.json  # reads specified file
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

DEFAULT_FILE = "portfolio.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_portfolio(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "holdings" not in data:
        raise ValueError("Portfolio file must contain a 'holdings' list.")
    return data


# ---------------------------------------------------------------------------
# yfinance data fetching
# ---------------------------------------------------------------------------

def _get_isin(ticker_obj: yf.Ticker) -> str:
    """Try multiple yfinance sources for ISIN; return 'N/A' if unavailable."""
    # 1. Dedicated isin property (scrapes Yahoo Finance page)
    try:
        val = ticker_obj.isin
        if val and val not in ("-", "None", "N/A"):
            return val
    except Exception:
        pass
    # 2. info dict (available for some instruments)
    try:
        val = ticker_obj.info.get("isin")
        if val and val not in ("-", "None"):
            return val
    except Exception:
        pass
    return "N/A"


def fetch_ticker_data(ticker: str) -> dict:
    """Return {name, isin, currency, current_price} for one ticker."""
    t = yf.Ticker(ticker)

    # fast_info is a lightweight call (no heavy scraping)
    fi = t.fast_info
    current_price = getattr(fi, "last_price", None)
    currency = getattr(fi, "currency", "") or ""

    # longName / shortName come from the heavier .info dict
    info = t.info
    name = info.get("longName") or info.get("shortName") or ticker

    # Fallback price sources in case fast_info is empty
    if current_price is None:
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

    isin = _get_isin(t)

    return {
        "name": name,
        "isin": isin,
        "currency": currency,
        "current_price": current_price,
    }


def fetch_all(tickers: list[str]) -> dict:
    """Fetch market data for all tickers, printing per-ticker progress."""
    result = {}
    for ticker in tickers:
        print(f"  {ticker}…", end=" ", flush=True)
        try:
            result[ticker] = fetch_ticker_data(ticker)
            print("ok")
        except Exception as exc:
            print(f"error ({exc})")
            result[ticker] = {}
    return result


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def build_rows(holdings: list, market_data: dict) -> tuple[list, dict]:
    """
    Merge holding definitions with live market data.
    Returns (rows sorted by value desc, summary dict).
    """
    rows = []
    for h in holdings:
        ticker = h["ticker"]
        qty = float(h.get("quantity", 0))
        md = market_data.get(ticker, {})
        current_price = md.get("current_price")
        value = qty * (current_price or 0.0)
        rows.append({
            "ticker": ticker,
            "name": md.get("name", ticker),
            "isin": md.get("isin", "N/A"),
            "currency": md.get("currency", ""),
            "entry_date": h.get("entry_date"),
            "entry_price": h.get("entry_price"),
            "current_price": current_price,
            "quantity": qty,
            "value": value,
        })

    total_value = sum(r["value"] for r in rows)
    rows.sort(key=lambda x: x["value"], reverse=True)
    return rows, {"total_value": total_value, "count": len(rows)}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_price(value, currency="") -> str:
    if value is None:
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.4f}"


def _fmt_dt(dt_str) -> str:
    if not dt_str:
        return "N/A"
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(dt_str, fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return str(dt_str)


def print_table(portfolio: dict, rows: list, summary: dict):
    name = portfolio.get("name", "Portfolio")
    total = summary["total_value"]

    bar = "=" * 110
    print(f"\n{bar}")
    print(f"  AutoPilot Portfolio: {name}  |  Holdings: {summary['count']}")
    print(f"{bar}\n")

    C = [34, 32, 18, 12, 12, 10]
    header = (
        f"{'Name':<{C[0]}} {'Ticker / ISIN':<{C[1]}} "
        f"{'Entry Date':<{C[2]}} {'Entry Price':>{C[3]}} "
        f"{'Curr. Price':>{C[4]}} {'% Portfolio':>{C[5]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for row in rows:
        pct = (row["value"] / total * 100) if total else 0.0

        label = row["name"]
        if len(label) > C[0] - 1:
            label = label[: C[0] - 2] + "…"

        ticker_isin = f"{row['ticker']} / {row['isin']}"
        if len(ticker_isin) > C[1] - 1:
            ticker_isin = ticker_isin[: C[1] - 2] + "…"

        print(
            f"{label:<{C[0]}} {ticker_isin:<{C[1]}} "
            f"{_fmt_dt(row['entry_date']):<{C[2]}} "
            f"{_fmt_price(row['entry_price'], row['currency']):>{C[3]}} "
            f"{_fmt_price(row['current_price'], row['currency']):>{C[4]}} "
            f"{pct:>{C[5]-1}.2f}%"
        )

    print(sep)
    print(f"\n  Total Portfolio Value: {total:,.2f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    if not Path(path).exists():
        print(f"Error: '{path}' not found.", file=sys.stderr)
        print("  Create a portfolio file — see portfolio.json.example for the format.", file=sys.stderr)
        sys.exit(1)

    portfolio = load_portfolio(path)
    tickers = [h["ticker"] for h in portfolio["holdings"]]

    print(f"Fetching Yahoo Finance data for {len(tickers)} ticker(s)…")
    market_data = fetch_all(tickers)

    rows, summary = build_rows(portfolio["holdings"], market_data)
    print_table(portfolio, rows, summary)


if __name__ == "__main__":
    main()
