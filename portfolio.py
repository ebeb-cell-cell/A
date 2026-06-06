#!/usr/bin/env python3
"""
Trading212 AutoPilot (Pie) Portfolio Inspector

Usage:
    T212_API_KEY=<key> python portfolio.py                   # list all pies
    T212_API_KEY=<key> python portfolio.py "My Pie"          # by name (partial match ok)
    T212_API_KEY=<key> python portfolio.py 12345             # by pie ID
    T212_API_KEY=<key> T212_ENV=demo python portfolio.py ... # use demo account
"""

import os
import sys
from datetime import datetime, timezone
import requests

BASE_URLS = {
    "live": "https://live.trading212.com/api/v0",
    "demo": "https://demo.trading212.com/api/v0",
}


def _session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": api_key})
    return s


def fetch_instruments(session: requests.Session, base_url: str) -> dict:
    """Return {ticker: {name, isin, shortName, currency}} for all instruments.

    The endpoint may return a plain list or a paginated object with nextPagePath.
    """
    instruments: dict = {}
    path = f"{base_url}/equity/metadata/instruments"
    while path:
        r = session.get(path)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            items, path = data, None
        else:
            items = data.get("items", [])
            nxt = data.get("nextPagePath")
            path = f"{base_url}{nxt}" if nxt else None
        for item in items:
            ticker = item.get("ticker")
            if ticker:
                instruments[ticker] = {
                    "name": item.get("name", ""),
                    "shortName": item.get("shortName", ticker),
                    "isin": item.get("isin", "N/A"),
                    "currency": item.get("currencyCode", ""),
                }
    return instruments


def fetch_portfolio(session: requests.Session, base_url: str) -> dict:
    """Return {ticker: {currentPrice, averagePricePaid, createdAt, quantity}}."""
    r = session.get(f"{base_url}/equity/portfolio")
    r.raise_for_status()
    return {
        pos["ticker"]: {
            "currentPrice": pos.get("currentPrice"),
            "averagePricePaid": pos.get("averagePricePaid"),
            "createdAt": pos.get("createdAt"),
            "quantity": pos.get("quantity", 0),
        }
        for pos in r.json()
        if pos.get("ticker")
    }


def fetch_pies(session: requests.Session, base_url: str) -> list:
    r = session.get(f"{base_url}/equity/pies")
    r.raise_for_status()
    return r.json()


def fetch_pie(session: requests.Session, base_url: str, pie_id: int) -> dict:
    r = session.get(f"{base_url}/equity/pies/{pie_id}")
    r.raise_for_status()
    return r.json()


def _fmt_price(value, currency=""):
    if value is None:
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.4f}"


def _fmt_dt(dt_str):
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt_str)


def _pie_settings(pie_raw: dict) -> dict:
    return pie_raw.get("settings", pie_raw)


def find_pie(pies: list, query: str):
    for p in pies:
        s = _pie_settings(p)
        if str(s.get("id")) == query:
            return p
    for p in pies:
        s = _pie_settings(p)
        if s.get("name", "").lower() == query.lower():
            return p
    for p in pies:
        s = _pie_settings(p)
        if query.lower() in s.get("name", "").lower():
            return p
    return None


def print_pie_list(pies: list):
    print(f"\n{'ID':>8}  Name")
    print("-" * 50)
    for p in pies:
        s = _pie_settings(p)
        print(f"{s.get('id', '?'):>8}  {s.get('name', '(unnamed)')}")
    print(f"\n{len(pies)} pie(s) found.")
    print("\nUsage: python portfolio.py <pie-name-or-id>")


def build_rows(pie_detail: dict, instruments: dict, portfolio: dict) -> tuple[list, dict]:
    rows = []
    for inst in pie_detail.get("instruments", []):
        ticker = inst.get("ticker", "")
        result = inst.get("result", {})

        # Quantity: prefer result.quantity, fallback to ownedQuantity
        qty = result.get("quantity") or inst.get("ownedQuantity") or 0

        # Entry price: prefer result.priceAvgBuy, fallback to portfolio averagePricePaid
        port = portfolio.get(ticker, {})
        entry_price = result.get("priceAvgBuy") or port.get("averagePricePaid")

        # Current price: prefer live portfolio currentPrice, fallback to entry price
        current_price = port.get("currentPrice") or entry_price

        # Entry time: portfolio createdAt is per position
        entry_time = port.get("createdAt")

        meta = instruments.get(ticker, {})
        value = qty * (current_price or 0)

        rows.append({
            "ticker": ticker,
            "name": meta.get("name") or meta.get("shortName", ticker),
            "isin": meta.get("isin", "N/A"),
            "currency": meta.get("currency", ""),
            "entry_time": entry_time,
            "entry_price": entry_price,
            "current_price": current_price,
            "quantity": qty,
            "value": value,
        })

    total_value = sum(r["value"] for r in rows)
    rows.sort(key=lambda x: x["value"], reverse=True)
    return rows, {"total_value": total_value, "count": len(rows)}


def print_portfolio(pie_detail: dict, instruments: dict, portfolio: dict):
    settings = _pie_settings(pie_detail)
    rows, summary = build_rows(pie_detail, instruments, portfolio)

    pie_name = settings.get("name", "Unknown")
    pie_id = settings.get("id", "?")
    created = _fmt_dt(settings.get("createdAt"))
    dividends = settings.get("dividendCashAction", "N/A")
    total_value = summary["total_value"]

    bar = "=" * 108
    print(f"\n{bar}")
    print(f"  AutoPilot Pie: {pie_name}  (ID: {pie_id})")
    print(f"  Created: {created}  |  Dividends: {dividends}  |  Holdings: {summary['count']}")
    print(f"{bar}\n")

    C = [34, 30, 18, 12, 12, 10]
    header = (
        f"{'Name':<{C[0]}} {'Ticker / ISIN':<{C[1]}} "
        f"{'Entry Date (UTC)':<{C[2]}} {'Entry Price':>{C[3]}} "
        f"{'Curr. Price':>{C[4]}} {'% Portfolio':>{C[5]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for row in rows:
        pct = (row["value"] / total_value * 100) if total_value else 0.0
        name = row["name"]
        if len(name) > C[0] - 1:
            name = name[: C[0] - 2] + "…"
        ticker_isin = f"{row['ticker']} / {row['isin']}"
        if len(ticker_isin) > C[1] - 1:
            ticker_isin = ticker_isin[: C[1] - 2] + "…"

        print(
            f"{name:<{C[0]}} {ticker_isin:<{C[1]}} "
            f"{_fmt_dt(row['entry_time']):<{C[2]}} "
            f"{_fmt_price(row['entry_price'], row['currency']):>{C[3]}} "
            f"{_fmt_price(row['current_price'], row['currency']):>{C[4]}} "
            f"{pct:>{C[5]-1}.2f}%"
        )

    print(sep)
    print(f"\n  Total Pie Market Value: {total_value:,.2f}\n")


def main():
    api_key = os.environ.get("T212_API_KEY")
    if not api_key:
        print("Error: T212_API_KEY environment variable is not set.", file=sys.stderr)
        print("  Generate your API key in Trading212: Settings → API", file=sys.stderr)
        sys.exit(1)

    env = os.environ.get("T212_ENV", "live").lower()
    if env not in BASE_URLS:
        print(f"Error: T212_ENV must be 'live' or 'demo', got '{env}'", file=sys.stderr)
        sys.exit(1)

    base_url = BASE_URLS[env]
    pie_query = sys.argv[1] if len(sys.argv) > 1 else None

    session = _session(api_key)

    print(f"Connecting to Trading212 ({env} account)…")

    try:
        pies = fetch_pies(session, base_url)
    except requests.HTTPError as e:
        print(f"Error fetching pies: {e}", file=sys.stderr)
        if e.response is not None and e.response.status_code == 401:
            print("  Check that your API key is correct and matches your account type (live/demo).", file=sys.stderr)
        sys.exit(1)

    if not pies:
        print("No AutoPilot pies found on this account.")
        sys.exit(0)

    if pie_query is None:
        print_pie_list(pies)
        sys.exit(0)

    pie_raw = find_pie(pies, pie_query)
    if pie_raw is None:
        print(f"Error: No pie found matching '{pie_query}'.", file=sys.stderr)
        print("Run without arguments to list all pies.", file=sys.stderr)
        sys.exit(1)

    pie_id = _pie_settings(pie_raw).get("id")

    print("Fetching pie details…")
    pie_detail = fetch_pie(session, base_url, pie_id)

    print("Fetching portfolio positions…")
    portfolio = fetch_portfolio(session, base_url)

    print("Fetching instrument metadata…")
    instruments = fetch_instruments(session, base_url)

    print_portfolio(pie_detail, instruments, portfolio)


if __name__ == "__main__":
    main()
