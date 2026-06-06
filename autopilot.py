#!/usr/bin/env python3
"""
Autopilot Portfolio Inspector — Claude & Grok

Loads the public joinautopilot.com marketplace pages in a headless Chromium
browser (bypassing CDN bot-detection), intercepts the underlying API calls
for structured holdings/trade data, falls back to DOM/__NEXT_DATA__ parsing,
and enriches each position with live prices and ISINs from yfinance.

Requirements:
    pip install playwright yfinance
    playwright install chromium

Usage:
    python autopilot.py           # both Claude and Grok
    python autopilot.py claude    # Claude only
    python autopilot.py grok      # Grok only
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
from playwright.async_api import async_playwright

PORTFOLIOS = {
    "claude": {
        "name": "Claude",
        "url": "https://marketplace.joinautopilot.com/landing/5/950048",
    },
    "grok": {
        "name": "Grok",
        "url": "https://marketplace.joinautopilot.com/landing/5/568906",
    },
}

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Browser scraping
# ---------------------------------------------------------------------------

async def scrape_page(page, url: str) -> dict:
    """
    Navigate to *url*, intercept every JSON API response, and extract DOM data.
    Returns {"api": {url: json_data}, "dom": {...}}.
    """
    api_responses: dict = {}

    async def on_response(response):
        ct = response.headers.get("content-type", "")
        if "json" in ct:
            try:
                api_responses[response.url] = await response.json()
            except Exception:
                pass

    page.on("response", on_response)
    await page.goto(url, wait_until="networkidle", timeout=45_000)
    await page.wait_for_timeout(2_500)

    dom_data = await _extract_dom(page)
    return {"api": api_responses, "dom": dom_data}


async def _extract_dom(page) -> dict:
    """Pull structured data and fallback text from the rendered DOM."""
    return await page.evaluate(
        """
        () => {
            // Next.js server-side props (most reliable if present)
            const nd = document.getElementById('__NEXT_DATA__');
            if (nd) {
                try { return { source: '__NEXT_DATA__', data: JSON.parse(nd.textContent) }; }
                catch(e) {}
            }
            // Inline <script type="application/json"> blobs
            const blobs = Array.from(
                document.querySelectorAll('script[type="application/json"]')
            ).map(s => { try { return JSON.parse(s.textContent); } catch(e) { return null; } })
             .filter(Boolean);
            if (blobs.length) return { source: 'json_scripts', data: blobs };

            // Plain page text as last resort
            return { source: 'text', data: document.body.innerText };
        }
        """
    )


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------

def _coerce_ticker(raw) -> str:
    return str(raw).strip().upper() if raw else ""


def _find_list(obj, *keys):
    """Walk a nested dict looking for the first key that holds a list."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        for v in obj.values():
            result = _find_list(v, *keys)
            if result is not None:
                return result
    return None


def parse_holdings(captured_api: dict, dom: dict) -> list[dict]:
    """
    Try multiple sources (intercepted API → __NEXT_DATA__ → DOM text)
    and return a normalised list of holding dicts.
    """
    HOLDING_KEYS = ("holdings", "positions", "stocks", "portfolio",
                    "instruments", "allocations", "tickers")
    TRADE_KEYS   = ("trades", "orders", "transactions", "history", "activity")

    holdings: list = []
    trades: list   = []

    # 1. Intercepted API responses
    for _url, data in captured_api.items():
        items = _find_list(data, *HOLDING_KEYS)
        if items:
            for item in items:
                if not isinstance(item, dict):
                    continue
                ticker = _coerce_ticker(
                    item.get("ticker") or item.get("symbol") or
                    item.get("stock") or item.get("instrument")
                )
                if ticker:
                    holdings.append(_normalise_holding(ticker, item))

        for key in TRADE_KEYS:
            tlist = data.get(key) if isinstance(data, dict) else None
            if isinstance(tlist, list):
                trades.extend(tlist)

    if holdings:
        return holdings, trades

    # 2. __NEXT_DATA__ / JSON script blobs
    source = dom.get("source", "")
    data   = dom.get("data")

    if source in ("__NEXT_DATA__", "json_scripts") and data:
        blobs = data if isinstance(data, list) else [data]
        for blob in blobs:
            items = _find_list(blob, *HOLDING_KEYS)
            if items:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    ticker = _coerce_ticker(
                        item.get("ticker") or item.get("symbol") or item.get("stock")
                    )
                    if ticker:
                        holdings.append(_normalise_holding(ticker, item))
                if holdings:
                    break

    return holdings, trades


def _normalise_holding(ticker: str, raw: dict) -> dict:
    weight = raw.get("weight") or raw.get("percentage") or raw.get("allocation") or raw.get("target_pct")
    if weight is not None:
        try:
            weight = float(weight)
            if weight <= 1.0:           # stored as 0-1 fraction
                weight *= 100
        except (TypeError, ValueError):
            weight = None

    entry_price = (
        raw.get("avg_price") or raw.get("averagePricePaid") or
        raw.get("entry_price") or raw.get("cost_basis") or
        raw.get("average_cost") or raw.get("avgCost")
    )

    entry_date = (
        raw.get("entry_date") or raw.get("createdAt") or
        raw.get("opened_at") or raw.get("date")
    )

    return {
        "ticker":      ticker,
        "name":        raw.get("name") or raw.get("company_name") or raw.get("longName") or "",
        "weight":      weight,
        "quantity":    raw.get("quantity") or raw.get("shares"),
        "entry_price": entry_price,
        "entry_date":  entry_date,
        "_raw":        raw,
    }


def parse_trades(trade_list: list) -> list[dict]:
    out = []
    for t in trade_list:
        if not isinstance(t, dict):
            continue
        out.append({
            "date":   t.get("date") or t.get("created_at") or t.get("timestamp") or t.get("executed_at"),
            "action": (t.get("action") or t.get("side") or t.get("type") or "?").upper(),
            "ticker": _coerce_ticker(t.get("ticker") or t.get("symbol") or "?"),
            "price":  t.get("price") or t.get("executed_price") or t.get("fill_price"),
            "qty":    t.get("quantity") or t.get("shares") or t.get("amount"),
        })
    return out


# ---------------------------------------------------------------------------
# yfinance enrichment
# ---------------------------------------------------------------------------

def _get_isin(t) -> str:
    try:
        v = t.isin
        if v and v not in ("-", "None", "N/A"):
            return v
    except Exception:
        pass
    try:
        v = t.info.get("isin")
        if v and v not in ("-", "None"):
            return v
    except Exception:
        pass
    return "N/A"


def enrich_with_yfinance(holdings: list) -> list:
    enriched = []
    for h in holdings:
        ticker = h.get("ticker", "")
        extra: dict = {"isin": "N/A", "current_price": None, "currency": ""}
        if ticker:
            try:
                t     = yf.Ticker(ticker)
                fi    = t.fast_info
                info  = t.info
                extra["current_price"] = (
                    getattr(fi, "last_price", None)
                    or info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                )
                extra["currency"] = getattr(fi, "currency", "") or info.get("currency", "")
                if not h.get("name"):
                    extra["name"] = info.get("longName") or info.get("shortName") or ticker
                extra["isin"] = _get_isin(t)
            except Exception:
                pass
        enriched.append({**h, **extra})
    return enriched


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_price(value, currency="") -> str:
    if value is None:
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.4f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_dt(value) -> str:
    if not value:
        return "N/A"
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return str(value)[:16]


def print_holdings_table(portfolio_name: str, holdings: list, trades: list):
    total_value = sum(
        (h.get("quantity") or 0) * (h.get("current_price") or 0)
        for h in holdings
    )

    def sort_key(h):
        w = h.get("weight")
        if w is not None:
            return float(w)
        return (h.get("quantity") or 0) * (h.get("current_price") or 0)

    sorted_h = sorted(holdings, key=sort_key, reverse=True)

    bar = "=" * 118
    print(f"\n{bar}")
    print(f"  {portfolio_name} Portfolio on Autopilot  |  Holdings: {len(holdings)}")
    print(f"{bar}\n")

    # Holdings
    C = [30, 36, 18, 13, 13, 8]
    header = (
        f"{'Name':<{C[0]}} {'Ticker / ISIN':<{C[1]}} "
        f"{'Entry Date':<{C[2]}} {'Entry Price':>{C[3]}} "
        f"{'Curr. Price':>{C[4]}} {'Weight':>{C[5]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for h in sorted_h:
        ticker   = h.get("ticker", "?")
        isin     = h.get("isin", "N/A")
        name_str = (h.get("name") or ticker)
        if len(name_str) > C[0] - 1:
            name_str = name_str[: C[0] - 2] + "…"
        ti_str = f"{ticker} / {isin}"
        if len(ti_str) > C[1] - 1:
            ti_str = ti_str[: C[1] - 2] + "…"
        curr = h.get("currency", "")
        print(
            f"{name_str:<{C[0]}} {ti_str:<{C[1]}} "
            f"{_fmt_dt(h.get('entry_date')):<{C[2]}} "
            f"{_fmt_price(h.get('entry_price'), curr):>{C[3]}} "
            f"{_fmt_price(h.get('current_price'), curr):>{C[4]}} "
            f"{_fmt_pct(h.get('weight')):>{C[5]}}"
        )

    print(sep)
    if total_value:
        print(f"\n  Total visible value: {total_value:,.2f}")

    # Trades
    if trades:
        print(f"\n  Trades ({len(trades)}):\n")
        th = f"  {'Date':<18} {'Action':<8} {'Ticker':<10} {'Price':>12} {'Qty':>10}"
        print(th)
        print("  " + "-" * (len(th) - 2))
        for t in trades[:25]:
            price = t.get("price")
            qty   = t.get("qty")
            print(
                f"  {_fmt_dt(t.get('date')):<18} "
                f"{str(t.get('action','?')):<8} "
                f"{str(t.get('ticker','?')):<10} "
                f"{f'{price:,.4f}' if price else 'N/A':>12} "
                f"{f'{qty:,.4f}' if isinstance(qty,(int,float)) else str(qty or 'N/A'):>10}"
            )

    print(f"\n{'=' * 118}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(targets: list):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=_BROWSER_UA,
            viewport={"width": 1280, "height": 900},
            java_script_enabled=True,
        )
        page = await ctx.new_page()

        for key in targets:
            cfg = PORTFOLIOS[key]
            print(f"\nLoading {cfg['name']} portfolio…")

            try:
                raw = await scrape_page(page, cfg["url"])
            except Exception as e:
                print(f"  Page load failed: {e}")
                continue

            holdings, raw_trades = parse_holdings(raw["api"], raw["dom"])
            trades = parse_trades(raw_trades)

            if not holdings:
                dump_path = Path(f"{key}_raw.json")
                dump_path.write_text(
                    json.dumps(
                        {
                            "api_urls": list(raw["api"].keys()),
                            "api_sample": {k: v for k, v in list(raw["api"].items())[:3]},
                            "dom": raw["dom"] if raw["dom"].get("source") != "text"
                                   else {"source": "text", "preview": str(raw["dom"].get("data", ""))[:3000]},
                        },
                        indent=2,
                        default=str,
                    )
                )
                print(
                    f"  Could not extract structured holdings.\n"
                    f"  Raw data saved to {dump_path} — inspect it to see the actual API shape.\n"
                    f"  API URLs captured: {list(raw['api'].keys())[:5]}"
                )
                continue

            print(f"  Fetching yfinance data for {len(holdings)} ticker(s)…")
            holdings = enrich_with_yfinance(holdings)
            print_holdings_table(cfg["name"], holdings, trades)

        await browser.close()


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if arg == "all":
        targets = list(PORTFOLIOS.keys())
    elif arg in PORTFOLIOS:
        targets = [arg]
    else:
        print(f"Unknown target '{arg}'. Choose: {', '.join(PORTFOLIOS.keys())} or 'all'")
        sys.exit(1)
    asyncio.run(run(targets))


if __name__ == "__main__":
    main()
