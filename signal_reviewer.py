"""
Signal Reviewer — Phase 4 AI analysis of sector_scanner_rich.py signals
========================================================================
For each BUY / SELL signal produced by the scanner this script:
  1. Fetches yfinance fundamentals and recent news headlines
  2. Sends everything to Claude (claude-opus-4-8) for financial review
  3. Returns a structured CONFIRM / NEGATE verdict with reasoning

Usage — standalone (reads a CSV produced by the scanner):
    python signal_reviewer.py --input results.csv --apikey YOUR_ANTHROPIC_KEY

Usage — pipeline (scanner + reviewer in one shot):
    python sector_scanner_rich.py --out results.csv [--tdkey TD_KEY]
    python signal_reviewer.py --input results.csv --apikey YOUR_ANTHROPIC_KEY

Get a free Anthropic API key at: https://console.anthropic.com

Requirements:
    pip install anthropic yfinance pandas rich
"""

import argparse
import json
import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

import anthropic

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False

MODEL = "claude-opus-4-8"

# ── Fundamentals & news via yfinance ─────────────────────────────────────────

def _safe(val, fmt=None):
    if val is None or (isinstance(val, float) and (val != val)):
        return "N/A"
    return fmt % val if fmt else val


def gather_context(symbol: str) -> dict:
    """Pull fundamentals + recent news headlines from yfinance."""
    try:
        t    = yf.Ticker(symbol)
        info = t.info or {}
    except Exception:
        info = {}

    price    = info.get("currentPrice") or info.get("regularMarketPrice")
    fw52_hi  = info.get("fiftyTwoWeekHigh")
    fw52_lo  = info.get("fiftyTwoWeekLow")
    pct_from_hi = (
        round((price - fw52_hi) / fw52_hi * 100, 1)
        if price and fw52_hi else None
    )
    pct_from_lo = (
        round((price - fw52_lo) / fw52_lo * 100, 1)
        if price and fw52_lo else None
    )

    fundamentals = {
        "market_cap_B":     _safe(info.get("marketCap"),       "%.2fB") if info.get("marketCap") else "N/A",
        "trailing_PE":      _safe(info.get("trailingPE"),      "%.1f"),
        "forward_PE":       _safe(info.get("forwardPE"),       "%.1f"),
        "peg_ratio":        _safe(info.get("pegRatio"),        "%.2f"),
        "price_to_book":    _safe(info.get("priceToBook"),     "%.2f"),
        "revenue_growth_pct": _safe(info.get("revenueGrowth"), "%.1f%%") if info.get("revenueGrowth") else "N/A",
        "earnings_growth_pct": _safe(info.get("earningsGrowth"), "%.1f%%") if info.get("earningsGrowth") else "N/A",
        "profit_margin_pct": _safe(info.get("profitMargins"), "%.1f%%") if info.get("profitMargins") else "N/A",
        "debt_to_equity":   _safe(info.get("debtToEquity"),   "%.1f"),
        "current_ratio":    _safe(info.get("currentRatio"),   "%.2f"),
        "short_ratio_days": _safe(info.get("shortRatio"),     "%.1f"),
        "short_pct_float":  _safe(info.get("shortPercentOfFloat"), "%.1f%%") if info.get("shortPercentOfFloat") else "N/A",
        "52w_high":         _safe(fw52_hi,  "%.2f"),
        "52w_low":          _safe(fw52_lo,  "%.2f"),
        "pct_from_52w_high": _safe(pct_from_hi, "%.1f%%") if pct_from_hi is not None else "N/A",
        "pct_from_52w_low":  _safe(pct_from_lo, "%.1f%%") if pct_from_lo is not None else "N/A",
        "analyst_target":   _safe(info.get("targetMeanPrice"), "%.2f"),
        "analyst_rating":   info.get("recommendationKey", "N/A"),
        "next_earnings":    info.get("earningsDate", [None])[0] if isinstance(info.get("earningsDate"), list) else info.get("earningsDate"),
        "beta":             _safe(info.get("beta"), "%.2f"),
        "industry":         info.get("industry", "N/A"),
        "business_summary": (info.get("longBusinessSummary") or "")[:300],
    }

    # Fix market_cap formatting
    mc = info.get("marketCap")
    if mc:
        fundamentals["market_cap_B"] = f"${mc/1e9:.2f}B"

    # Recent news (up to 5 headlines)
    news_headlines = []
    try:
        news = t.news or []
        for n in news[:5]:
            title = n.get("title") or n.get("content", {}).get("title", "")
            ts    = n.get("providerPublishTime") or n.get("content", {}).get("pubDate", "")
            if title:
                news_headlines.append({"title": title, "published": str(ts)[:10] if ts else "?"})
    except Exception:
        pass

    return {"fundamentals": fundamentals, "recent_news": news_headlines}


# ── Claude API review ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior equity analyst specialising in short-to-medium term
(days to weeks) technical and fundamental trading. You are concise, direct, and evidence-based.
You neither over-confirm nor dismiss signals without justification. Today's date is {date}."""

USER_TEMPLATE = """
## Signal to review

| Field             | Value |
|-------------------|-------|
| Direction         | {signal} |
| Symbol            | {symbol} |
| Sector            | {sector} ({etf}) |
| Price             | ${price} |
| Daily RSI         | {drsi} ({rsi_dir}) |
| MACD              | {macd_dir} |
| Weekly RSI        | {wrsi} ({wrsi_dir}) |
| RSI extreme 15d   | {extreme} |
| RSI threshold met | {thresh} |

## Fundamental context

{fundamentals}

## Recent news headlines

{news}

---

Please evaluate this {signal} signal for a short-term trade and provide your answer in
exactly the following format (no markdown headers inside sections):

VERDICT: CONFIRM or NEGATE
CONFIDENCE: HIGH / MEDIUM / LOW

REASONS:
• <reason 1>
• <reason 2>
• <reason 3 if applicable>

RISKS:
• <risk 1>
• <risk 2 if applicable>

ACTION: <one concrete sentence — e.g. "Enter long near ${price} with stop below $X, target $Y" or "Wait for next earnings before entry" or "Avoid — deteriorating fundamentals outweigh technical setup">
"""


def _fmt_fundamentals(f: dict) -> str:
    lines = []
    for k, v in f.items():
        if k == "business_summary":
            continue
        lines.append(f"  {k.replace('_', ' ').title():<28} {v}")
    summary = f.get("business_summary", "")
    if summary:
        lines.append(f"  Business summary: {summary}...")
    return "\n".join(lines)


def _fmt_news(news: list) -> str:
    if not news:
        return "  No recent headlines available."
    return "\n".join(f"  [{n['published']}] {n['title']}" for n in news)


def review_signal(row: dict, client: anthropic.Anthropic) -> dict:
    """Call Claude to review a single signal row and return parsed verdict."""
    ctx = gather_context(row["Symbol"])

    prompt = USER_TEMPLATE.format(
        signal    = row["Signal"],
        symbol    = row["Symbol"],
        sector    = row["Sector"],
        etf       = row["ETF"],
        price     = row["Price"],
        drsi      = row["D-RSI"],
        rsi_dir   = row["RSI Dir"],
        macd_dir  = row["MACD Dir"],
        wrsi      = row["W-RSI"],
        wrsi_dir  = row["W-RSI Dir"],
        extreme   = row["Min/Max RSI 15d"],
        thresh    = row["RSI Thresh"],
        fundamentals = _fmt_fundamentals(ctx["fundamentals"]),
        news         = _fmt_news(ctx["recent_news"]),
    )

    message = client.messages.create(
        model  = MODEL,
        max_tokens = 700,
        system = SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
        messages = [{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()

    # Parse structured fields
    verdict    = "CONFIRM" if "CONFIRM" in text.upper().split("\n")[0] else "NEGATE"
    confidence = "MEDIUM"
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        if lvl in text.upper():
            confidence = lvl
            break

    action = ""
    for line in text.split("\n"):
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()

    return {
        "Symbol":     row["Symbol"],
        "Signal":     row["Signal"],
        "Verdict":    verdict,
        "Confidence": confidence,
        "Action":     action,
        "Full":       text,
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def _verdict_color(v: str) -> str:
    return "green" if v == "CONFIRM" else "red"


def print_rich_review(review: dict):
    console = Console()
    verdict_style = "bold green" if review["Verdict"] == "CONFIRM" else "bold red"
    conf_style    = {"HIGH": "bold yellow", "MEDIUM": "yellow", "LOW": "dim"}.get(review["Confidence"], "white")

    header = (
        f"[bold]{review['Symbol']}[/bold]  "
        f"[cyan]{review['Signal']}[/cyan]  →  "
        f"[{verdict_style}]{review['Verdict']}[/{verdict_style}]  "
        f"[{conf_style}]({review['Confidence']})[/{conf_style}]"
    )
    body = review["Full"]
    console.print(Panel(body, title=header, border_style=_verdict_color(review["Verdict"]), padding=(1, 2)))


def print_rich_summary(reviews: list):
    console = Console()
    t = Table(title="AI Signal Review — Summary", box=box.SIMPLE_HEAVY, border_style="dim")
    t.add_column("Symbol",     style="bold")
    t.add_column("Signal",     style="cyan")
    t.add_column("Verdict",    style="bold")
    t.add_column("Confidence")
    t.add_column("Action",     max_width=60)
    for r in reviews:
        vstyle = "green" if r["Verdict"] == "CONFIRM" else "red"
        t.add_row(
            r["Symbol"],
            r["Signal"],
            f"[{vstyle}]{r['Verdict']}[/{vstyle}]",
            r["Confidence"],
            r["Action"],
        )
    console.print(t)


def print_plain_review(review: dict):
    line = "─" * 72
    verdict = review["Verdict"]
    print(f"\n{line}")
    print(f"  {review['Symbol']}  {review['Signal']}  →  {verdict}  ({review['Confidence']})")
    print(line)
    print(review["Full"])
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI signal reviewer for sector_scanner_rich.py")
    parser.add_argument("--input",  required=True,
                        help="CSV file produced by sector_scanner_rich.py --out <file>")
    parser.add_argument("--apikey", default=None,
                        help="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.")
    parser.add_argument("--delay",  type=float, default=1.0,
                        help="Seconds between API calls to stay within rate limits (default 1.0).")
    parser.add_argument("--out",    default=None,
                        help="Optional CSV path to save review results.")
    args = parser.parse_args()

    # Load signals
    try:
        signals = pd.read_csv(args.input).to_dict(orient="records")
    except FileNotFoundError:
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not signals:
        print("No signals found in input file. Exiting.")
        return

    # Build Anthropic client
    client_kwargs = {}
    if args.apikey:
        client_kwargs["api_key"] = args.apikey
    client = anthropic.Anthropic(**client_kwargs)  # falls back to ANTHROPIC_API_KEY env var

    console = Console() if RICH else None
    started = datetime.now()

    print(f"\n🔍  AI Signal Reviewer  ({len(signals)} signal(s))  —  model: {MODEL}")
    print(f"    {started.strftime('%Y-%m-%d %H:%M')}\n")

    reviews = []
    for i, row in enumerate(signals, 1):
        sym = row.get("Symbol", "?")
        print(f"  [{i}/{len(signals)}]  Reviewing {sym} ({row.get('Signal', '?')}) …", flush=True)

        try:
            review = review_signal(row, client)
            reviews.append(review)
            if RICH:
                print_rich_review(review)
            else:
                print_plain_review(review)
        except anthropic.APIError as e:
            print(f"    ⚠️  API error for {sym}: {e}")
        except Exception as e:
            print(f"    ⚠️  Error for {sym}: {e}")

        if i < len(signals):
            time.sleep(args.delay)

    # Summary table
    if reviews:
        print("\n" + "═" * 72)
        print("  SUMMARY")
        print("═" * 72)
        if RICH:
            print_rich_summary(reviews)
        else:
            for r in reviews:
                print(f"  {r['Symbol']:6s}  {r['Signal']:4s}  {r['Verdict']:7s}  {r['Confidence']:6s}  {r['Action'][:60]}")

    elapsed = (datetime.now() - started).seconds
    confirmed = sum(1 for r in reviews if r["Verdict"] == "CONFIRM")
    negated   = len(reviews) - confirmed
    print(f"\n  Done in {elapsed}s  |  Reviewed: {len(reviews)}  "
          f"Confirmed: {confirmed}  Negated: {negated}\n")

    if args.out and reviews:
        pd.DataFrame([{k: v for k, v in r.items() if k != "Full"} for r in reviews]).to_csv(args.out, index=False)
        print(f"  💾  Review summary saved to {args.out}\n")


if __name__ == "__main__":
    main()
