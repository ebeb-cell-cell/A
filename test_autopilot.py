"""Tests for autopilot.py — all network calls mocked."""

import asyncio
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import autopilot as ap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HOLDINGS_API_RESPONSE = {
    "holdings": [
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "weight": 0.25, "quantity": 2,   "entry_price": 450.00, "entry_date": "2024-09-01T10:00:00"},
        {"ticker": "MSFT", "name": "Microsoft",          "weight": 0.20, "quantity": 3,   "entry_price": 320.00, "entry_date": "2024-09-01T10:01:00"},
        {"ticker": "AVGO", "name": "Broadcom Inc.",      "weight": 0.30, "quantity": 1,   "entry_price": 1600.0, "entry_date": "2024-09-02T09:30:00"},
        {"ticker": "NOW",  "name": "ServiceNow",         "weight": 0.15, "quantity": 0.5, "entry_price": 800.00, "entry_date": "2024-10-15T14:00:00"},
        {"ticker": "ZETA", "name": "Zeta Global",        "weight": 0.10, "quantity": 10,  "entry_price": 25.00,  "entry_date": "2024-11-01T09:00:00"},
    ],
    "trades": [
        {"date": "2024-09-01", "action": "buy",  "ticker": "NVDA", "price": 450.00, "quantity": 2},
        {"date": "2024-10-15", "action": "buy",  "ticker": "NOW",  "price": 800.00, "quantity": 0.5},
        {"date": "2024-11-01", "action": "sell", "ticker": "MSFT", "price": 370.00, "quantity": 1},
        {"date": "2024-11-01", "action": "buy",  "ticker": "ZETA", "price": 25.00,  "quantity": 10},
    ],
}

NEXT_DATA = {
    "props": {
        "pageProps": {
            "holdings": [
                {"ticker": "AAPL", "weight": 0.50, "quantity": 5, "entry_price": 180.0, "entry_date": "2025-01-01"},
                {"ticker": "TSLA", "weight": 0.50, "quantity": 2, "entry_price": 200.0, "entry_date": "2025-01-01"},
            ]
        }
    }
}

MARKET_DATA = {
    "NVDA": {"last_price": 900.0,  "currency": "USD", "longName": "NVIDIA Corporation",  "isin": "US67066G1040"},
    "MSFT": {"last_price": 420.0,  "currency": "USD", "longName": "Microsoft Corporation","isin": "US5949181045"},
    "AVGO": {"last_price": 1800.0, "currency": "USD", "longName": "Broadcom Inc.",        "isin": "US11135F1012"},
    "NOW":  {"last_price": 900.0,  "currency": "USD", "longName": "ServiceNow Inc.",      "isin": "US81762P1021"},
    "ZETA": {"last_price": 30.0,   "currency": "USD", "longName": "Zeta Global Holdings", "isin": "US98980F1012"},
    "AAPL": {"last_price": 195.0,  "currency": "USD", "longName": "Apple Inc.",           "isin": "US0378331005"},
    "TSLA": {"last_price": 250.0,  "currency": "USD", "longName": "Tesla Inc.",           "isin": "US88160R1014"},
}


def _mock_yf_ticker(symbol: str):
    md = MARKET_DATA.get(symbol, {})
    t = MagicMock()
    t.fast_info = MagicMock()
    t.fast_info.last_price = md.get("last_price")
    t.fast_info.currency   = md.get("currency", "")
    t.info = {
        "longName": md.get("longName", symbol),
        "currency": md.get("currency", ""),
        "currentPrice": md.get("last_price"),
    }
    type(t).isin = PropertyMock(return_value=md.get("isin", "N/A"))
    return t


# ---------------------------------------------------------------------------
# parse_holdings
# ---------------------------------------------------------------------------

class TestParseHoldings(unittest.TestCase):
    def test_from_api_holdings_key(self):
        api = {"https://api.example.com/data": HOLDINGS_API_RESPONSE}
        holdings, trades = ap.parse_holdings(api, {})
        self.assertEqual(len(holdings), 5)
        tickers = {h["ticker"] for h in holdings}
        self.assertIn("NVDA", tickers)
        self.assertIn("ZETA", tickers)

    def test_from_api_trades_extracted(self):
        api = {"https://api.example.com/data": HOLDINGS_API_RESPONSE}
        _, trades = ap.parse_holdings(api, {})
        self.assertEqual(len(trades), 4)

    def test_weight_fraction_normalised_to_percent(self):
        api = {"u": {"holdings": [{"ticker": "AAPL", "weight": 0.30}]}}
        holdings, _ = ap.parse_holdings(api, {})
        self.assertAlmostEqual(holdings[0]["weight"], 30.0)

    def test_weight_already_percent_unchanged(self):
        api = {"u": {"holdings": [{"ticker": "AAPL", "weight": 30.0}]}}
        holdings, _ = ap.parse_holdings(api, {})
        self.assertAlmostEqual(holdings[0]["weight"], 30.0)

    def test_from_next_data_fallback(self):
        dom = {"source": "__NEXT_DATA__", "data": NEXT_DATA}
        holdings, _ = ap.parse_holdings({}, dom)
        self.assertEqual(len(holdings), 2)
        tickers = {h["ticker"] for h in holdings}
        self.assertIn("AAPL", tickers)
        self.assertIn("TSLA", tickers)

    def test_api_takes_priority_over_dom(self):
        api = {"u": {"holdings": [{"ticker": "NVDA", "weight": 0.25}]}}
        dom = {"source": "__NEXT_DATA__", "data": NEXT_DATA}
        holdings, _ = ap.parse_holdings(api, dom)
        tickers = {h["ticker"] for h in holdings}
        self.assertIn("NVDA", tickers)
        self.assertNotIn("AAPL", tickers)

    def test_unknown_ticker_key_variants(self):
        for key in ("ticker", "symbol", "stock", "instrument"):
            api = {"u": {"holdings": [{key: "AAPL", "weight": 0.1}]}}
            holdings, _ = ap.parse_holdings(api, {})
            self.assertEqual(holdings[0]["ticker"], "AAPL", f"failed for key={key}")

    def test_empty_returns_empty(self):
        holdings, trades = ap.parse_holdings({}, {})
        self.assertEqual(holdings, [])
        self.assertEqual(trades, [])

    def test_ticker_uppercased(self):
        api = {"u": {"holdings": [{"ticker": "nvda", "weight": 0.1}]}}
        holdings, _ = ap.parse_holdings(api, {})
        self.assertEqual(holdings[0]["ticker"], "NVDA")


class TestParseHoldingsNested(unittest.TestCase):
    def test_deeply_nested_holdings(self):
        api = {"u": {"data": {"result": {"holdings": [{"ticker": "MSFT", "weight": 0.2}]}}}}
        holdings, _ = ap.parse_holdings(api, {})
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["ticker"], "MSFT")


# ---------------------------------------------------------------------------
# parse_trades
# ---------------------------------------------------------------------------

class TestParseTrades(unittest.TestCase):
    def test_normalises_fields(self):
        raw = [{"date": "2024-09-01", "action": "buy", "ticker": "nvda", "price": 450.0, "quantity": 2}]
        trades = ap.parse_trades(raw)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["action"], "BUY")
        self.assertEqual(trades[0]["ticker"], "NVDA")
        self.assertEqual(trades[0]["price"], 450.0)

    def test_side_and_symbol_aliases(self):
        raw = [{"date": "2024-01-01", "side": "sell", "symbol": "aapl", "executed_price": 190.0, "shares": 3}]
        trades = ap.parse_trades(raw)
        self.assertEqual(trades[0]["action"], "SELL")
        self.assertEqual(trades[0]["ticker"], "AAPL")
        self.assertEqual(trades[0]["price"], 190.0)

    def test_skips_non_dicts(self):
        self.assertEqual(ap.parse_trades(["not", "a", "dict"]), [])


# ---------------------------------------------------------------------------
# enrich_with_yfinance
# ---------------------------------------------------------------------------

class TestEnrichWithYfinance(unittest.TestCase):
    def _run(self, holdings):
        with patch("yfinance.Ticker", side_effect=_mock_yf_ticker):
            return ap.enrich_with_yfinance(holdings)

    def test_adds_current_price(self):
        result = self._run([{"ticker": "NVDA", "name": "NVIDIA", "weight": 25.0}])
        self.assertEqual(result[0]["current_price"], 900.0)

    def test_adds_isin(self):
        result = self._run([{"ticker": "MSFT", "name": "Microsoft", "weight": 20.0}])
        self.assertEqual(result[0]["isin"], "US5949181045")

    def test_adds_currency(self):
        result = self._run([{"ticker": "AAPL", "name": "Apple", "weight": 10.0}])
        self.assertEqual(result[0]["currency"], "USD")

    def test_yfinance_error_returns_na(self):
        def bad_ticker(sym):
            raise RuntimeError("network error")
        with patch("yfinance.Ticker", side_effect=bad_ticker):
            result = ap.enrich_with_yfinance([{"ticker": "BAD", "name": "Bad Corp"}])
        self.assertEqual(result[0]["isin"], "N/A")
        self.assertIsNone(result[0]["current_price"])

    def test_preserves_existing_fields(self):
        result = self._run([{"ticker": "NVDA", "weight": 25.0, "entry_price": 450.0}])
        self.assertEqual(result[0]["entry_price"], 450.0)
        self.assertEqual(result[0]["weight"], 25.0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFormatHelpers(unittest.TestCase):
    def test_fmt_price(self):
        self.assertEqual(ap._fmt_price(900.0, "USD"), "USD 900.0000")
        self.assertEqual(ap._fmt_price(900.0), "900.0000")
        self.assertEqual(ap._fmt_price(None), "N/A")

    def test_fmt_pct(self):
        self.assertEqual(ap._fmt_pct(25.0), "25.00%")
        self.assertEqual(ap._fmt_pct(None), "N/A")
        self.assertEqual(ap._fmt_pct("bad"), "bad")

    def test_fmt_dt(self):
        self.assertEqual(ap._fmt_dt("2024-09-01T10:00:00"), "2024-09-01 10:00")
        self.assertEqual(ap._fmt_dt("2024-09-01T10:00:00Z"), "2024-09-01 10:00")
        self.assertEqual(ap._fmt_dt("2024-09-01"), "2024-09-01 00:00")
        self.assertEqual(ap._fmt_dt(None), "N/A")


# ---------------------------------------------------------------------------
# print_holdings_table
# ---------------------------------------------------------------------------

class TestPrintHoldingsTable(unittest.TestCase):
    def _make_holdings(self):
        raw = [ap._normalise_holding(item["ticker"], item) for item in HOLDINGS_API_RESPONSE["holdings"]]
        with patch("yfinance.Ticker", side_effect=_mock_yf_ticker):
            return ap.enrich_with_yfinance(raw)

    def _capture(self):
        holdings = self._make_holdings()
        trades = ap.parse_trades(HOLDINGS_API_RESPONSE["trades"])
        buf = StringIO()
        with patch("sys.stdout", buf):
            ap.print_holdings_table("Claude", holdings, trades)
        return buf.getvalue()

    def test_name_in_output(self):
        self.assertIn("Claude", self._capture())

    def test_tickers_shown(self):
        out = self._capture()
        for ticker in ("NVDA", "MSFT", "AVGO", "NOW", "ZETA"):
            self.assertIn(ticker, out)

    def test_isins_shown(self):
        out = self._capture()
        self.assertIn("US67066G1040", out)
        self.assertIn("US5949181045", out)

    def test_entry_prices_shown(self):
        self.assertIn("450.0000", self._capture())

    def test_current_prices_shown(self):
        self.assertIn("900.0000", self._capture())

    def test_weights_shown(self):
        self.assertIn("25.00%", self._capture())

    def test_trades_section_shown(self):
        out = self._capture()
        self.assertIn("Trades", out)
        self.assertIn("BUY", out)
        self.assertIn("SELL", out)

    def test_sorted_by_weight_descending(self):
        out = self._capture()
        avgo_pos = out.find("AVGO")
        nvda_pos = out.find("NVDA")
        self.assertLess(avgo_pos, nvda_pos)  # AVGO weight=30 should appear before NVDA weight=25


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

class TestMainArgs(unittest.TestCase):
    def _mock_run(self, called_with):
        async def fake_run(targets):
            called_with.extend(targets)
        return fake_run

    def test_all_targets_by_default(self):
        called = []
        with patch("sys.argv", ["autopilot.py"]), \
             patch.object(ap, "run", side_effect=self._mock_run(called)):
            ap.main()
        self.assertEqual(set(called), {"claude", "grok"})

    def test_single_claude(self):
        called = []
        with patch("sys.argv", ["autopilot.py", "claude"]), \
             patch.object(ap, "run", side_effect=self._mock_run(called)):
            ap.main()
        self.assertEqual(called, ["claude"])

    def test_single_grok(self):
        called = []
        with patch("sys.argv", ["autopilot.py", "grok"]), \
             patch.object(ap, "run", side_effect=self._mock_run(called)):
            ap.main()
        self.assertEqual(called, ["grok"])

    def test_unknown_arg_exits(self):
        with patch("sys.argv", ["autopilot.py", "unknown"]):
            with self.assertRaises(SystemExit) as cm:
                ap.main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
