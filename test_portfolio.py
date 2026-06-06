"""Tests for portfolio.py — yfinance calls are mocked throughout."""

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import portfolio as p


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HOLDINGS = [
    {"ticker": "AAPL", "quantity": 5,   "entry_price": 178.50, "entry_date": "2023-07-10T14:30:00"},
    {"ticker": "MSFT", "quantity": 3,   "entry_price": 320.00, "entry_date": "2023-07-10T14:31:00"},
    {"ticker": "NVDA", "quantity": 2,   "entry_price": 450.00, "entry_date": "2023-08-01T10:00:00"},
]

PORTFOLIO_DOC = {"name": "Global Tech", "holdings": HOLDINGS}

MARKET_DATA = {
    "AAPL": {"name": "Apple Inc.",           "isin": "US0378331005", "currency": "USD", "current_price": 191.00},
    "MSFT": {"name": "Microsoft Corporation","isin": "US5949181045", "currency": "USD", "current_price": 355.00},
    "NVDA": {"name": "NVIDIA Corporation",   "isin": "US67066G1040", "currency": "USD", "current_price": 600.00},
}


def _make_ticker_mock(name, isin, currency, last_price):
    """Build a mock yf.Ticker whose attributes mirror the real API."""
    t = MagicMock()
    t.info = {
        "longName": name,
        "currency": currency,
        "currentPrice": last_price,
    }
    t.fast_info = MagicMock()
    t.fast_info.last_price = last_price
    t.fast_info.currency = currency
    # isin is a property in the real class
    type(t).isin = PropertyMock(return_value=isin)
    return t


def _ticker_side_effect(symbol):
    md = MARKET_DATA.get(symbol)
    if md is None:
        raise ValueError(f"Unknown mock ticker: {symbol}")
    return _make_ticker_mock(md["name"], md["isin"], md["currency"], md["current_price"])


# ---------------------------------------------------------------------------
# load_portfolio
# ---------------------------------------------------------------------------

class TestLoadPortfolio(unittest.TestCase):
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(PORTFOLIO_DOC, f)
            name = f.name
        result = p.load_portfolio(name)
        self.assertEqual(result["name"], "Global Tech")
        self.assertEqual(len(result["holdings"]), 3)

    def test_missing_holdings_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"name": "bad"}, f)
            name = f.name
        with self.assertRaises(ValueError):
            p.load_portfolio(name)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            p.load_portfolio("/nonexistent/path/to/file.json")


# ---------------------------------------------------------------------------
# fetch_ticker_data / _get_isin
# ---------------------------------------------------------------------------

class TestFetchTickerData(unittest.TestCase):
    def _run(self, symbol="AAPL"):
        mock_ticker = _ticker_side_effect(symbol)
        with patch("yfinance.Ticker", return_value=mock_ticker):
            return p.fetch_ticker_data(symbol)

    def test_name(self):
        self.assertEqual(self._run()["name"], "Apple Inc.")

    def test_isin(self):
        self.assertEqual(self._run()["isin"], "US0378331005")

    def test_currency(self):
        self.assertEqual(self._run()["currency"], "USD")

    def test_current_price(self):
        self.assertEqual(self._run()["current_price"], 191.00)

    def test_isin_fallback_to_info(self):
        t = _make_ticker_mock("Test Corp", "US9999999999", "USD", 100.0)
        # Make the isin property raise so we fall through to info dict
        type(t).isin = PropertyMock(side_effect=Exception("scrape failed"))
        t.info["isin"] = "US9999999999"
        with patch("yfinance.Ticker", return_value=t):
            result = p.fetch_ticker_data("TEST")
        self.assertEqual(result["isin"], "US9999999999")

    def test_isin_returns_na_when_unavailable(self):
        t = _make_ticker_mock("Test Corp", None, "USD", 100.0)
        type(t).isin = PropertyMock(return_value="-")
        t.info.pop("isin", None)
        with patch("yfinance.Ticker", return_value=t):
            result = p.fetch_ticker_data("TEST")
        self.assertEqual(result["isin"], "N/A")

    def test_price_fallback_from_info(self):
        t = _make_ticker_mock("Test Corp", "US0000000000", "USD", None)
        t.fast_info.last_price = None
        t.info["regularMarketPrice"] = 42.0
        with patch("yfinance.Ticker", return_value=t):
            result = p.fetch_ticker_data("TEST")
        self.assertEqual(result["current_price"], 42.0)


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------

class TestBuildRows(unittest.TestCase):
    def _run(self):
        return p.build_rows(HOLDINGS, MARKET_DATA)

    def test_row_count(self):
        rows, summary = self._run()
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["count"], 3)

    def test_sorted_by_value_descending(self):
        rows, _ = self._run()
        values = [r["value"] for r in rows]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_total_value(self):
        _, summary = self._run()
        expected = 5 * 191.00 + 3 * 355.00 + 2 * 600.00
        self.assertAlmostEqual(summary["total_value"], expected, places=4)

    def test_percentages_sum_to_100(self):
        rows, summary = self._run()
        total = summary["total_value"]
        pct_sum = sum(r["value"] / total * 100 for r in rows)
        self.assertAlmostEqual(pct_sum, 100.0, places=5)

    def test_isin_propagated(self):
        rows, _ = self._run()
        isins = {r["isin"] for r in rows}
        self.assertIn("US0378331005", isins)
        self.assertIn("US67066G1040", isins)

    def test_entry_fields_preserved(self):
        rows, _ = self._run()
        aapl = next(r for r in rows if r["ticker"] == "AAPL")
        self.assertEqual(aapl["entry_price"], 178.50)
        self.assertEqual(aapl["entry_date"], "2023-07-10T14:30:00")

    def test_missing_market_data_gives_zero_value(self):
        rows, summary = p.build_rows(
            [{"ticker": "UNKNOWN", "quantity": 10, "entry_price": 50.0}],
            {}
        )
        self.assertEqual(rows[0]["value"], 0.0)
        self.assertEqual(summary["total_value"], 0.0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFormatHelpers(unittest.TestCase):
    def test_fmt_price_with_currency(self):
        self.assertEqual(p._fmt_price(191.0, "USD"), "USD 191.0000")

    def test_fmt_price_no_currency(self):
        self.assertEqual(p._fmt_price(50.5), "50.5000")

    def test_fmt_price_none(self):
        self.assertEqual(p._fmt_price(None), "N/A")

    def test_fmt_dt_iso_with_time(self):
        self.assertEqual(p._fmt_dt("2023-07-10T14:30:00"), "2023-07-10 14:30")

    def test_fmt_dt_iso_with_z(self):
        self.assertEqual(p._fmt_dt("2023-07-10T14:30:00Z"), "2023-07-10 14:30")

    def test_fmt_dt_date_only(self):
        self.assertEqual(p._fmt_dt("2023-07-10"), "2023-07-10 00:00")

    def test_fmt_dt_none(self):
        self.assertEqual(p._fmt_dt(None), "N/A")

    def test_fmt_dt_unrecognised_returned_as_is(self):
        self.assertEqual(p._fmt_dt("not-a-date"), "not-a-date")


# ---------------------------------------------------------------------------
# print_table output
# ---------------------------------------------------------------------------

class TestPrintTable(unittest.TestCase):
    def _capture(self):
        rows, summary = p.build_rows(HOLDINGS, MARKET_DATA)
        buf = StringIO()
        with patch("sys.stdout", buf):
            p.print_table(PORTFOLIO_DOC, rows, summary)
        return buf.getvalue()

    def test_pie_name_shown(self):
        self.assertIn("Global Tech", self._capture())

    def test_company_names_shown(self):
        out = self._capture()
        self.assertIn("Apple Inc.", out)
        self.assertIn("NVIDIA Corporation", out)

    def test_isins_shown(self):
        out = self._capture()
        self.assertIn("US0378331005", out)
        self.assertIn("US5949181045", out)

    def test_entry_date_shown(self):
        self.assertIn("2023-07-10", self._capture())

    def test_current_price_shown(self):
        self.assertIn("191.0000", self._capture())

    def test_entry_price_shown(self):
        self.assertIn("178.5000", self._capture())

    def test_percentage_symbol_shown(self):
        self.assertIn("%", self._capture())

    def test_total_value_shown(self):
        out = self._capture()
        expected = 5 * 191.00 + 3 * 355.00 + 2 * 600.00
        self.assertIn(f"{expected:,.2f}", out)


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    def _run_main(self, portfolio_doc, argv=None):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(portfolio_doc, f)
            fname = f.name

        argv = argv or ["portfolio.py", fname]
        buf = StringIO()
        with patch("yfinance.Ticker", side_effect=_ticker_side_effect), \
             patch("sys.argv", argv), \
             patch("sys.stdout", buf):
            p.main()
        return buf.getvalue()

    def test_full_run(self):
        out = self._run_main(PORTFOLIO_DOC)
        self.assertIn("Global Tech", out)
        self.assertIn("Apple Inc.", out)
        self.assertIn("US67066G1040", out)

    def test_missing_file_exits_1(self):
        with patch("sys.argv", ["portfolio.py", "/no/such/file.json"]):
            with self.assertRaises(SystemExit) as cm:
                p.main()
        self.assertEqual(cm.exception.code, 1)

    def test_default_file_used_when_no_arg(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", dir=".", prefix="portfolio",
            delete=False
        ) as f:
            json.dump(PORTFOLIO_DOC, f)
            fname = f.name

        orig_default = p.DEFAULT_FILE
        try:
            p.DEFAULT_FILE = fname
            buf = StringIO()
            with patch("yfinance.Ticker", side_effect=_ticker_side_effect), \
                 patch("sys.argv", ["portfolio.py"]), \
                 patch("sys.stdout", buf):
                p.main()
            self.assertIn("Global Tech", buf.getvalue())
        finally:
            p.DEFAULT_FILE = orig_default
            Path(fname).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
