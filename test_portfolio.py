"""Tests for portfolio.py using mocked HTTP responses."""

import json
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import requests

import portfolio as p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PIES_RESPONSE = [
    {
        "settings": {
            "id": 42,
            "name": "Global Tech",
            "createdAt": "2023-06-15T09:00:00Z",
            "dividendCashAction": "REINVEST",
        }
    },
    {
        "settings": {
            "id": 99,
            "name": "Dividend Kings",
            "createdAt": "2024-01-01T00:00:00Z",
            "dividendCashAction": "FREE_CASH",
        }
    },
]

PIE_DETAIL = {
    "settings": {
        "id": 42,
        "name": "Global Tech",
        "createdAt": "2023-06-15T09:00:00Z",
        "dividendCashAction": "REINVEST",
    },
    "instruments": [
        {
            "ticker": "AAPL_US_EQ",
            "result": {
                "priceAvgBuy": 178.50,
                "quantity": 5.0,
                "result": 62.50,
                "resultCoefficient": 0.07,
            },
        },
        {
            "ticker": "MSFT_US_EQ",
            "result": {
                "priceAvgBuy": 320.00,
                "quantity": 3.0,
                "result": 105.00,
                "resultCoefficient": 0.11,
            },
        },
        {
            "ticker": "NVDA_US_EQ",
            "result": {
                "priceAvgBuy": 450.00,
                "quantity": 2.0,
                "result": 300.00,
                "resultCoefficient": 0.33,
            },
        },
    ],
}

PORTFOLIO_RESPONSE = [
    {
        "ticker": "AAPL_US_EQ",
        "averagePricePaid": 178.50,
        "currentPrice": 191.00,
        "createdAt": "2023-07-10T14:30:00Z",
        "quantity": 5.0,
    },
    {
        "ticker": "MSFT_US_EQ",
        "averagePricePaid": 320.00,
        "currentPrice": 355.00,
        "createdAt": "2023-07-10T14:31:00Z",
        "quantity": 3.0,
    },
    {
        "ticker": "NVDA_US_EQ",
        "averagePricePaid": 450.00,
        "currentPrice": 600.00,
        "createdAt": "2023-08-01T10:00:00Z",
        "quantity": 2.0,
    },
]

INSTRUMENTS_RESPONSE = [
    {"ticker": "AAPL_US_EQ", "name": "Apple Inc.", "shortName": "AAPL", "isin": "US0378331005", "currencyCode": "USD"},
    {"ticker": "MSFT_US_EQ", "name": "Microsoft Corporation", "shortName": "MSFT", "isin": "US5949181045", "currencyCode": "USD"},
    {"ticker": "NVDA_US_EQ", "name": "NVIDIA Corporation", "shortName": "NVDA", "isin": "US67066G1040", "currencyCode": "USD"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(data, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _make_session(pies=None, pie=None, portfolio=None, instruments=None):
    """Return a mock session whose get() returns correct data per URL path."""
    session = MagicMock(spec=requests.Session)

    def side_effect(url, **kwargs):
        if "/equity/pies" in url and url.endswith("/pies"):
            return _mock_response(pies or PIES_RESPONSE)
        if "/equity/pies/" in url:
            return _mock_response(pie or PIE_DETAIL)
        if "/equity/portfolio" in url:
            return _mock_response(portfolio or PORTFOLIO_RESPONSE)
        if "/equity/metadata/instruments" in url:
            return _mock_response(instruments or INSTRUMENTS_RESPONSE)
        raise ValueError(f"Unexpected URL: {url}")

    session.get.side_effect = side_effect
    return session


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFetchInstruments(unittest.TestCase):
    def test_list_response(self):
        session = _make_session()
        result = p.fetch_instruments(session, "http://mock")
        self.assertIn("AAPL_US_EQ", result)
        self.assertEqual(result["AAPL_US_EQ"]["isin"], "US0378331005")
        self.assertEqual(result["AAPL_US_EQ"]["name"], "Apple Inc.")
        self.assertEqual(result["MSFT_US_EQ"]["currency"], "USD")

    def test_paginated_response(self):
        page1 = {"items": INSTRUMENTS_RESPONSE[:2], "nextPagePath": "/equity/metadata/instruments?page=2"}
        page2 = {"items": INSTRUMENTS_RESPONSE[2:], "nextPagePath": None}

        session = MagicMock(spec=requests.Session)
        calls = [_mock_response(page1), _mock_response(page2)]
        session.get.side_effect = lambda url, **kw: calls.pop(0)

        result = p.fetch_instruments(session, "http://mock")
        self.assertEqual(len(result), 3)
        self.assertIn("NVDA_US_EQ", result)


class TestFetchPortfolio(unittest.TestCase):
    def test_keyed_by_ticker(self):
        session = _make_session()
        result = p.fetch_portfolio(session, "http://mock")
        self.assertIn("AAPL_US_EQ", result)
        self.assertEqual(result["AAPL_US_EQ"]["currentPrice"], 191.00)
        self.assertEqual(result["MSFT_US_EQ"]["createdAt"], "2023-07-10T14:31:00Z")

    def test_missing_ticker_skipped(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response([{"averagePricePaid": 100}, {"ticker": "X_EQ", "currentPrice": 10}])
        result = p.fetch_portfolio(session, "http://mock")
        self.assertEqual(list(result.keys()), ["X_EQ"])


class TestFindPie(unittest.TestCase):
    def test_by_id(self):
        pie = p.find_pie(PIES_RESPONSE, "42")
        self.assertIsNotNone(pie)
        self.assertEqual(p._pie_settings(pie)["name"], "Global Tech")

    def test_by_exact_name(self):
        pie = p.find_pie(PIES_RESPONSE, "Dividend Kings")
        self.assertIsNotNone(pie)
        self.assertEqual(p._pie_settings(pie)["id"], 99)

    def test_by_partial_name_case_insensitive(self):
        pie = p.find_pie(PIES_RESPONSE, "global")
        self.assertIsNotNone(pie)
        self.assertEqual(p._pie_settings(pie)["id"], 42)

    def test_not_found(self):
        self.assertIsNone(p.find_pie(PIES_RESPONSE, "XYZ Unknown"))


class TestBuildRows(unittest.TestCase):
    def _run(self):
        instruments = {item["ticker"]: {
            "name": item["name"], "shortName": item["shortName"],
            "isin": item["isin"], "currency": item["currencyCode"],
        } for item in INSTRUMENTS_RESPONSE}
        portfolio = {pos["ticker"]: pos for pos in PORTFOLIO_RESPONSE}
        return p.build_rows(PIE_DETAIL, instruments, portfolio)

    def test_row_count(self):
        rows, summary = self._run()
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["count"], 3)

    def test_sorted_by_value_descending(self):
        rows, _ = self._run()
        values = [r["value"] for r in rows]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_total_value(self):
        rows, summary = self._run()
        expected = 5 * 191.00 + 3 * 355.00 + 2 * 600.00
        self.assertAlmostEqual(summary["total_value"], expected, places=2)

    def test_percentages_sum_to_100(self):
        rows, summary = self._run()
        total = summary["total_value"]
        pct_sum = sum(r["value"] / total * 100 for r in rows)
        self.assertAlmostEqual(pct_sum, 100.0, places=5)

    def test_isin_present(self):
        rows, _ = self._run()
        isins = {r["isin"] for r in rows}
        self.assertIn("US0378331005", isins)
        self.assertIn("US5949181045", isins)

    def test_current_price_from_portfolio(self):
        rows, _ = self._run()
        aapl = next(r for r in rows if r["ticker"] == "AAPL_US_EQ")
        self.assertEqual(aapl["current_price"], 191.00)

    def test_entry_price_from_pie_result(self):
        rows, _ = self._run()
        aapl = next(r for r in rows if r["ticker"] == "AAPL_US_EQ")
        self.assertEqual(aapl["entry_price"], 178.50)

    def test_entry_time_from_portfolio(self):
        rows, _ = self._run()
        aapl = next(r for r in rows if r["ticker"] == "AAPL_US_EQ")
        self.assertEqual(aapl["entry_time"], "2023-07-10T14:30:00Z")


class TestPrintPortfolio(unittest.TestCase):
    def test_output_contains_key_data(self):
        instruments = {item["ticker"]: {
            "name": item["name"], "shortName": item["shortName"],
            "isin": item["isin"], "currency": item["currencyCode"],
        } for item in INSTRUMENTS_RESPONSE}
        portfolio = {pos["ticker"]: pos for pos in PORTFOLIO_RESPONSE}

        captured = StringIO()
        with patch("sys.stdout", captured):
            p.print_portfolio(PIE_DETAIL, instruments, portfolio)

        output = captured.getvalue()
        self.assertIn("Global Tech", output)
        self.assertIn("Apple Inc.", output)
        self.assertIn("US0378331005", output)   # AAPL ISIN
        self.assertIn("MSFT_US_EQ", output)
        self.assertIn("2023-07-10", output)      # entry date
        self.assertIn("191.0000", output)        # AAPL current price
        self.assertIn("178.5000", output)        # AAPL entry price
        self.assertIn("%", output)


class TestFormatHelpers(unittest.TestCase):
    def test_fmt_price_with_currency(self):
        self.assertEqual(p._fmt_price(191.0, "USD"), "USD 191.0000")

    def test_fmt_price_none(self):
        self.assertEqual(p._fmt_price(None), "N/A")

    def test_fmt_dt_iso(self):
        self.assertEqual(p._fmt_dt("2023-07-10T14:30:00Z"), "2023-07-10 14:30")

    def test_fmt_dt_none(self):
        self.assertEqual(p._fmt_dt(None), "N/A")

    def test_fmt_dt_invalid(self):
        self.assertEqual(p._fmt_dt("not-a-date"), "not-a-date")


class TestNoApiKey(unittest.TestCase):
    def test_exits_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove T212_API_KEY if set
            env = {k: v for k, v in __import__("os").environ.items() if k != "T212_API_KEY"}
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(SystemExit) as cm:
                    with patch("sys.argv", ["portfolio.py"]):
                        p.main()
                self.assertEqual(cm.exception.code, 1)


class TestListPies(unittest.TestCase):
    def test_lists_pies_when_no_query(self):
        session = _make_session()
        captured = StringIO()
        with patch.object(p, "_session", return_value=session), \
             patch.dict("os.environ", {"T212_API_KEY": "test-key", "T212_ENV": "demo"}), \
             patch("sys.argv", ["portfolio.py"]), \
             patch("sys.stdout", captured):
            with self.assertRaises(SystemExit) as cm:
                p.main()
            self.assertEqual(cm.exception.code, 0)

        output = captured.getvalue()
        self.assertIn("Global Tech", output)
        self.assertIn("Dividend Kings", output)
        self.assertIn("42", output)


class TestMainIntegration(unittest.TestCase):
    def test_full_run_by_name(self):
        session = _make_session()
        captured = StringIO()
        with patch.object(p, "_session", return_value=session), \
             patch.dict("os.environ", {"T212_API_KEY": "test-key", "T212_ENV": "demo"}), \
             patch("sys.argv", ["portfolio.py", "Global Tech"]), \
             patch("sys.stdout", captured):
            p.main()

        output = captured.getvalue()
        self.assertIn("Global Tech", output)
        self.assertIn("NVIDIA Corporation", output)
        self.assertIn("US67066G1040", output)

    def test_full_run_by_id(self):
        session = _make_session()
        captured = StringIO()
        with patch.object(p, "_session", return_value=session), \
             patch.dict("os.environ", {"T212_API_KEY": "test-key", "T212_ENV": "demo"}), \
             patch("sys.argv", ["portfolio.py", "42"]), \
             patch("sys.stdout", captured):
            p.main()

        output = captured.getvalue()
        self.assertIn("Global Tech", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
