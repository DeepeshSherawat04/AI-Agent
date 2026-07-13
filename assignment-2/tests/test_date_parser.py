"""
Unit tests for relative date resolution.
"""

import pytest
from datetime import datetime, timedelta
from src.utils.date_parser import resolve_relative_date, normalize_time, extract_date_and_time


class TestResolveRelativeDate:
    """Test cases for date resolution from natural language."""

    def test_today(self):
        result = resolve_relative_date("today")
        expected = datetime.now().strftime("%Y-%m-%d")
        assert result == expected

    def test_tomorrow(self):
        result = resolve_relative_date("tomorrow")
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_day_after_tomorrow(self):
        result = resolve_relative_date("day after tomorrow")
        expected = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        assert result == expected

    def test_in_x_days(self):
        result = resolve_relative_date("in 5 days")
        expected = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        assert result == expected

    def test_iso_date_passthrough(self):
        result = resolve_relative_date("2026-07-15")
        assert result == "2026-07-15"

    def test_invalid_date(self):
        result = resolve_relative_date("not a date")
        assert result is None


class TestNormalizeTime:
    """Test cases for time normalization."""

    def test_noon(self):
        assert normalize_time("noon") == "12:00"

    def test_3pm(self):
        assert normalize_time("3 PM") == "15:00"

    def test_930am(self):
        assert normalize_time("9:30 am") == "09:30"

    def test_24h_format(self):
        assert normalize_time("14:30") == "14:30"

    def test_morning(self):
        assert normalize_time("morning") == "09:00"

    def test_invalid_time(self):
        assert normalize_time("not a time") is None


class TestExtractDateAndTime:
    """Test combined extraction."""

    def test_extract_both(self):
        date, time = extract_date_and_time("Book for 2026-07-15 at 3 PM")
        assert date == "2026-07-15"
        assert time == "15:00"
