"""
Unit tests for calendar tool functions.
"""

import pytest
from src.tools.calendar_tools import check_availability, reserve_slot, get_alternative_slots
from src.utils.database import DatabaseManager
from src.utils.config import get_config


class TestCheckAvailability:
    """Test availability checking."""

    def test_check_existing_date(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        result = check_availability.invoke({"date": today})

        assert "available" in result
        assert "slots" in result
        assert isinstance(result["available"], bool)

    def test_check_invalid_date(self):
        result = check_availability.invoke({"date": "2020-01-01"})
        assert result["available"] is False


class TestReserveSlot:
    """Test slot reservation."""

    def test_reserve_and_check_unavailable(self):
        # First find an available slot
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")

        avail = check_availability.invoke({"date": today})
        open_slots = avail.get("open_slots", [])

        if open_slots:
            time = open_slots[0]
            result = reserve_slot.invoke({
                "date": today,
                "time": time,
                "email": "test@example.com",
                "service": "Test"
            })

            assert result["success"] is True
            assert "appointment_id" in result

            # Verify slot is now unavailable
            avail2 = check_availability.invoke({"date": today})
            assert time not in avail2.get("open_slots", [])


class TestAlternativeSlots:
    """Test alternative slot retrieval."""

    def test_get_alternatives(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        result = get_alternative_slots.invoke({"date": today, "limit": 3})

        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)
