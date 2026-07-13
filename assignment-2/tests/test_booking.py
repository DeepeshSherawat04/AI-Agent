"""
Unit tests for Booking Specialist validation & negotiation.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.booking_specialist import BookingSpecialist
from src.state.conversation_state import ConversationState, BookingDetails


class TestBookingSpecialist:
    """Test booking workflow logic."""

    @pytest.fixture
    def specialist(self):
        return BookingSpecialist()

    def test_asks_for_missing_date(self, specialist):
        state = ConversationState(
            booking_details=BookingDetails(time="14:00", email="test@example.com"),
            current_agent="booking"
        )
        result = specialist.process(state)

        assert result["last_action"] == "ask_date"
        assert "date" in result["messages"][0].content.lower()

    def test_asks_for_missing_time(self, specialist):
        state = ConversationState(
            booking_details=BookingDetails(date="2026-12-25", email="test@example.com"),
            current_agent="booking"
        )
        result = specialist.process(state)

        assert result["last_action"] == "ask_time"
        assert "time" in result["messages"][0].content.lower()

    def test_asks_for_missing_email(self, specialist):
        state = ConversationState(
            booking_details=BookingDetails(date="2026-12-25", time="14:00"),
            current_agent="booking"
        )
        result = specialist.process(state)

        assert result["last_action"] == "ask_email"
        assert "email" in result["messages"][0].content.lower()

    def test_normalizes_relative_date(self, specialist):
        state = ConversationState(
            booking_details=BookingDetails(date="tomorrow", time="14:00", email="test@example.com"),
            current_agent="booking",
            last_action="check_availability"
        )
        result = specialist.process(state)

        # Date should have been normalized
        assert result["booking_details"].date != "tomorrow"
        assert "-" in result["booking_details"].date

    def test_suggests_alternatives_when_unavailable(self, specialist):
        # Use a date far in the past that should have no slots
        state = ConversationState(
            booking_details=BookingDetails(date="2020-01-01", time="14:00", email="test@example.com"),
            current_agent="booking",
            last_action="check_availability"
        )
        result = specialist.process(state)

        assert result["last_action"] == "suggested_alternatives"
        assert len(result["suggested_slots"]) > 0
