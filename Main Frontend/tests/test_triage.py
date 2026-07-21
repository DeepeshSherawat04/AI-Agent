"""
Unit tests for Triage Agent routing logic.
"""

import pytest
from langchain_core.messages import HumanMessage
from src.agents.triage_agent import TriageAgent
from src.state.conversation_state import ConversationState


class TestTriageAgent:
    """Test intent classification and routing."""

    @pytest.fixture
    def agent(self):
        return TriageAgent()

    def test_booking_intent(self, agent):
        state = ConversationState(
            messages=[HumanMessage(content="I want to book an appointment for tomorrow")]
        )
        result = agent.analyze(state)

        assert result["intent"] == "booking"
        assert result["current_agent"] == "booking"

    def test_general_intent(self, agent):
        state = ConversationState(
            messages=[HumanMessage(content="What services do you offer?")]
        )
        result = agent.analyze(state)

        assert result["intent"] == "general"
        assert result["current_agent"] == "triage"

    def test_greeting_intent(self, agent):
        state = ConversationState(
            messages=[HumanMessage(content="Hello there")]
        )
        result = agent.analyze(state)

        assert result["intent"] == "general"

    def test_extracts_date(self, agent):
        state = ConversationState(
            messages=[HumanMessage(content="Book for 2026-07-15 at 2 PM")]
        )
        result = agent.analyze(state)

        assert result["intent"] == "booking"
        # Should have extracted date
        assert result["booking_details"].date is not None or result["missing_info"] == []
