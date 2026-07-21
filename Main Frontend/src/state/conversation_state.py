"""
Pydantic models for LangGraph state management.
Defines thread state, appointment details, and validation status.
"""

from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class BookingDetails(BaseModel):
    """Structured booking information extracted from user messages."""

    date: Optional[str] = Field(None, description="Appointment date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Appointment time in HH:MM format (24h)")
    email: Optional[str] = Field(None, description="User email for confirmation")
    service: Optional[str] = Field(None, description="Type of appointment/service")
    confirmed: bool = Field(False, description="Whether booking is finalized")

    def missing_fields(self) -> List[str]:
        """Return list of fields that are still None."""
        missing = []
        if not self.date:
            missing.append("date")
        if not self.time:
            missing.append("time")
        if not self.email:
            missing.append("email")
        if not self.service:
            missing.append("service")
        return missing

    def is_complete(self) -> bool:
        """Check if all required booking fields are filled."""
        # BUG FIX: Include 'service' so the agent doesn't skip asking for it
        return all([self.date, self.time, self.email, self.service])


class ConversationState(BaseModel):
    """
    LangGraph state model for the multi-agent scheduling assistant.
    Persisted across conversation turns via SQLite Saver.
    """

    messages: Annotated[List[Any], add_messages] = Field(
        default_factory=list,
        description="Chat message history (HumanMessage, AIMessage, ToolMessage)"
    )

    intent: str = Field(
        default="unknown",
        description="Detected intent: 'general', 'booking', or 'unknown'"
    )

    booking_details: BookingDetails = Field(
        default_factory=BookingDetails,
        description="Extracted booking information"
    )

    missing_info: List[str] = Field(
        default_factory=list,
        description="Fields still needed from user"
    )

    current_agent: str = Field(
        default="triage",
        description="Currently active agent: 'triage' or 'booking'"
    )

    last_action: str = Field(
        default="",
        description="Last action taken by the system"
    )

    error: Optional[str] = Field(
        None,
        description="Error message if something went wrong"
    )

    suggested_slots: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Alternative slots offered when requested slot is unavailable"
    )

    booking_confirmed: bool = Field(
        False,
        description="True when reservation + notification are complete"
    )

    thread_id: str = Field(
        default="default",
        description="Unique conversation thread identifier"
    )

    class Config:
        arbitrary_types_allowed = True