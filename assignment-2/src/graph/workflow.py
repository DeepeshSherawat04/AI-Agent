"""
LangGraph workflow definition.
Composes Triage Agent -> Booking Specialist routing graph.
"""

from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.state.conversation_state import ConversationState
from src.agents.triage_agent import TriageAgent
from src.agents.booking_specialist import BookingSpecialist


class SchedulingWorkflow:
    """
    Multi-agent LangGraph workflow orchestrating:
    - Triage Agent (intent classification)
    - Booking Specialist (booking workflow)
    """

    def __init__(self):
        self.triage_agent = TriageAgent()
        self.booking_specialist = BookingSpecialist()
        # BUG FIX: Removed LangGraph internal checkpointer to prevent
        # msgpack deserialization corruption of BookingDetails.
        # Persistence is handled by custom SQLiteCheckpointer in streamlit_app.py.
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph state machine."""

        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("general_response", self._general_response_node)
        workflow.add_node("booking", self._booking_node)

        # Entry point
        workflow.set_entry_point("start")

        # Start router: skip triage if we're already mid-booking
        workflow.add_conditional_edges(
            "start",
            self._route_start,
            {
                "triage": "triage",
                "booking": "booking"
            }
        )

        # Triage → general or booking
        workflow.add_conditional_edges(
            "triage",
            self._route_intent,
            {
                "general": "general_response",
                "booking": "booking",
                "unknown": "general_response"
            }
        )

        # General response ends the turn
        workflow.add_edge("general_response", END)

        # Booking ends when waiting for user input or when complete
        workflow.add_conditional_edges(
            "booking",
            self._route_booking_status,
            {
                "needs_input": END,            # FIXED: end turn, let Streamlit wait
                "complete": END,
                "continue_booking": "booking"   # internal multi-step if needed
            }
        )

        # BUG FIX: Compile WITHOUT checkpointer. The custom SQLiteCheckpointer
        # in streamlit_app.py handles persistence. LangGraph's internal
        # msgpack checkpointing corrupts BookingDetails on deserialization.
        return workflow.compile()

    # ── Nodes ───────────────────────────────────────────────────

    def _start_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        BUG FIX: Preserve existing state instead of returning empty dict.
        Returning {} would wipe fields like suggested_slots and booking_details.
        """
        return {
            "current_agent": state.current_agent,
            "last_action": state.last_action,
            "booking_details": state.booking_details,
            "suggested_slots": state.suggested_slots,
            "booking_confirmed": state.booking_confirmed,
        }

    def _triage_node(self, state: ConversationState) -> Dict[str, Any]:
        """Execute triage agent to classify intent."""
        return self.triage_agent.analyze(state)

    def _general_response_node(self, state: ConversationState) -> Dict[str, Any]:
        """Deliver the general response and end."""
        return {"current_agent": "triage", "last_action": "general_complete"}

    def _booking_node(self, state: ConversationState) -> Dict[str, Any]:
        """Execute booking specialist workflow."""
        return self.booking_specialist.process(state)

    # ── Routers ─────────────────────────────────────────────────

    def _route_start(self, state: ConversationState) -> Literal["triage", "booking"]:
        """
        BUG FIX: More robust routing that preserves booking context.
        Route to booking if:
        1. current_agent is already "booking", OR
        2. We have partial booking details (date/time/email/service), OR
        3. We have suggested_slots from a previous negotiation
        """
        if state.current_agent == "booking":
            return "booking"

        # If we have any booking details or suggested slots, we're mid-booking
        if (state.booking_details.date or
            state.booking_details.time or
            state.booking_details.email or
            state.booking_details.service or
            state.suggested_slots):
            return "booking"

        return "triage"

    def _route_intent(self, state: ConversationState) -> Literal["general", "booking", "unknown"]:
        """Route based on triage classification."""
        intent = state.intent
        if intent == "general":
            return "general"
        elif intent == "booking":
            return "booking"
        return "unknown"

    def _route_booking_status(self, state: ConversationState) -> Literal["needs_input", "complete", "continue_booking"]:
        """Route based on booking specialist status."""
        if state.booking_confirmed:
            return "complete"

        # If we're waiting for the user to reply, END this turn
        if state.last_action in (
            "awaiting_confirmation", "ask_date", "ask_time",
            "ask_email", "ask_service", "suggested_alternatives",
            "suggested_times", "awaiting_human_input", "responded"
        ):
            return "needs_input"

        # Internal processing steps that don't need user input yet
        if state.last_action in ("check_availability", "validate_details"):
            return "continue_booking"

        return "needs_input"

    def invoke(self, state: ConversationState, thread_id: str = "default") -> ConversationState:
        """
        BUG FIX: Pass thread_id in config. Persistence is handled by
        custom SQLiteCheckpointer in streamlit_app.py, not LangGraph's
        internal checkpointing (which corrupts BookingDetails via msgpack).
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state, config=config)
        return result

    def stream(self, state: ConversationState, thread_id: str = "default"):
        """
        BUG FIX: Pass thread_id in config for streaming.
        """
        config = {"configurable": {"thread_id": thread_id}}
        for update in self.graph.stream(state, config=config):
            yield update