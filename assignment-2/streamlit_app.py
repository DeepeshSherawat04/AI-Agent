import os
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

"""
Main Streamlit entry point for the Multi-Agent Scheduling Assistant.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.state.conversation_state import ConversationState, BookingDetails
from src.graph.workflow import SchedulingWorkflow
from src.utils.config import get_config
from src.utils.database import get_db
from src.utils.persistence import SQLiteCheckpointer


# ── Workflow Helper ─────────────────────────────────────────
def _run_workflow(current_state: ConversationState) -> None:
    """Execute the scheduling workflow and update session state."""
    with st.spinner("Thinking..."):
        try:
            workflow = st.session_state.workflow
            new_state = workflow.invoke(current_state, thread_id=st.session_state.thread_id)

            # Defensive: ensure new_state is ConversationState, not dict
            if isinstance(new_state, dict):
                bd_raw = new_state.get("booking_details", {})
                if isinstance(bd_raw, BookingDetails):
                    bd = bd_raw
                else:
                    bd = BookingDetails(
                        date=bd_raw.get("date") if isinstance(bd_raw, dict) else None,
                        time=bd_raw.get("time") if isinstance(bd_raw, dict) else None,
                        email=bd_raw.get("email") if isinstance(bd_raw, dict) else None,
                        service=bd_raw.get("service") if isinstance(bd_raw, dict) else None,
                        confirmed=bd_raw.get("confirmed", False) if isinstance(bd_raw, dict) else False
                    )
                new_state = ConversationState(
                    messages=new_state.get("messages", []),
                    intent=new_state.get("intent", "unknown"),
                    booking_details=bd,
                    missing_info=new_state.get("missing_info", []),
                    current_agent=new_state.get("current_agent", "triage"),
                    last_action=new_state.get("last_action", ""),
                    error=new_state.get("error"),
                    suggested_slots=new_state.get("suggested_slots", []),
                    booking_confirmed=new_state.get("booking_confirmed", False),
                    thread_id=new_state.get("thread_id", st.session_state.thread_id)
                )

            st.session_state.state = new_state

            # Add new AI messages
            for msg in new_state.messages:
                if isinstance(msg, AIMessage) and msg not in st.session_state.messages:
                    st.session_state.messages.append(msg)

            # Track confirmed bookings
            if new_state.booking_confirmed and new_state.last_action == "booking_complete":
                st.session_state.booking_history.append({
                    "date": new_state.booking_details.date,
                    "time": new_state.booking_details.time,
                    "email": new_state.booking_details.email,
                    "timestamp": datetime.now().isoformat()
                })

                st.session_state.state.booking_details = BookingDetails()
                st.session_state.state.booking_confirmed = False
                st.session_state.state.last_action = "booking_complete"
                st.session_state.state.suggested_slots = []
                st.session_state.state.current_agent = "triage"

            # Persist to SQLite
            try:
                checkpointer = SQLiteCheckpointer()
                checkpointer.save(
                    st.session_state.thread_id,
                    st.session_state.messages,
                    new_state.model_dump()
                )
            except Exception as persist_err:
                if config.debug:
                    st.warning(f"Persistence warning: {persist_err}")

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))


# ── Page Config ───────────────────────────────────────────────
config = get_config()
st.set_page_config(
    page_title=config.app_title,
    page_icon=config.app_icon,
    layout="centered",
    initial_sidebar_state="expanded"
)


# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 2rem; }
    .agent-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }
    .badge-triage { background: #e3f2fd; color: #1565c0; }
    .badge-booking { background: #e8f5e9; color: #2e7d32; }
    .status-box { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .status-success { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .status-info { background: #e3f2fd; border-left: 4px solid #2196f3; }
    .status-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
</style>
""", unsafe_allow_html=True)


# ── CRITICAL: State safety helper ─────────────────────────────
def _ensure_state() -> ConversationState:
    """Guarantee st.session_state.state is a ConversationState, never a dict."""
    state = st.session_state.get("state")

    if isinstance(state, ConversationState):
        return state

    if isinstance(state, dict):
        bd_raw = state.get("booking_details", {})
        if isinstance(bd_raw, BookingDetails):
            bd = bd_raw
        else:
            bd = BookingDetails(
                date=bd_raw.get("date") if isinstance(bd_raw, dict) else None,
                time=bd_raw.get("time") if isinstance(bd_raw, dict) else None,
                email=bd_raw.get("email") if isinstance(bd_raw, dict) else None,
                service=bd_raw.get("service") if isinstance(bd_raw, dict) else None,
                confirmed=bd_raw.get("confirmed", False) if isinstance(bd_raw, dict) else False
            )
        st.session_state.state = ConversationState(
            messages=state.get("messages", []),
            intent=state.get("intent", "unknown"),
            booking_details=bd,
            missing_info=state.get("missing_info", []),
            current_agent=state.get("current_agent", "triage"),
            last_action=state.get("last_action", ""),
            error=state.get("error"),
            suggested_slots=state.get("suggested_slots", []),
            booking_confirmed=state.get("booking_confirmed", False),
            thread_id=state.get("thread_id", st.session_state.get("thread_id", "default"))
        )
        return st.session_state.state

    # state is None or something else — create fresh
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4())[:8])
    st.session_state.state = ConversationState(thread_id=thread_id)
    return st.session_state.state


# ── Init Session State ────────────────────────────────────────
def init_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content=(
                "👋 Welcome to the **TrulyIAS Scheduling Assistant**!\n\n"
                "I can help you with:\n"
                "• 📅 Booking appointments\n"
                "• 🔍 Checking availability\n"
                "• ❓ General questions about our services\n\n"
                "How can I assist you today?"
            ))
        ]

    if "workflow" not in st.session_state:
        try:
            st.session_state.workflow = SchedulingWorkflow()
        except Exception as e:
            st.error(f"Failed to initialize workflow: {e}")
            st.session_state.workflow = None

    # CRITICAL: Always ensure state is a ConversationState before anything else
    if "state" not in st.session_state or not isinstance(st.session_state.state, ConversationState):
        # Try to load from SQLite first
        loaded = False
        try:
            checkpointer = SQLiteCheckpointer()
            saved = checkpointer.load(st.session_state.thread_id)
            if saved:
                if saved.get("messages"):
                    st.session_state.messages = saved["messages"]
                bd_raw = saved.get("booking_details", {})
                if isinstance(bd_raw, BookingDetails):
                    bd = bd_raw
                else:
                    bd = BookingDetails(
                        date=bd_raw.get("date") if isinstance(bd_raw, dict) else None,
                        time=bd_raw.get("time") if isinstance(bd_raw, dict) else None,
                        email=bd_raw.get("email") if isinstance(bd_raw, dict) else None,
                        service=bd_raw.get("service") if isinstance(bd_raw, dict) else None
                    )
                st.session_state.state = ConversationState(
                    messages=st.session_state.messages.copy(),
                    current_agent=saved.get("current_agent", "triage"),
                    last_action=saved.get("last_action", ""),
                    booking_details=bd,
                    suggested_slots=saved.get("suggested_slots", []),  # BUG FIX: Load suggested_slots from SQLite
                    thread_id=st.session_state.thread_id
                )
                loaded = True
        except Exception:
            pass

        if not loaded:
            st.session_state.state = ConversationState(
                thread_id=st.session_state.thread_id
            )

    if "booking_history" not in st.session_state:
        st.session_state.booking_history = []

    if "restored" not in st.session_state:
        st.session_state.restored = False


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div class='main-header'>{config.app_icon} {config.app_title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>AI-Powered Multi-Agent Scheduling</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("🔧 Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")

    if st.session_state.get("restored"):
        st.success("✅ State restored from SQLite")

    # ALWAYS use _ensure_state() for safe access
    safe_state = _ensure_state()
    current_agent = safe_state.current_agent
    badge_class = f"badge-{current_agent}"
    st.markdown(
        f"Active Agent: <span class='agent-badge {badge_class}'>{current_agent.upper()}</span>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📋 Booking Status")
    booking = safe_state.booking_details

    if booking.date or booking.time or booking.email:
        status_items = []
        if booking.date:
            status_items.append(f"📅 Date: {booking.date}")
        if booking.time:
            status_items.append(f"⏰ Time: {booking.time}")
        if booking.email:
            status_items.append(f"📧 Email: {booking.email}")
        if booking.service:
            status_items.append(f"📋 Service: {booking.service}")

        for item in status_items:
            st.markdown(f"<div class='status-box status-info'>{item}</div>", unsafe_allow_html=True)

        if booking.is_complete():
            st.markdown("<div class='status-box status-success'>✅ All details collected!</div>", unsafe_allow_html=True)
        else:
            missing = booking.missing_fields()
            st.markdown(
                f"<div class='status-box status-warning'>⏳ Missing: {', '.join(missing)}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No active booking in progress.")

    st.divider()

    st.subheader("⚡ Quick Actions")

    if st.button("🆕 New Conversation", use_container_width=True):
        try:
            old_thread = st.session_state.thread_id
            checkpointer = SQLiteCheckpointer()
            checkpointer.delete(old_thread)
        except Exception:
            pass

        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = [
            AIMessage(content="👋 New conversation started! How can I help you today?")
        ]
        st.session_state.state = ConversationState(thread_id=st.session_state.thread_id)
        st.session_state.restored = False
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            AIMessage(content="Chat cleared. How can I help you?")
        ]
        st.session_state.state = ConversationState(thread_id=st.session_state.thread_id)
        st.session_state.restored = False
        st.rerun()

    st.divider()

    st.subheader("📚 Recent Bookings")
    try:
        db = get_db()
        appointments = db.get_appointments()
        if appointments:
            for appt in appointments[-5:]:
                with st.container():
                    st.markdown(f"**{appt['date']}** at {appt['time']}")
                    st.caption(f"{appt['email']} | {appt['service']}")
        else:
            st.caption("No bookings yet.")
    except Exception:
        st.caption("Database not initialized. Run: python scripts/setup_db.py")

    st.divider()
    st.caption("Built with LangGraph + Streamlit")
    st.caption("© 2026 TrulyIAS Internship")


# ── Main Chat Interface ─────────────────────────────────────
st.markdown(f"<div class='main-header'>{config.app_icon} {config.app_title}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Multi-Agent Scheduling Assistant with Persistent Memory</div>", unsafe_allow_html=True)

st.divider()

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)
            agent = _ensure_state().current_agent
            badge_class = f"badge-{agent}"
            st.markdown(
                f"<span class='agent-badge {badge_class}'>{agent.upper()} AGENT</span>",
                unsafe_allow_html=True
            )
            


# ── Service Selection Dropdown ────────────────────────────────
# Show interactive service buttons when agent asks for service type
safe_state = _ensure_state()
last_ai_msg = ""
for msg in reversed(safe_state.messages):
    if isinstance(msg, AIMessage):
        last_ai_msg = msg.content.lower()
        break

agent_asking_service = (
    "service" in last_ai_msg and 
    any(phrase in last_ai_msg for phrase in [
        "what type of service", "what service", "which service",
        "service you'd like", "service you want", "service are you looking for",
        "service you're interested in", "service you need", "service would you like",
        "what kind of service", "service to book", "service for your",
        "interested in booking"
    ])
)

if agent_asking_service and not safe_state.booking_details.service and st.session_state.workflow is not None:
    st.markdown("---")
    st.markdown("**📋 Select a service type:**")

    col1, col2 = st.columns(2)
    service_options = {
        "🤝 Meeting": "Meeting",
        "🎤 Interview": "Interview", 
        "💡 Consultation": "Consultation",
        "🖥️ Demo": "Demo",
        "📊 Review": "Review",
        "🎯 Coaching": "Coaching"
    }

    for idx, (label, value) in enumerate(service_options.items()):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(label, key=f"svc_{value}", use_container_width=True):
                safe_state.booking_details.service = value
                human_msg = HumanMessage(content=value)
                st.session_state.messages.append(human_msg)
                safe_state.messages.append(human_msg)
                _run_workflow(safe_state)
                st.rerun()

    st.markdown("---")
    # NOTE: We do NOT call st.stop() here. The chat input remains visible below
    # the dropdown buttons, so the user can either click a button or type.
    # This avoids the scroll-to-top issue caused by st.stop() halting execution.

# ── Chat Input ────────────────────────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input and st.session_state.workflow is not None:
    # ALWAYS use _ensure_state() before processing
    current_state = _ensure_state()

    # Add user message
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    current_state.messages.append(human_msg)

    # ═══════════════════════════════════════════════════════════
    # BUG FIX: Minimal extraction - only update what's explicitly
    # provided, NEVER wipe existing booking details.
    # ═══════════════════════════════════════════════════════════
    from src.utils.date_parser import resolve_relative_date, normalize_time

    text_lower = user_input.lower()

    # Only extract date/time if explicitly present in the message
    date_extracted = resolve_relative_date(user_input)
    time_extracted = normalize_time(user_input)

    # BUG FIX: Merge extracted values with existing state, never wipe
    if date_extracted:
        current_state.booking_details.date = date_extracted

    if time_extracted:
        current_state.booking_details.time = time_extracted

    # Extract email only if not already collected
    import re
    email_match = re.search(r"[\w.-]+@[\w.-]+\.\w+", user_input)
    if email_match and not current_state.booking_details.email:
        current_state.booking_details.email = email_match.group(0)

    # Extract service only if not already collected
        # Fallback: extract service from text input if not already collected
    if not current_state.booking_details.service:
        services = ["consultation", "review", "meeting", "interview", "demo", 
                    "general", "coaching", "training", "session", "call", 
                    "class", "webinar", "workshop"]
        for svc in services:
            if svc in text_lower:
                current_state.booking_details.service = svc.capitalize()
                break

    # Handle confirmation/cancellation
    confirm_words = ["yes", "yeah", "sure", "confirm", "go ahead", "ok", "okay", "proceed"]
    if any(w in text_lower for w in confirm_words):
        if current_state.last_action == "awaiting_confirmation":
            pass  # Let the agent handle confirmation

    cancel_words = ["no", "nope", "cancel", "stop", "nevermind", "abort"]
    if any(w in text_lower for w in cancel_words):
        if current_state.last_action in ("awaiting_confirmation", "suggested_alternatives"):
            # FIX: Preserve date/email/service, only clear time to re-negotiate
            current_state.booking_details.time = None
            current_state.last_action = "cancelled"
            current_state.booking_confirmed = False
            # Keep current_agent as "booking" so we stay in the booking flow

    # Run the workflow
    _run_workflow(current_state)
    st.rerun()


# NEW: Tip in a container that doesn't interfere with clicks
st.divider()
with st.container():
    st.markdown(
        "<div style='color: #888; font-size: 0.85rem; padding: 0.5rem 0; user-select: text;'>"
        "💡 <strong>Tip:</strong> Try saying <em>'Book an appointment for tomorrow at 3 PM'</em> "
        "or <em>'What slots are available next Monday?'</em>"
        "</div>",
        unsafe_allow_html=True
    )