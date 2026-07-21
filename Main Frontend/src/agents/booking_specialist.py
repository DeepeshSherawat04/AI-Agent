"""
Booking Specialist: Manages calendar tool execution,
tracks slot details, and prompts for missing information.
Uses LLM for natural language generation with deterministic flow control.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from src.utils.config import get_config
from src.utils.date_parser import resolve_relative_date, normalize_time
from src.state.conversation_state import ConversationState, BookingDetails
from src.tools.calendar_tools import check_availability, reserve_slot, send_booking_notification, get_alternative_slots


BOOKING_SYSTEM_PROMPT = """You are the Booking Specialist for TrulyIAS Scheduling Assistant.

Your job is to help users book appointments. You will be given the current state of a booking and a specific instruction. Generate a friendly, professional, concise response.

## Rules:
- If asking for missing info, ask for ONLY ONE item at a time. Be conversational.
- When confirming a date, state the normalized date clearly.
- If suggesting alternatives, list them cleanly with bullet points.
- After booking, summarize all details and confirm.
- Be polite, efficient, and warm. Do NOT ask for information that is already provided.
- Do NOT assume a service type. If the user hasn't specified one, ask them.
- Use the EXACT friendly date string provided in the context. Do NOT guess the day of the week.
"""


class BookingSpecialist:
    """Handles booking workflow: validation, availability check, reservation, notification."""

    def __init__(self):
        config = get_config()
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=config.get_llm_api_key(),
            temperature=0.2,
            max_tokens=1024
        )
        self.system_prompt = BOOKING_SYSTEM_PROMPT

    @staticmethod
    def _friendly_date(iso_date: str) -> str:
        """Convert YYYY-MM-DD to a deterministic friendly string like "Monday, July 13, 2026"."""
        try:
            d = datetime.strptime(iso_date, "%Y-%m-%d")
            return d.strftime("%A, %B %d, %Y")
        except (ValueError, TypeError):
            return iso_date or "the selected date"

    def _llm_generate(self, state: ConversationState, instruction: str) -> str:
        """Generate a natural language response using the LLM."""
        friendly_date = self._friendly_date(state.booking_details.date)

        context = f"""
Current Booking State:
- Date: {state.booking_details.date or 'Not provided'} ({friendly_date})
- Time: {state.booking_details.time or 'Not provided'}
- Email: {state.booking_details.email or 'Not provided'}
- Service: {state.booking_details.service or 'Not provided'}
- Missing fields: {', '.join(state.booking_details.missing_fields()) or 'None'}
- Last action: {state.last_action}

Instruction: {instruction}
"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=context)
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _recover_date_from_context(self, state: ConversationState) -> None:
        """If date is missing but we have suggested slots, recover the date."""
        booking = state.booking_details
        if booking.date:
            return

        if state.suggested_slots:
            if booking.time:
                for slot in state.suggested_slots:
                    if slot.get("time") == booking.time and slot.get("date"):
                        booking.date = slot.get("date")
                        return

            dates = [s.get("date") for s in state.suggested_slots if s.get("date")]
            if dates:
                from collections import Counter
                most_common = Counter(dates).most_common(1)
                if most_common:
                    booking.date = most_common[0][0]

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Detect when user picks an alternative slot.
    # Handles: "09:00", "10:00", "11:00", "yes", "ok", "Monday at 09:00", etc.
    # Returns a dict with updated state so LangGraph persists it properly.
    # ═══════════════════════════════════════════════════════════════════════
    def _handle_alternative_confirmation(self, state: ConversationState) -> Dict[str, Any] | None:
        """Check if user selected or confirmed a suggested alternative.
        Returns a state-update dict if handled, else None."""
        if state.last_action not in ("suggested_times", "suggested_alternatives"):
            return None

        if not state.suggested_slots:
            return None

        last_user_msg = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content.lower().strip()
                break

        selected_time = None
        selected_date = None

        # 1) Direct match: user message contains a suggested time verbatim
        for slot in state.suggested_slots:
            slot_time = slot.get("time", "")
            if slot_time and slot_time in last_user_msg:
                selected_time = slot_time
                selected_date = slot.get("date")
                break

        # 2) normalize_time fallback ("3 PM", "11am", etc.)
        if not selected_time:
            normalized = normalize_time(last_user_msg)
            if normalized:
                for slot in state.suggested_slots:
                    if slot.get("time") == normalized:
                        selected_time = slot.get("time")
                        selected_date = slot.get("date")
                        break

        # 3) Regex fallback for "11", "11:30", "9am", "2 pm", etc.
        if not selected_time:
            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', last_user_msg)
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2) or "00"
                ampm = time_match.group(3)
                if ampm:
                    if ampm.lower() == "pm" and hour != 12:
                        hour += 12
                    elif ampm.lower() == "am" and hour == 12:
                        hour = 0
                extracted_time = f"{hour:02d}:{minute}"
                for slot in state.suggested_slots:
                    if slot.get("time") == extracted_time:
                        selected_time = slot.get("time")
                        selected_date = slot.get("date")
                        break

        # ── User picked a specific time from the list ──
        if selected_time:
            state.booking_details.time = selected_time
            if selected_date:
                state.booking_details.date = selected_date
            # Proceed to _check_and_book which will ask for final confirmation
            return {
                "booking_details": state.booking_details,
                "suggested_slots": state.suggested_slots,
                "last_action": state.last_action,
            }

        # ── User said generic confirm words without a specific time ──
        confirm_words = ["yes", "yeah", "sure", "confirm", "go ahead", "ok", "okay",
                         "proceed", "book it", "that works", "sounds good"]
        if any(w in last_user_msg for w in confirm_words):
            first = state.suggested_slots[0]
            state.booking_details.time = first.get("time")
            if first.get("date"):
                state.booking_details.date = first.get("date")
            state.last_action = "awaiting_confirmation"
            return {
                "booking_details": state.booking_details,
                "suggested_slots": state.suggested_slots,
                "last_action": "awaiting_confirmation",
            }

        return None

    def process(self, state: ConversationState) -> Dict[str, Any]:
        booking = state.booking_details

        # ── Recover date if user only sent a time ─────────────────
        self._recover_date_from_context(state)

        # ── CRITICAL FIX: Handle user selecting OR confirming a suggested alternative ─
        alt_update = self._handle_alternative_confirmation(state)
        if alt_update is not None:
            # Merge the alt_update into the state and proceed to check & book
            return {**alt_update, **self._check_and_book(state)}

        missing = booking.missing_fields()

        # ── BUG FIX: Don't silently default service ─────────────
        if not booking.service and "service" not in missing:
            missing = list(missing) + ["service"]

        # ── Detect availability / slot queries ──────────────────────
        last_user_msg = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content.lower()
                break

        availability_keywords = [
            "slots", "available", "availability", "what times",
            "open slots", "free slots", "show slots"
        ]
        is_availability_query = any(kw in last_user_msg for kw in availability_keywords)

        # ═══════════════════════════════════════════════════════════════════════
        # FIX: Preserve availability-check intent across the service-asking turn.
        # If agent just asked for service due to an availability query, and user
        # is now providing the service, continue the availability flow.
        # ═══════════════════════════════════════════════════════════════════════
        if not is_availability_query and state.last_action == "ask_service":
            human_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
            if len(human_msgs) >= 2:
                prev_msg = human_msgs[-2].content.lower()
                is_availability_query = any(kw in prev_msg for kw in availability_keywords)
        # ── Ask for service before showing slots ───────────────────
        if is_availability_query and booking.date:
            if not booking.service:
                instruction = (
                    "The user is asking about available slots but hasn't specified "
                    "what service they need. Ask them what type of service they are "
                    "looking for (e.g., Meeting, Consultation, etc.)."
                )
                prompt = self._llm_generate(state, instruction)
                return self._respond(state, prompt, last_action="ask_service")
            return self._check_and_book(state)

        # Step 1: Collect missing info (one at a time)
        if missing and not state.last_action.startswith("check_"):
            field = missing[0]
            instruction = (
                f"Ask the user for their {field} in a friendly, conversational way. "
                f"This is the only missing piece we need right now. Be concise."
            )
            prompt = self._llm_generate(state, instruction)
            return {
                "messages": [AIMessage(content=prompt)],
                "missing_info": missing,
                "current_agent": "booking",
                "last_action": f"ask_{field}"
            }

        # Step 2: Validate/normalize date and time
        if booking.date and not self._is_valid_iso_date(booking.date):
            normalized = resolve_relative_date(booking.date)
            if normalized:
                booking.date = normalized
            else:
                instruction = (
                    f"The date '{booking.date}' is invalid. "
                    f"Ask them to provide it as YYYY-MM-DD or say 'tomorrow' / 'next Monday'."
                )
                prompt = self._llm_generate(state, instruction)
                return self._respond(state, prompt)

        if booking.time and not self._is_valid_time(booking.time):
            normalized_time = normalize_time(booking.time)
            if normalized_time:
                booking.time = normalized_time
            else:
                instruction = (
                    f"The time '{booking.time}' is invalid. "
                    f"Ask them to say something like '3 PM' or '14:30'."
                )
                prompt = self._llm_generate(state, instruction)
                return self._respond(state, prompt)

        # Step 3 & 4: Check availability and book
        if (state.last_action == "check_availability" or
            (booking.date and booking.time and not state.booking_confirmed) or
            (booking.is_complete() and not state.booking_confirmed)):
            return self._check_and_book(state)

        # Step 5: Follow-up
        instruction = "Ask if there is anything else you can help with regarding their booking."
        prompt = self._llm_generate(state, instruction)
        return self._respond(state, prompt)

    def _check_and_book(self, state: ConversationState) -> Dict[str, Any]:
        booking = state.booking_details
        avail_result = check_availability.invoke({"date": booking.date})

        # Date is fully booked — suggest alternative dates
        if not avail_result["available"]:
            alt_result = get_alternative_slots.invoke({"date": booking.date, "limit": 3})
            alternatives = alt_result.get("alternatives", [])

            friendly_date = self._friendly_date(booking.date)

            # ═════════════════════════════════════════════════════════════════
            # FIX: Include correct weekday name for EACH alternative in the prompt
            # to prevent LLM hallucination of wrong day names.
            # ═════════════════════════════════════════════════════════════════
            alt_lines = []
            for alt in alternatives[:3]:
                alt_date = alt.get("date", "")
                alt_time = alt.get("time", "")
                if alt_date:
                    alt_weekday = self._friendly_date(alt_date)  # e.g. "Wednesday, July 16, 2026"
                    alt_lines.append(f"{alt_weekday} ({alt_date}) at {alt_time}")
                else:
                    alt_lines.append(f"{alt_time}")

            alt_text = "; ".join(alt_lines) if alt_lines else "none available"

            instruction = (
                f"The date {friendly_date} ({booking.date}) is fully booked. "
                f"Suggest these alternative slots: {alt_text}. "
                f"Ask if any of them work for the user."
            )
            prompt = self._llm_generate(state, instruction)
            # ═════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Include suggested_slots in the return dict so
            # LangGraph persists it for the next turn.
            # ═════════════════════════════════════════════════════════════════
            return {
                "messages": [AIMessage(content=prompt)],
                "suggested_slots": alternatives,
                "current_agent": "booking",
                "last_action": "suggested_alternatives",
                "error": f"Date {booking.date} fully booked"
            }

        open_slots = avail_result.get("open_slots", [])

        # ── User asked about slots without specifying a time ──────
        if not booking.time:
            service_text = booking.service if booking.service else "appointment"
            friendly_date = self._friendly_date(booking.date)
            instruction = (
                f"Here are the available slots for {friendly_date} ({booking.date}): {open_slots}. "
                f"Present them clearly and ask which one the user would like to book "
                f"for their {service_text}."
            )
            prompt = self._llm_generate(state, instruction)
            return {
                "messages": [AIMessage(content=prompt)],
                "suggested_slots": [{"date": booking.date, "time": s} for s in open_slots],
                "current_agent": "booking",
                "last_action": "suggested_times"
            }

        # ── Requested time is not available — suggest nearby slots ─
        if booking.time not in open_slots:
            nearby = [s for s in open_slots if abs(int(s[:2]) - int(booking.time[:2])) <= 2]
            if not nearby:
                nearby = open_slots[:3]

            friendly_date = self._friendly_date(booking.date)
            instruction = (
                f"The time {booking.time} is not available on {friendly_date} ({booking.date}). "
                f"Available nearby slots: {nearby}. Ask which one the user prefers."
            )
            prompt = self._llm_generate(state, instruction)
            return {
                "messages": [AIMessage(content=prompt)],
                "suggested_slots": [{"date": booking.date, "time": s} for s in nearby],
                "current_agent": "booking",
                "last_action": "suggested_times"
            }

        # ── Requested time IS available — confirm before booking ──
        if state.last_action != "awaiting_confirmation":
            service_text = booking.service if booking.service else "appointment"
            friendly_date = self._friendly_date(booking.date)
            instruction = (
                f"Confirm the booking details: {friendly_date} ({booking.date}) at {booking.time} for {booking.email}. "
                f"Service: {service_text}. Ask if they want to proceed."
            )
            prompt = self._llm_generate(state, instruction)
            return {
                "messages": [AIMessage(content=prompt)],
                "current_agent": "booking",
                "last_action": "awaiting_confirmation"
            }

        # User confirmed — execute booking
        return self._execute_booking(state)

    def _execute_booking(self, state: ConversationState) -> Dict[str, Any]:
        booking = state.booking_details

        # ── Guard against missing service ─────
        if not booking.service:
            instruction = (
                "The booking is almost complete, but we still need to know what "
                "service the user wants. Ask them to specify the service type."
            )
            prompt = self._llm_generate(state, instruction)
            return self._respond(state, prompt, last_action="ask_service")

        reserve_result = reserve_slot.invoke({
            "date": booking.date,
            "time": booking.time,
            "email": booking.email,
            "service": booking.service
        })

        if not reserve_result["success"]:
            alt_result = get_alternative_slots.invoke({"date": booking.date, "limit": 3})
            alternatives = alt_result.get("alternatives", [])

            friendly_date = self._friendly_date(booking.date)

            # ═════════════════════════════════════════════════════════════════
            # FIX: Include correct weekday name for EACH alternative here too,
            # in case the LLM needs to present fallback options after a
            # failed reservation.
            # ═════════════════════════════════════════════════════════════════
            alt_lines = []
            for alt in alternatives[:3]:
                alt_date = alt.get("date", "")
                alt_time = alt.get("time", "")
                if alt_date:
                    alt_weekday = self._friendly_date(alt_date)
                    alt_lines.append(f"{alt_weekday} ({alt_date}) at {alt_time}")
                else:
                    alt_lines.append(f"{alt_time}")

            alt_text = "; ".join(alt_lines) if alt_lines else "none available"

            instruction = (
                f"The slot was just taken by someone else on {friendly_date} ({booking.date}). "
                f"Present alternatives: {alt_text}. Ask if they want to pick one."
            )
            prompt = self._llm_generate(state, instruction)
            # ═════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Include suggested_slots here too.
            # ═════════════════════════════════════════════════════════════════
            return {
                "messages": [AIMessage(content=prompt)],
                "suggested_slots": alternatives,
                "current_agent": "booking",
                "last_action": "suggested_alternatives",
                "error": reserve_result.get("message", "Slot reservation failed")
            }

        details = {
            "date": booking.date,
            "time": booking.time,
            "service": booking.service,
            "appointment_id": reserve_result.get("appointment_id"),
            "timestamp": reserve_result.get("created_at", "")
        }

        notify_result = send_booking_notification.invoke({
            "email": booking.email,
            "details": json.dumps(details)
        })

        notify_status = "sent successfully" if notify_result["success"] else "in mock mode"
        friendly_date = self._friendly_date(booking.date)
        instruction = (
            f"Booking confirmed for {friendly_date} ({booking.date}) at {booking.time}. "
            f"Booking ID: {reserve_result.get('appointment_id')}. "
            f"Notification status: {notify_status}. "
            f"Congratulate the user and ask if they need anything else."
        )
        prompt = self._llm_generate(state, instruction)

        return {
            "messages": [AIMessage(content=prompt)],
            "booking_confirmed": True,
            "booking_details": BookingDetails(),
            "missing_info": [],
            "current_agent": "triage",
            "last_action": "booking_complete",
            "suggested_slots": []
        }

    def _respond(self, state: ConversationState, text: str, last_action: str = "responded") -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content=text)],
            "current_agent": "booking",
            "last_action": last_action
        }

    @staticmethod
    def _is_valid_iso_date(date_str: str) -> bool:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date_str))

    @staticmethod
    def _is_valid_time(time_str: str) -> bool:
        return bool(re.match(r"^\d{2}:\d{2}$", time_str))