"""
Triage Agent: Analyzes incoming user messages.
Routes to Booking Specialist if scheduling intent detected.
Otherwise responds directly to general queries.
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from src.utils.config import get_config
from src.state.conversation_state import ConversationState, BookingDetails


TRIAGE_SYSTEM_PROMPT = """You are the Triage Agent for TrulyIAS Scheduling Assistant.

Your job is to analyze the user's message and classify their intent.

## Intent Classification Rules:

1. **booking** — User wants to schedule, book, check, cancel, or modify an appointment.
   Examples: "Book an appointment", "I want to schedule a meeting", "Check my booking", 
   "Cancel my appointment", "What slots are available tomorrow?", "Reserve a slot"

2. **general** — User asks a general question, greets, or makes small talk.
   Examples: "Hello", "What services do you offer?", "How are you?", "Tell me about TrulyIAS",
   "What are your business hours?", "Where are you located?"

## Response Rules:

- If intent is **general**: Respond directly in a friendly, professional tone. 
  Do NOT mention booking unless the user asks.

- If intent is **booking**: Acknowledge the request and transition to the Booking Specialist.
  Extract any booking details already mentioned (date, time, email, service type).
  Say something like: "I'd be happy to help you with that! Let me connect you with our Booking Specialist."

## Output Format:

Return a JSON object with exactly these keys:
```json
{
  "intent": "booking" or "general",
  "response": "Your direct response to the user",
  "extracted_details": {
    "date": "YYYY-MM-DD or null",
    "time": "HH:MM or null", 
    "email": "email or null",
    "service": "service type or null"
  }
}
```

Be concise but warm. Always set intent accurately.
"""


class TriageAgent:
    """Classifies user intent and routes to appropriate agent."""

    def __init__(self):
        config = get_config()
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=config.get_llm_api_key(),
            temperature=0.1,
            max_tokens=512
        )
        self.system_prompt = TRIAGE_SYSTEM_PROMPT

    def analyze(self, state: ConversationState) -> Dict[str, Any]:
        """
        Analyze the latest user message and classify intent.

        Returns updated state with intent, response, and extracted booking details.
        """
        # Get the last human message
        last_human_msg = None
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break

        if not last_human_msg:
            return {
                "intent": "unknown",
                "response": "I didn't catch that. Could you please repeat?",
                "current_agent": "triage"
            }

        # Build messages for LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User message: {last_human_msg}")
        ]

        # Get LLM response
        response = self.llm.invoke(messages)
        content = response.content

        # Parse JSON from response
        import json
        import re

        try:
            # Extract JSON block if wrapped in markdown
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            result = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: simple keyword-based classification
            text_lower = last_human_msg.lower()
            booking_keywords = [
                "book", "schedule", "appointment", "slot", "reserve", "available",
                "cancel", "reschedule", "meeting", "consultation", "tomorrow",
                "next week", "next monday", "this week", "check availability"
            ]

            is_booking = any(kw in text_lower for kw in booking_keywords)
            result = {
                "intent": "booking" if is_booking else "general",
                "response": "I'd be happy to help you with that!" if is_booking else "I'm here to help!",
                "extracted_details": {"date": None, "time": None, "email": None, "service": None}
            }

        # Update state
        intent = result.get("intent", "unknown")
        response_text = result.get("response", "")
        extracted = result.get("extracted_details", {})

        # Build updated booking details from extraction
        booking_details = state.booking_details
        if extracted.get("date"):
            booking_details.date = extracted["date"]
        if extracted.get("time"):
            booking_details.time = extracted["time"]
        if extracted.get("email"):
            booking_details.email = extracted["email"]
        if extracted.get("service"):
            booking_details.service = extracted["service"]

        # Determine missing fields
        missing = booking_details.missing_fields()

        return {
            "intent": intent,
            "messages": [AIMessage(content=response_text)],
            "booking_details": booking_details,
            "missing_info": missing,
            "current_agent": "booking" if intent == "booking" else "triage",
            "last_action": "triage_classified"
        }
