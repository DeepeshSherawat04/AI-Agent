### TrulyIAS Scheduling Assistant

Assignment 2: Multi-Agent Scheduling Assistant with Tool Validation & Mock Notifications
An AI-powered multi-agent scheduling system built with LangGraph and Streamlit, featuring persistent memory, tool validation, and mock webhook notifications.


### Live Demo
Hosted App URL: Streamlit Cloud Deployment


### Table of Contents

Overview
Architecture
Agent Workflow
Features
Tech Stack
Project Structure
Setup Instructions
Environment Variables
Usage Guide
Testing
Deployment
Screenshots
Assignment Compliance


### Overview
The TrulyIAS Scheduling Assistant is a multi-agent conversational AI that handles calendar bookings through natural language. It orchestrates two specialized agents:

Triage Agent — Analyzes user intent and routes to the appropriate specialist
Booking Specialist — Manages calendar operations, validates inputs, and negotiates alternatives
The system features persistent conversation memory (survives page refreshes), relative date resolution ("tomorrow", "next Monday"), and mock webhook notifications for booking confirmations.



### Architecture

    ┌─────────────────────────────────────────────────────────────┐
    │                    Streamlit Frontend                       │
    │              (Chat UI + Service Dropdown)                   │
    └──────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  LangGraph Workflow                         │
    │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐  │
    │  │   Triage    │───▶│  Booking        │──▶│  Tool       │  │
    │  │   Agent     │    │  Specialist     │    │  Execution  │  │
    │  └─────────────┘    └─────────────────┘    └─────────────┘  │
    │         │                   │                    │          │
    │         └───────────────────┴────────────────────┘          │
    │                           │                                 │
    │                           ▼                                 │
    │                ┌─────────────────────┐                      │
    │                │ SQLite Checkpointer │                      │
    │                │ (State Persistence) │                      │
    │                └─────────────────────┘                      │
    └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     SQLite Database                         │
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
    │  │ appointments │   │availability_ │   │  checkpoints │     │
    │  │    table     │   │   slots      │   │    table     │     │
    │  └──────────────┘   └──────────────┘   └──────────────┘     │
    └─────────────────────────────────────────────────────────────┘


### 🤖 Agent Workflow

**Triage Agent**
Role: First point of contact for all user messages
Responsibilities:
Classify intent: general, booking, availability_check
Respond directly to general queries ("What services do you offer?")
Route scheduling intents to the Booking Specialist
Return to triage after booking completion

**Booking Specialist**
Role: Handles all calendar-related operations
Responsibilities:
Date Resolution: Converts relative dates to YYYY-MM-DD format
Input Validation: Ensures dates are not in the past
Tool Execution: Calls check_availability, reserve_slot, get_alternative_slots
Negotiation: Offers alternative dates/times when slots are unavailable
Confirmation: Summarizes booking details and awaits user confirmation


### Features
*Table*
|           Feature          |   	                         Description                                     |
|----------------------------|-------------------------------------------------------------------------------|
| 📅 Relative Date Parsing   |   "tomorrow", "day after tomorrow", "next Monday", "in 3 days" → YYYY-MM-DD   |
| ⏰ Time Normalization      |	"3 PM", "15:00", "noon" → HH:MM (24-hour)                                   |  
| 🔍 Availability Check      |	 Query open slots for any future date                                        |
| 🔄 Alternative Negotiation |	 When a slot is taken, suggests next available dates                         |
| 📧 Mock Notifications      |	 Simulates email/WhatsApp confirmation via webhook                           |
| 💾 Persistent Memory       |	 Conversation state survives page refreshes (SQLite)                         |
| 🎛️ Service Dropdown        |   Interactive buttons for service selection                                   |
| 🏷️ Agent Badges            |   Visual indicator showing which agent is active                              |
| 📊 Recent Bookings         |   Sidebar shows last 5 confirmed appointments                                 |


### Tech Stack
*Table*

| Layer |	Technology |
|-------|--------------|

|Frontend|Streamlit|
|Agent Framework|LangGraph (LangChain)|
|LLM|Groq API (Llama 3)|
|Database|SQLite|
|State Persistence|LangGraph SQLite Checkpointer|
|Language|Python 3.10+|



### 📁 Project Structure

ai-agent-internship/
├── assignment-2/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── triage_agent.py          # Intent classification & routing
│   │   │   └── booking_specialist.py    # Calendar operations & negotiation
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   └── workflow.py              # LangGraph state machine
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   └── conversation_state.py    # Pydantic state models
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   └── calendar_tools.py        # @tool decorated functions
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── database.py              # SQLite DB manager
│   │   │   ├── date_parser.py           # Relative date/time resolution
│   │   │   ├── webhook.py               # Mock notification sender
│   │   │   ├── config.py                # App configuration
│   │   │   └── persistence.py           # SQLite checkpointer
│   │   └── tests/
│   │       ├── test_booking.py
│   │       ├── test_date_parser.py
│   │       └── test_triage.py
│   ├── scripts/
│   │   ├── setup_db.py                  # Database initialization
│   │   └── test_webhook.py              # Webhook endpoint tester
│   ├── data/
│   │   └── .gitkeep
│   ├── docs/
│   │   ├── architecture.md
│   │   └── api_reference.md
│   ├── config/
│   │   └── settings.yaml
│   ├── .env.example                     # Environment variable template
│   ├── .env                             # Your local secrets (gitignored)
│   ├── requirements.txt                 # Python dependencies
│   ├── pyproject.toml                   # Project metadata
│   ├── streamlit_app.py                 # Main Streamlit entry point
│   └── README.md                        # This file


### 🚀 Setup Instructions

1. Clone the Repository
bash
git clone https://github.com/YOUR_USERNAME/trulyias-scheduling-assistant.git
cd trulyias-scheduling-assistant/assignment-2

2. Create Virtual Environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Set Up Environment Variables
bash
cp .env.example .env
Edit .env and add your API keys:

# Required: Groq API Key for LLM inference
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Enable real email notifications
SEND_REAL_EMAIL=false
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password

# Optional: Webhook.site for mock notification visualization
WEBHOOK_URL=https://webhook.site/your-unique-url
⚠️ Never commit .env to GitHub. It is already in .gitignore.

5. Initialize Database
bash
python scripts/setup_db.py
This creates:
data/appointments.db — Stores confirmed bookings
data/checkpoints.db — Stores conversation state for persistence

6. Run the Application
bash
streamlit run streamlit_app.py
The app will open at http://localhost:8501.



### 🔐 Environment Variables


|     Variable     |	Required   |                 Description                       |
|------------------|---------------|---------------------------------------------------|
|  GROQ_API_KEY    |	✅ Yes	 |   API key for Groq LLM inference                  |
|  SEND_REAL_EMAIL |	❌ No	 |   Set to true to enable real Gmail SMTP           |
|  SENDER_EMAIL    |	❌ No	 |   Gmail address for real email sending            |
|  SENDER_PASSWORD |	❌ No	 |   Gmail app password (not your login password)    |
|  WEBHOOK_URL	   |    ✅ Yes	 |   Webhook.site URL for mock notification logging  |




### 🧪 Test credentials

## Scenario 1: Book an Appointment (Full Flow)
Goal: Complete a full booking from start to finish.
Table

| Step | Action | Expected Result |
|------|--------|-----------------|
|1|	Type: `hi`	| Bot welcomes you and lists capabilities|
|2|	Type: `Book an appointment for tomorrow at 3 PM`	| Bot asks for your email address|
|3|	Type: `test@example.com`	| Bot asks which service you want to book|
|4|	Type: `Review`	| Bot shows available slots (if your time is taken) or confirms booking|
|5|	Select a time slot (e.g., `09:00`)	| Bot shows booking summary and asks for confirmation|
|6|	Type: `yes`	| Bot confirms booking with Booking ID|

✅ Success Check:
Booking appears in the Recent Bookings panel on the left sidebar
Confirmation message includes Booking ID and email notification sent


## Scenario 2: Check Availability for a Specific Day
Goal: View available time slots for a future date.
Table
| Step | Action | Expected Result |
|------|--------|-----------------|
|1|	Type: What slots are available next Monday?	| Bot asks for the service type|
|2|	Type: Demo	| Bot shows available slots for that Monday|
✅ Success Check:
Slots displayed as bullet points (e.g., 10:00, 14:00, 16:00)
Date is correctly resolved (e.g., Monday, July 20, 2026)


## Scenario 3: Multi-Service Booking
Goal: Book different services to verify the system handles multiple service types.

Available Services to Test:
Review
Consultation
Meeting
Demo
Test each service by asking:
plain
Book a Consultation for next Tuesday at 10 AM
Book a Meeting for Friday at 2 PM

✅ Success Check:
Each booking appears in Recent Bookings with correct service label
No slot conflicts between different services



### 🧪 Testing

Run the test suite:
bash
# Test date/time parsing
python -m pytest src/tests/test_date_parser.py -v

# Test booking flow
python -m pytest src/tests/test_booking.py -v

# Test triage routing
python -m pytest src/tests/test_triage.py -v

# Test webhook endpoint
python scripts/test_webhook.py



#### 🌐 Deployment

# Streamlit Cloud (Recommended)

Push code to GitHub (ensure .env is in .gitignore)

Go to share.streamlit.io
Connect your GitHub repository

Set secrets in Streamlit Cloud dashboard:
toml
GROQ_API_KEY = "your-key-here"
Deploy! 



### ✅ Assignment Compliance

|    Requirement    |      Status           |                   Implementation                     |
|-------------------|-----------------------|------------------------------------------------------|
|   Triage Agent     |	         ✅	      |   src/agents/triage_agent.py — routes general vs. scheduling queries |
|  Booking Specialist  |	     ✅	      |   src/agents/booking_specialist.py — manages calendar tools  |
|  LangGraph Workflow  |	     ✅   	  |   src/graph/workflow.py — state-machine routing |
|  Date Normalization  |	     ✅	      |   src/utils/date_parser.py — "tomorrow" → YYYY-MM-DD |
|  check_availability()|        ✅	      |   src/tools/calendar_tools.py — queries mock DB |
|  reserve_slot()      |        ✅	      |   src/tools/calendar_tools.py — saves to SQLite |
|  send_booking_notification() | ✅        |   src/utils/webhook.py — mock webhook trigger |
|  Negotiation & Alternatives  | ✅	      |   _get_alternative_slots() — forward-only date filtering |
|  State Persistence   |	     ✅	      |    SQLiteCheckpointer — survives page refreshes |
|  Tool Validation     |	     ✅	      |    Validates dates/times before tool execution |
|  Mock Notifications  |	     ✅	      |    Webhook.site / Pipedream integration |



### 🐛 Known Issues & Fixes Applied

|                 Issue                   |       File    |                        Fix                           |
|-----------------------------------------|---------------|------------------------------------------------------|
|        Past dates in alternatives       |  database.py  | Added date > ? filter to _get_alternative_slots      |
|Agent badge shows "TRIAGE" during booking|booking_specialist.py|Added "current_agent": "booking" to all returns |
|   Service dropdown not appearing   |  streamlit_app.py  |	Detects service-asking intent from last AI message   |
| "next Wednesday" resolves to this week  |  date_parser.py  | Added days_ahead += 7 for "next X" from same week |
|       Missing bookings in sidebar       |	 database.py  | Order by id DESC + explicit created_at on insert     |
|       Weekday name hallucination        |  booking_specialist.py  | Inject correct weekday into LLM prompt     |


© 2026 TrulyIAS Internship
All rights reserved.

👤 Author
Deepesh Sherawat