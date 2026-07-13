# Architecture Overview

## Multi-Agent System

```
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
```

## Agent Responsibilities

### Triage Agent
- **Input**: Raw user message
- **Process**: LLM-based intent classification (general vs booking)
- **Output**: Intent label + initial response + extracted booking details
- **Model**: `llama-3.3-70b-versatile` (Groq)

### Booking Specialist
- **Input**: Conversation state with booking details
- **Process**:
  1. Identify missing fields (date, time, email, service)
  2. Normalize relative dates ("tomorrow" → YYYY-MM-DD)
  3. Validate time formats
  4. Check availability via SQLite
  5. Negotiate alternatives if slot unavailable
  6. Reserve slot upon confirmation
  7. Send webhook notification
- **Output**: Updated state + AI response message
- **Model**: `llama-3.3-70b-versatile` (Groq)

## State Persistence

LangGraph's **SQLiteSaver** checkpoints the full conversation state after every node execution. This means:
- Conversation survives page refreshes
- Multi-turn booking workflows resume seamlessly
- Thread IDs isolate different user sessions

## Database Schema

### `appointments` table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment ID |
| date | TEXT | YYYY-MM-DD |
| time | TEXT | HH:MM |
| email | TEXT | User email |
| service | TEXT | Service type |
| status | TEXT | confirmed/cancelled |
| created_at | TEXT | ISO timestamp |

### `availability_slots` table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment ID |
| date | TEXT | YYYY-MM-DD |
| time | TEXT | HH:MM |
| is_available | INTEGER | 1=available, 0=booked |

## Tool Registry

| Tool | Function | Mock/Real |
|------|----------|-----------|
| `check_availability` | Query SQLite for open slots | Real (SQLite) |
| `reserve_slot` | Insert appointment, mark slot taken | Real (SQLite) |
| `send_booking_notification` | POST to Webhook.site | Real (HTTP) |
| `get_alternative_slots` | Find nearby available slots | Real (SQLite) |
