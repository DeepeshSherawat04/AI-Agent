# API Reference

## Tools

### `check_availability(date: str) -> dict`
Checks if a given date has available appointment slots.

**Parameters:**
- `date` (str): Date in YYYY-MM-DD format

**Returns:**
```json
{
  "date": "2026-07-15",
  "available": true,
  "slots": [
    {"time": "09:00", "available": true},
    {"time": "10:00", "available": false}
  ],
  "open_slots": ["09:00", "11:00", "14:00"],
  "message": "3 slots available on 2026-07-15."
}
```

---

### `reserve_slot(date: str, time: str, email: str, service: str) -> dict`
Saves the appointment slot in the SQLite database.

**Parameters:**
- `date` (str): YYYY-MM-DD
- `time` (str): HH:MM (24h)
- `email` (str): Confirmation email
- `service` (str): Service type (default: "General Consultation")

**Returns:**
```json
{
  "success": true,
  "appointment_id": 42,
  "date": "2026-07-15",
  "time": "14:00",
  "email": "user@example.com",
  "service": "Consultation",
  "message": "Appointment confirmed for 2026-07-15 at 14:00."
}
```

---

### `send_booking_notification(email: str, details: str) -> dict`
Performs a POST request to the configured webhook endpoint.

**Parameters:**
- `email` (str): Recipient email
- `details` (str): JSON string of booking details

**Returns:**
```json
{
  "success": true,
  "status_code": 200,
  "message": "Notification sent successfully via webhook.",
  "webhook_response": "OK"
}
```

---

### `get_alternative_slots(date: str, limit: int) -> dict`
Finds alternative available slots when the requested date is fully booked.

**Parameters:**
- `date` (str): Original requested date (to exclude)
- `limit` (int): Max alternatives (default 3)

**Returns:**
```json
{
  "original_date": "2026-07-15",
  "alternatives": [
    {"date": "2026-07-16", "time": "09:00"},
    {"date": "2026-07-16", "time": "10:00"}
  ],
  "count": 2,
  "message": "Found 2 alternative slots."
}
```

## State Model

### `ConversationState`

| Field | Type | Description |
|-------|------|-------------|
| `messages` | List[Message] | Chat history |
| `intent` | str | "general", "booking", "unknown" |
| `booking_details` | BookingDetails | Extracted appointment info |
| `missing_info` | List[str] | Fields still needed |
| `current_agent` | str | Active agent name |
| `last_action` | str | Last system action |
| `error` | str | Error message if any |
| `suggested_slots` | List[dict] | Alternative slots offered |
| `booking_confirmed` | bool | True when complete |
| `thread_id` | str | Session identifier |

### `BookingDetails`

| Field | Type | Description |
|-------|------|-------------|
| `date` | str | YYYY-MM-DD |
| `time` | str | HH:MM |
| `email` | str | User email |
| `service` | str | Service type |
| `confirmed` | bool | Finalization flag |
