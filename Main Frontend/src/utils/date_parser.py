"""
Date/Time normalization utilities.
Converts relative dates ("tomorrow", "next Monday") to YYYY-MM-DD.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta


def resolve_relative_date(text: str, reference_date: Optional[datetime] = None) -> Optional[str]:
    """
    Convert a relative date expression found within text to YYYY-MM-DD format.
    Searches for date patterns within full sentences.
    """
    if not text:
        return None

    original_text = text.strip()
    text_lower = original_text.lower()
    ref = reference_date or datetime.now()

    # Already in ISO format (standalone or in text)
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text_lower)
    if iso_match:
        return iso_match.group(1)

    # "today", "now"
    if re.search(r"\b(today|now)\b", text_lower):
        return ref.strftime("%Y-%m-%d")

    # "tomorrow", "tmrw", "tom"
    if re.search(r"\b(tomorrow|tmrw|tom)\b", text_lower):
        return (ref + timedelta(days=1)).strftime("%Y-%m-%d")

    # "day after tomorrow"
    if re.search(r"\b(day after tomorrow|day after tmrw)\b", text_lower):
        return (ref + timedelta(days=2)).strftime("%Y-%m-%d")

    # "in X days"
    days_match = re.search(r"\bin\s+(\d+)\s+days?\b", text_lower)
    if days_match:
        days = int(days_match.group(1))
        return (ref + timedelta(days=days)).strftime("%Y-%m-%d")

    # Weekdays mapping
    weekdays = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    # "next Monday", "next Tuesday", etc.
        # "next Monday", "next Tuesday", etc.
    next_day_match = re.search(
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri|sat|sun)\b",
        text_lower,
    )
    if next_day_match:
        day_name = next_day_match.group(1)
        target_weekday = weekdays.get(day_name)
        if target_weekday is not None:
            days_ahead = (target_weekday - ref.weekday()) % 7
            # ═══════════════════════════════════════════════════════════
            # FIX: "next X" always means the occurrence in the NEXT week.
            # If the target day is in the current week, add 7 more days.
            # ═══════════════════════════════════════════════════════════
            if days_ahead == 0:
                days_ahead = 7  # Today is that day → next week
            elif days_ahead < 7:
                # Target day is still in current week (e.g., "next Wednesday" 
                # from Monday = 2 days, but that's THIS Wednesday, not NEXT)
                # Add 7 to push to next week
                days_ahead += 7
            return (ref + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "this Monday", "this Tuesday", etc.
    this_day_match = re.search(
        r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri|sat|sun)\b",
        text_lower,
    )
    if this_day_match:
        day_name = this_day_match.group(1)
        target_weekday = weekdays.get(day_name)
        if target_weekday is not None:
            days_ahead = (target_weekday - ref.weekday()) % 7
            if days_ahead == 0:
                return ref.strftime("%Y-%m-%d")
            return (ref + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "Monday", "Tuesday" (without this/next) — upcoming occurrence
    upcoming_day_match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri|sat|sun)\b",
        text_lower,
    )
    if upcoming_day_match:
        day_name = upcoming_day_match.group(1)
        target_weekday = weekdays.get(day_name)
        if target_weekday is not None:
            days_ahead = (target_weekday - ref.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week if today
            return (ref + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Try dateutil parser for expressions like "July 15", "15 July", "15/07/2026", "14 July, 2026"
    try:
        # Extract potential date portion (remove common non-date words)
        cleaned = re.sub(r"\b(what|slots|are|available|book|appointment|for|at|pm|am|in|the|a|an)\b", " ", text_lower)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # ═══════════════════════════════════════════════════════════
        # BUG FIX: If after cleaning the text is purely a time
        # reference (e.g. "10:00", "3 pm", "noon"), skip dateutil
        # parsing so we don't return *today's* date and overwrite
        # the previously resolved date in session state.
        # ═══════════════════════════════════════════════════════════
        if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", cleaned) or \
           re.match(r"^\d{1,2}\s*(am|pm)$", cleaned) or \
           re.match(r"^(noon|midnight|morning|afternoon|evening|night)$", cleaned):
            return None

        parsed = dateutil_parser.parse(cleaned, default=ref, fuzzy=True)
        if parsed.year < ref.year:
            parsed = parsed.replace(year=ref.year)
        if parsed.date() < ref.date() and parsed.year == ref.year:
            parsed = parsed + relativedelta(years=1)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    return None


def normalize_time(text: str) -> Optional[str]:
    """
    Convert a time expression found within text to HH:MM (24-hour) format.
    """
    if not text:
        return None

    text_lower = text.strip().lower()

    # Helper: detect whether the text contains an explicit AM/PM indicator
    # (including dotted forms like a.m. / p.m.) so we don't overwrite them.
    def _has_ampm(txt: str) -> bool:
        return bool(re.search(r"(?<!\w)(am|pm|a\.m\.|p\.m\.)(?!\w)", txt))

    # Special cases
    if re.search(r"\b(noon|midday|12 noon)\b", text_lower):
        return "12:00"
    if re.search(r"\b(midnight|12 midnight)\b", text_lower):
        return "00:00"

    # "X AM/PM" pattern
    am_pm_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text_lower)
    if am_pm_match:
        hour = int(am_pm_match.group(1))
        minute = int(am_pm_match.group(2) or 0)
        period = am_pm_match.group(3)

        if period == "pm" and hour != 12:
            hour += 12
        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # "morning", "afternoon", "evening"
    time_of_day = {
        "morning": "09:00",
        "afternoon": "14:00",
        "evening": "17:00",
        "night": "19:00",
    }
    for tod, time_str in time_of_day.items():
        if re.search(rf"\b{tod}\b", text_lower):
            return time_str

    # HH:MM or HH:MM:SS pattern
    time_match = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b", text_lower)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        # BUG FIX: Ambiguous time without AM/PM — infer PM during business hours.
        # If hour is 1–6 (e.g., "2:00" → 14:00, "3:30" → 15:30).
        # Hours 7+ are kept as AM (e.g., "7:00" → 07:00, "11:00" → 11:00).
        if 1 <= hour <= 6 and not _has_ampm(text_lower):
            hour += 12
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    # BUG FIX: Bare hour number — only match if clearly a time reference
    # (preceded by a time indicator like "at", "around", "for", or standalone)
    # to prevent matching dates like "July 12" or "2026-07-12".
    hour_match = re.search(
        r"(?:\b(?:at|around|for|from|to|until|by)\s+|^)(\d{1,2})(?:\s*(?:am|pm|o'clock)?\b|$)",
        text_lower,
    )
    if hour_match:
        hour = int(hour_match.group(1))
        if 0 <= hour <= 23:
            # BUG FIX: Ambiguous bare hour without AM/PM — infer PM during business hours.
            if 1 <= hour <= 6 and not _has_ampm(text_lower):
                hour += 12
            return f"{hour:02d}:00"

    return None


def extract_date_and_time(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Try to extract both date and time from a user message.
    """
    # Extract date first
    iso_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_date:
        date_str = iso_date.group(1)
        # BUG FIX: Remove the ISO date before time extraction so that the day
        # number (e.g., "12" from "2026-07-12") is not falsely parsed as a time.
        text_for_time = text.replace(iso_date.group(0), " ", 1)
    else:
        date_str = resolve_relative_date(text)
        text_for_time = text

    # Extract time using specific patterns to avoid matching date numbers as times
    time_str = None

    # Pattern 1: HH:MM or HH:MM:SS
    time_match = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", text_for_time)
    if time_match:
        time_str = normalize_time(time_match.group(1))

    # Pattern 2: X AM/PM
    if not time_str:
        am_pm_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\b", text_for_time)
        if am_pm_match:
            time_str = normalize_time(am_pm_match.group(1))

    # Pattern 3: Let normalize_time handle the rest with its safer bare-number logic
    if not time_str:
        time_str = normalize_time(text_for_time)

    return date_str, time_str