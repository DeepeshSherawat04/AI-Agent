"""
Quick test script to verify webhook endpoint is reachable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.utils.webhook import send_booking_notification


def main():
    print("🌐 Webhook Endpoint Test")
    print("=" * 50)

    config = get_config()
    url = config.webhook_url

    if not url or "webhook.site" in url and "xxxx" in url:
        print("❌ No valid webhook URL configured.")
        print("   Please set WEBHOOK_URL in your .env file.")
        print("   Get a free URL from: https://webhook.site")
        return

    print(f"Testing URL: {url}")

    test_details = {
        "date": "2026-07-15",
        "time": "14:00",
        "service": "Test Consultation",
        "appointment_id": 999,
        "timestamp": "2026-07-11T20:00:00"
    }

    result = send_booking_notification(
        email="test@example.com",
        details=test_details,
        webhook_url=url
    )

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Message: {result['message']}")
    if result.get('status_code'):
        print(f"  Status Code: {result['status_code']}")

    print("\n✅ Test complete. Check your webhook.site dashboard for the payload.")


if __name__ == "__main__":
    main()
