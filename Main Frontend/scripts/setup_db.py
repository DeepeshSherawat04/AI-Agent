"""
One-time database initialization script.
Creates SQLite tables for appointments and checkpoints.
Run this before starting the app for the first time.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager
from src.utils.config import get_config


def main():
    print("🔧 TrulyIAS Database Setup")
    print("=" * 50)

    # Initialize database (creates tables + seeds availability slots)
    db = DatabaseManager()

    print(f"✅ Database initialized at: {get_config().get_db_path()}")
    print(f"✅ Checkpoint database at: {get_config().get_checkpoint_db_path()}")
    print(f"✅ Availability slots seeded for next 14 days")

    # Quick verification
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    result = db.check_availability(today)
    print(f"\n📅 Sample availability check for {today}:")
    print(f"   Available: {result['available']}")
    print(f"   Open slots: {result.get('open_slots', [])}")

    print("\n🚀 Database ready! You can now run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
