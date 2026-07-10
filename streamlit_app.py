import sys
from pathlib import Path

# Add assignment-1 to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "assignment-1"))

from app import main

if __name__ == "__main__":
    main()