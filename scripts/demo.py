#!/usr/bin/env python3
"""Demo script for video compression."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent.parent / "demo" / "streamlit_app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")

if __name__ == "__main__":
    main()
