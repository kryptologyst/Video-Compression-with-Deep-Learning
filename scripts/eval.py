#!/usr/bin/env python3
"""Evaluation script for video compression models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.evaluator import main

if __name__ == "__main__":
    main()
