#!/usr/bin/env python
"""
Quick script to run membrane permeability analysis.

Usage:
    python run_analysis.py
    python run_analysis.py --membrane-type BBB --cholesterol 0.3
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from membrane_runner.runner import run_permeability_analysis, main

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use argparse if arguments provided
        main()
    else:
        # Run default analysis
        run_permeability_analysis()
