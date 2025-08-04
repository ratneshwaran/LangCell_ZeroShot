#!/usr/bin/env python3
"""
Run Kidney Cancer Data Preprocessing

This script runs the preprocessing for kidney cancer data.
"""

import sys
import os

# Add the data_preprocess directory to the path
sys.path.append('/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/data_preprocess')

# Import the preprocessing function
from preprocess_kidney import main

if __name__ == "__main__":
    print("Starting kidney cancer data preprocessing...")
    main() 