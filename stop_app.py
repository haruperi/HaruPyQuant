#!/usr/bin/env python3
"""
Script to stop the main application by creating a shutdown file.
"""

import os
from datetime import datetime

SHUTDOWN_FILE = "shutdown_requested.txt"

def create_shutdown_file():
    """Create shutdown file to request shutdown."""
    with open(SHUTDOWN_FILE, 'w') as f:
        f.write(f"Shutdown requested at {datetime.now()}\n")
    print(f"Shutdown file created: {SHUTDOWN_FILE}")
    print("The application should stop within a few seconds.")

if __name__ == "__main__":
    create_shutdown_file() 