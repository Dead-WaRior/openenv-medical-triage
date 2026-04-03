"""
OpenEnv Server Entry Point
"""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
import main as main_app  # Rename the import to avoid shadowing


def main():
    """Callable entry point expected by validators and script runners."""
    start_server()


def start_server():
    """OpenEnv server entry point"""
    print("Starting Medical Triage Environment server...")
    uvicorn.run(main_app.app, host="0.0.0.0", port=7860)

# This is required for the entry point to be callable
if __name__ == "__main__":
    main()