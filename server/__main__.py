"""
OpenEnv Server Main Entry Point
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv

def main():
    env = MedicalTriageEnv()
    print("Medical Triage Environment ready")
    return env

if __name__ == "__main__":
    main()