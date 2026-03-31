"""
OpenEnv Server Entry Point
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv

def main():
    """OpenEnv server entry point"""
    env = MedicalTriageEnv()
    print("Medical Triage Environment ready")
    return env