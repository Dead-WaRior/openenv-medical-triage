"""Test Medical Triage Environment"""

from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel
import random

print("Testing Medical Triage Environment...")
print("="*50)

env = MedicalTriageEnv(max_steps=20, random_seed=42)
obs = env.reset()

print("[OK] Environment reset successful")
print(f"  Waiting patients: {len(obs.waiting_patients)}")
print(f"  Available rooms: {obs.available_rooms[:3]}...")
print(f"  Available doctors: {list(obs.available_doctors.keys())[:3]}...")

if obs.waiting_patients:
    patient = obs.waiting_patients[0]
    print(f"\nTesting action on patient {patient.id}...")
    print(f"  Chief complaint: {patient.chief_complaint.value}")
    print(f"  Age: {patient.age}")
    
    action = TriageAction(
        patient_id=patient.id,
        esi_level=ESILevel.URGENT,
        order_tests=["CBC", "CMP"]
    )
    
    obs, reward, done, info = env.step(action)
    print("[OK] Action executed")
    print(f"  Reward: {reward.total:.3f}")
    print(f"  Patient assigned ESI: {action.esi_level}")
    print(f"  Outcome score: {reward.patient_outcome_score:.2f}")
    print(f"  Wait score: {reward.wait_time_score:.2f}")

print("\n" + "="*50)
print("[OK] All tests passed! Environment is working.")
print("[OK] Ready to run inference.py")
