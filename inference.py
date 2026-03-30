"""
Inference Script for Medical Triage Environment
"""

import os
import random
import numpy as np
from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MAX_STEPS = 50

def run_episode(env, episode_num, use_random=True):
    """Run a single episode"""
    print(f"\n--- Episode {episode_num} ---")
    
    observation = env.reset()
    total_reward = 0.0
    step_count = 0
    
    for step in range(MAX_STEPS):
        # Get action
        if use_random and observation.waiting_patients:
            patient = random.choice(observation.waiting_patients)
            action = TriageAction(
                patient_id=patient.id,
                esi_level=random.choice([1, 2, 3, 4, 5]),
                assigned_room=random.choice(observation.available_rooms) if observation.available_rooms else None,
                assigned_doctor_id=random.choice(list(observation.available_doctors.keys())) if observation.available_doctors else None,
                order_tests=[],
                initiate_resuscitation=False
            )
        elif observation.waiting_patients:
            patient = observation.waiting_patients[0]
            action = TriageAction(
                patient_id=patient.id,
                esi_level=ESILevel.URGENT,
                order_tests=["CBC"]
            )
        else:
            break
        
        # Execute action
        observation, reward, done, info = env.step(action)
        total_reward += reward.total
        step_count = step + 1
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"  Step {step}: Reward={reward.total:.3f}, LWBS={observation.lwbs_rate:.1%}, Patients={observation.total_patients}")
        
        if done:
            print(f"  Episode finished at step {step+1}")
            break
    
    print(f"\nEpisode {episode_num} Summary:")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Steps: {step_count}")
    print(f"  Patients Arrived: {info['metrics']['total_arrivals']}")
    print(f"  Patients Left (LWBS): {info['metrics']['total_lwbs']}")
    print(f"  Patients Died: {info['metrics']['total_mortality']}")
    
    return {
        "episode": episode_num,
        "total_reward": total_reward,
        "steps": step_count,
        "arrivals": info['metrics']['total_arrivals'],
        "lwbs": info['metrics']['total_lwbs'],
        "mortality": info['metrics']['total_mortality']
    }

def main():
    print("="*60)
    print("Medical Triage Environment - Baseline Inference")
    print("="*60)
    
    if not API_KEY:
        print("\n[INFO] No API_KEY found. Running with random agent...")
        use_random = True
    else:
        print(f"\n[INFO] API configured. Using random agent for baseline...")
        use_random = True
    
    # Create environment
    env = MedicalTriageEnv(max_steps=MAX_STEPS, random_seed=42)
    
    # Run episodes
    results = []
    for episode in range(1, 4):
        result = run_episode(env, episode, use_random)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_reward = sum(r["total_reward"] for r in results)
    total_arrivals = sum(r["arrivals"] for r in results)
    total_lwbs = sum(r["lwbs"] for r in results)
    total_mortality = sum(r["mortality"] for r in results)
    
    for r in results:
        print(f"\nEpisode {r['episode']}:")
        print(f"  Reward: {r['total_reward']:.3f}")
        print(f"  Steps: {r['steps']}")
        print(f"  Patients: {r['arrivals']}")
        print(f"  LWBS: {r['lwbs']}")
        print(f"  Mortality: {r['mortality']}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL OVER 3 EPISODES:")
    print(f"  Average Reward: {total_reward/3:.3f}")
    print(f"  Total Patients: {total_arrivals}")
    if total_arrivals > 0:
        print(f"  LWBS Rate: {total_lwbs/total_arrivals:.1%}")
        print(f"  Mortality Rate: {total_mortality/total_arrivals:.1%}")
    print(f"{'='*60}")
    print("\n[OK] Baseline inference complete!")
    print("[OK] Environment ready for submission.")

if __name__ == "__main__":
    main()