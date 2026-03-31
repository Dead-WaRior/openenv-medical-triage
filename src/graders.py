"""
Task graders for Medical Triage Environment
Each grader returns a score between 0.0 and 1.0
"""

def grade_easy_task(episode_history):
    """
    Easy Task: Basic ESI Triage Accuracy
    Scores based on correct ESI assignments
    """
    if not episode_history:
        return 0.0
    
    correct = 0
    total = 0
    
    for step in episode_history:
        if 'action' in step and step['action']:
            if hasattr(step['action'], 'esi_level') and step['action'].esi_level:
                total += 1
                # In real implementation, compare with correct ESI
                # For now, assume some assignments are correct
                if step['action'].esi_level in [2, 3]:  # Most common correct levels
                    correct += 1
    
    if total == 0:
        return 0.0
    
    # Accuracy score (70%) + timeliness (30%)
    accuracy = (correct / total) * 0.7
    timeliness = 0.3  # Assume timely for now
    
    return min(1.0, accuracy + timeliness)


def grade_medium_task(episode_history):
    """
    Medium Task: Resource Allocation Under Pressure
    Scores based on LWBS rate and wait times
    """
    if not episode_history:
        return 0.0
    
    total_patients = 0
    total_lwbs = 0
    
    for step in episode_history:
        if 'info' in step and 'metrics' in step['info']:
            total_patients = step['info']['metrics']['total_arrivals']
            total_lwbs = step['info']['metrics']['total_lwbs']
    
    if total_patients == 0:
        return 0.0
    
    # LWBS rate (lower is better)
    lwbs_rate = total_lwbs / total_patients
    lwbs_score = max(0, 1 - lwbs_rate * 10) * 0.5
    
    # Resource utilization (assume 70% for medium)
    resource_score = 0.35
    
    return min(1.0, lwbs_score + resource_score)


def grade_hard_task(episode_history):
    """
    Hard Task: Mass Casualty Incident
    Scores based on mortality rate and triage accuracy under pressure
    """
    if not episode_history:
        return 0.0
    
    total_patients = 0
    total_mortality = 0
    
    for step in episode_history:
        if 'info' in step and 'metrics' in step['info']:
            total_patients = step['info']['metrics']['total_arrivals']
            total_mortality = step['info']['metrics']['total_mortality']
    
    if total_patients == 0:
        return 0.0
    
    # Mortality rate (lower is better)
    mortality_rate = total_mortality / total_patients
    mortality_score = max(0, 1 - mortality_rate * 20) * 0.6
    
    # Triage speed under pressure
    speed_score = 0.3 if len(episode_history) > 20 else 0.15
    
    return min(1.0, mortality_score + speed_score)