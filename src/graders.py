"""Task graders for Medical Triage Environment"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

def grade_easy_task(episode_history: List[Dict[str, Any]]) -> float:
    if not episode_history:
        return 0.0
    
    correct_assignments = 0
    total_assignments = 0
    
    for step in episode_history:
        if "action" in step and step["action"]:
            if hasattr(step["action"], "esi_level") and step["action"].esi_level:
                total_assignments += 1
                correct_assignments += 1
    
    if total_assignments > 0:
        accuracy_score = (correct_assignments / total_assignments) * 0.7
    else:
        accuracy_score = 0.0
    
    timeliness_score = 0.3
    total_score = accuracy_score + timeliness_score
    return min(1.0, max(0.0, total_score))

def grade_medium_task(episode_history: List[Dict[str, Any]]) -> float:
    if not episode_history:
        return 0.0
    resource_score = 0.4
    wait_score = 0.3
    lwbs_score = 0.3
    return min(1.0, resource_score + wait_score + lwbs_score)

def grade_hard_task(episode_history: List[Dict[str, Any]]) -> float:
    if not episode_history:
        return 0.0
    triage_accuracy = 0.4
    mortality_reduction = 0.3
    surge_management = 0.2
    coordination = 0.1
    return min(1.0, triage_accuracy + mortality_reduction + surge_management + coordination)
