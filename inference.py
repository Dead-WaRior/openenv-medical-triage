"""Submission inference script with strict structured logs."""

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from src.environment import MedicalTriageEnv
from src.graders import grade_easy_task, grade_hard_task, grade_medium_task
from src.models import TriageAction


TASK_CONFIG = {
    "easy": {
        "max_steps": 30,
        "max_patients": 20,
        "random_seed": 42,
        "grader": grade_easy_task,
    },
    "medium": {
        "max_steps": 60,
        "max_patients": 40,
        "random_seed": 7,
        "grader": grade_medium_task,
    },
    "hard": {
        "max_steps": 100,
        "max_patients": 60,
        "random_seed": 99,
        "grader": grade_hard_task,
    },
}


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


MAX_RUNTIME_SECONDS = _get_int_env("MAX_RUNTIME_SECONDS", 1080)
MAX_TASK_RUNTIME_SECONDS = _get_int_env("MAX_TASK_RUNTIME_SECONDS", 360)
MAX_LLM_CALLS_PER_TASK = _get_int_env("MAX_LLM_CALLS_PER_TASK", 45)
LLM_EVERY_N_STEPS = max(1, _get_int_env("LLM_EVERY_N_STEPS", 3))
LLM_TIMEOUT_SECONDS = _get_float_env("LLM_TIMEOUT_SECONDS", 12.0)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_required_env() -> Tuple[str, str, str]:
    """Read and validate required submission environment variables."""
    api_base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1").strip()
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3").strip()
    # API_KEY and HF_TOKEN are interchangeable based on sample
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    hf_token = hf_token.strip() if hf_token else ""

    if not hf_token:
        # We must log an END event, but without task info it's minimal. We will just print failure and exit.
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
        print("Missing required environment variable: HF_TOKEN", file=sys.stderr)
        raise SystemExit(1)

    return api_base_url, model_name, hf_token


def build_prompt(observation) -> Optional[Tuple[str, str, list, list]]:
    """Build compact triage prompt for the first waiting patient."""
    if not observation.waiting_patients:
        return None

    patient = observation.waiting_patients[0]
    available_rooms = observation.available_rooms[:3]
    available_doctors = list(observation.available_doctors.keys())[:3]
    vitals = {k.value: v for k, v in patient.vital_signs.items()}

    prompt = (
        "You are an emergency triage nurse. "
        "Return ONLY valid JSON with keys: esi_level (1-5), initiate_resuscitation (bool).\n"
        f"patient_id={patient.id}\n"
        f"age={patient.age}\n"
        f"chief_complaint={patient.chief_complaint.value}\n"
        f"triage_note={patient.triage_note}\n"
        f"vitals={json.dumps(vitals, separators=(',', ':'))}\n"
        f"conditions={json.dumps(patient.conditions, separators=(',', ':'))}\n"
        f"available_rooms={json.dumps(available_rooms, separators=(',', ':'))}\n"
        f"available_doctors={json.dumps(available_doctors, separators=(',', ':'))}\n"
    )
    return patient.id, prompt, available_rooms, available_doctors


def llm_triage_action(observation, client: OpenAI, model_name: str) -> Optional[TriageAction]:
    """Select one triage action via OpenAI-compatible chat completions."""
    prompt_data = build_prompt(observation)
    if prompt_data is None:
        return None

    patient_id, prompt, available_rooms, available_doctors = prompt_data
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=80,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    esi_level = int(data.get("esi_level", 3))
    esi_level = max(1, min(5, esi_level))
    room = available_rooms[0] if available_rooms else None
    doctor = available_doctors[0] if available_doctors else None

    return TriageAction(
        patient_id=patient_id,
        esi_level=esi_level,
        assigned_room=room,
        assigned_doctor_id=doctor,
        order_tests=[],
        initiate_resuscitation=bool(data.get("initiate_resuscitation", esi_level == 1)),
    )


def heuristic_triage_action(observation) -> Optional[TriageAction]:
    """Low-cost fallback action to preserve runtime and avoid hard failures."""
    if not observation.waiting_patients:
        return None

    patient = observation.waiting_patients[0]
    complaint = patient.chief_complaint.value
    if complaint in ("unresponsive", "severe_bleeding", "seizure"):
        esi_level = 1
    elif complaint in ("chest_pain", "stroke_symptoms", "head_injury", "altered_mental_status"):
        esi_level = 2
    elif complaint in ("shortness_of_breath", "abdominal_pain", "fever"):
        esi_level = 3
    else:
        esi_level = 4

    room = observation.available_rooms[0] if observation.available_rooms else None
    doctors = list(observation.available_doctors.keys())
    doctor = doctors[0] if doctors else None

    return TriageAction(
        patient_id=patient.id,
        esi_level=esi_level,
        assigned_room=room,
        assigned_doctor_id=doctor,
        order_tests=[],
        initiate_resuscitation=(esi_level == 1),
    )


def run_task(
    task_name: str,
    config: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    global_deadline: float,
) -> Tuple[float, Dict[str, Any]]:
    """Run one task episode and emit per-step structured logs."""
    env = MedicalTriageEnv(
        max_steps=config["max_steps"],
        max_patients=config["max_patients"],
        random_seed=config["random_seed"],
    )
    observation = env.reset()
    episode_history = []
    total_reward = 0.0
    last_info: Dict[str, Any] = {"metrics": {"total_arrivals": 0, "total_lwbs": 0, "total_mortality": 0}}
    task_deadline = min(global_deadline, time.monotonic() + MAX_TASK_RUNTIME_SECONDS)
    llm_calls = 0

    log_start(task=task_name, env="medical-triage", model=model_name)

    rewards_list = []

    for step_index in range(config["max_steps"]):
        if time.monotonic() >= task_deadline:
            log_step(step=step_index + 1, action="time_budget_reached", reward=0.00, done=True, error=None)
            rewards_list.append(0.0)
            break

        use_llm = (step_index % LLM_EVERY_N_STEPS == 0) and (llm_calls < MAX_LLM_CALLS_PER_TASK)
        error_msg = None
        if use_llm:
            try:
                action = llm_triage_action(observation, client, model_name)
                llm_calls += 1
            except Exception as e:
                action = heuristic_triage_action(observation)
                error_msg = str(e)
        else:
            action = heuristic_triage_action(observation)

        if action is None:
            log_step(step=step_index + 1, action="no_waiting_patients", reward=0.00, done=True, error=None)
            rewards_list.append(0.0)
            break

        patient = env.patients.get(action.patient_id)
        observation, reward, done, info = env.step(action)
        last_info = info
        total_reward += reward.total

        episode_history.append(
            {
                "step": step_index,
                "action": action,
                "patient": patient,
                "reward": reward,
                "info": info,
            }
        )

        rewards_list.append(reward.total)
        action_str = json.dumps({"patient_id": action.patient_id, "esi": action.esi_level}, separators=(",", ":"))
        log_step(step=step_index + 1, action=action_str, reward=reward.total, done=bool(done), error=error_msg)

        if done:
            break

    score = config["grader"](episode_history)
    # Define success loosely as score crossing target or just > 0.0
    target_score = TASK_CONFIG[task_name].get("target", 0.0)
    success = score >= target_score if target_score else score > 0.0

    log_end(success=success, steps=len(episode_history), score=float(score), rewards=rewards_list)

    metrics = last_info.get("metrics", {})
    summary = {
        "task": task_name,
        "steps": len(episode_history),
        "score": round(float(score), 6),
        "total_reward": round(float(total_reward), 6),
        "arrivals": int(metrics.get("total_arrivals", 0)),
        "lwbs": int(metrics.get("total_lwbs", 0)),
        "mortality": int(metrics.get("total_mortality", 0)),
    }
    return score, summary


def main() -> None:
    """Submission entry point."""
    api_base_url, model_name, hf_token = get_required_env()
    client = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=LLM_TIMEOUT_SECONDS)
    global_deadline = time.monotonic() + MAX_RUNTIME_SECONDS

    scores: Dict[str, float] = {}
    summaries = []
    try:
        for task_name in ("easy", "medium", "hard"):
            score, summary = run_task(task_name, TASK_CONFIG[task_name], client, model_name, global_deadline)
            scores[task_name] = float(score)
            summaries.append(summary)

            if time.monotonic() >= global_deadline:
                break
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
        print(f"Runtime exception: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()