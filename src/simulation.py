"""Patient simulation and generation"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import numpy as np
from .models import Patient, ChiefComplaint, VitalSign, ResourceType, DoctorSpecialty


class PatientGenerator:
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.complaint_distribution = {
            ChiefComplaint.CHEST_PAIN: 0.12,
            ChiefComplaint.SHORTNESS_OF_BREATH: 0.10,
            ChiefComplaint.ABDOMINAL_PAIN: 0.15,
            ChiefComplaint.FEVER: 0.12,
            ChiefComplaint.HEAD_INJURY: 0.05,
            ChiefComplaint.FRACTURE: 0.08,
            ChiefComplaint.BURN: 0.03,
            ChiefComplaint.SEIZURE: 0.04,
            ChiefComplaint.ALTERED_MENTAL_STATUS: 0.06,
            ChiefComplaint.STROKE_SYMPTOMS: 0.05,
            ChiefComplaint.SEVERE_BLEEDING: 0.03,
            ChiefComplaint.ALLERGIC_REACTION: 0.04,
            ChiefComplaint.UNRESPONSIVE: 0.02,
            ChiefComplaint.OTHER: 0.11,
        }
    
    def generate_patient(self, current_time: datetime) -> Patient:
        complaints = list(self.complaint_distribution.keys())
        weights = list(self.complaint_distribution.values())
        chief_complaint = random.choices(complaints, weights=weights)[0]
        
        if random.random() < 0.3:
            age = random.randint(65, 95)
        else:
            age = random.randint(18, 50)
        
        vital_signs = self._generate_vital_signs(chief_complaint, age)
        triage_note = self._generate_triage_note(chief_complaint)
        conditions = self._generate_conditions(age)
        allergies = self._generate_allergies()
        
        return Patient(
            id=f"PAT_{uuid.uuid4().hex[:8]}",
            arrival_time=current_time,
            age=age,
            chief_complaint=chief_complaint,
            triage_note=triage_note,
            vital_signs=vital_signs,
            conditions=conditions,
            allergies=allergies,
            medications=[],
            ordered_tests=[]
        )
    
    def generate_batch(self, count: int, current_time: datetime) -> List[Patient]:
        patients = []
        for i in range(count):
            arrival_time = current_time + timedelta(minutes=i * random.randint(1, 3))
            patient = self.generate_patient(arrival_time)
            patients.append(patient)
        return patients
    
    def _generate_vital_signs(self, complaint: ChiefComplaint, age: int) -> Dict[VitalSign, float]:
        vital_signs = {
            VitalSign.HEART_RATE: random.uniform(60, 100),
            VitalSign.BLOOD_PRESSURE_SYS: random.uniform(110, 130),
            VitalSign.BLOOD_PRESSURE_DIA: random.uniform(70, 85),
            VitalSign.RESPIRATORY_RATE: random.uniform(12, 18),
            VitalSign.OXYGEN_SATURATION: random.uniform(95, 100),
            VitalSign.TEMPERATURE: random.uniform(36.5, 37.2),
            VitalSign.GCS: 15.0,
        }
        return vital_signs
    
    def _generate_triage_note(self, complaint: ChiefComplaint) -> str:
        notes = {
            ChiefComplaint.CHEST_PAIN: "Chest pain radiating to left arm, diaphoretic",
            ChiefComplaint.SHORTNESS_OF_BREATH: "Difficulty breathing, wheezing",
            ChiefComplaint.ABDOMINAL_PAIN: "Severe abdominal pain, nausea",
            ChiefComplaint.FEVER: "Fever, chills, productive cough",
        }
        return notes.get(complaint, f"Patient presents with {complaint.value}")
    
    def _generate_conditions(self, age: int) -> List[str]:
        conditions = []
        if age > 60 and random.random() < 0.6:
            conditions.append("hypertension")
        if age > 50 and random.random() < 0.4:
            conditions.append("diabetes")
        return conditions
    
    def _generate_allergies(self) -> List[str]:
        if random.random() < 0.15:
            return [random.choice(["penicillin", "sulfa", "latex"])]
        return []


class ResourceManager:
    def __init__(self):
        self.rooms = {
            "trauma_1": {"type": "trauma", "available": True},
            "trauma_2": {"type": "trauma", "available": True},
            "bed_1": {"type": "general", "available": True},
            "bed_2": {"type": "general", "available": True},
            "bed_3": {"type": "general", "available": True},
            "bed_4": {"type": "general", "available": True},
            "bed_5": {"type": "general", "available": True},
        }
        
        self.doctors = {
            "dr_smith": {"specialty": DoctorSpecialty.EMERGENCY, "available": True},
            "dr_jones": {"specialty": DoctorSpecialty.TRAUMA, "available": True},
            "dr_lee": {"specialty": DoctorSpecialty.CARDIOLOGY, "available": True},
            "dr_patel": {"specialty": DoctorSpecialty.NEUROLOGY, "available": True},
            "dr_wong": {"specialty": DoctorSpecialty.CRITICAL_CARE, "available": True},
        }
        
        self.equipment = {
            ResourceType.CT_SCANNER: 2,
            ResourceType.MRI: 1,
            ResourceType.ULTRASOUND: 2,
            ResourceType.CARDIAC_MONITOR: 5,
            ResourceType.VENTILATOR: 3,
        }
    
    @property
    def available_rooms(self) -> List[str]:
        return [room_id for room_id, room in self.rooms.items() if room["available"]]
    
    @property
    def available_doctors(self) -> Dict[str, DoctorSpecialty]:
        return {doc_id: doc["specialty"] for doc_id, doc in self.doctors.items() if doc["available"]}
    
    def assign_room(self, room_id: str) -> bool:
        if room_id in self.rooms and self.rooms[room_id]["available"]:
            self.rooms[room_id]["available"] = False
            return True
        return False
    
    def assign_doctor(self, doctor_id: str) -> bool:
        if doctor_id in self.doctors and self.doctors[doctor_id]["available"]:
            self.doctors[doctor_id]["available"] = False
            return True
        return False
    
    def free_room(self, room_id: str) -> None:
        if room_id in self.rooms:
            self.rooms[room_id]["available"] = True
    
    def free_doctor(self, doctor_id: str) -> None:
        if doctor_id in self.doctors:
            self.doctors[doctor_id]["available"] = True
