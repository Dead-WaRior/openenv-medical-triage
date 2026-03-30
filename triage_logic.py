"""
Clinical decision support logic for ESI triage
"""

from .models import Patient, ChiefComplaint, ESILevel


class ESIGuidelines:
    """Implementation of ESI triage algorithm"""
    
    @classmethod
    def calculate_esi(cls, patient: Patient) -> ESILevel:
        """Calculate ESI level based on clinical presentation"""
        
        # Level 1: Critical
        if patient.chief_complaint in [
            ChiefComplaint.UNRESPONSIVE,
            ChiefComplaint.SEVERE_BLEEDING,
            ChiefComplaint.SEIZURE
        ]:
            return ESILevel.RESUSCITATION
        
        # Level 2: High risk
        if patient.chief_complaint in [
            ChiefComplaint.CHEST_PAIN,
            ChiefComplaint.STROKE_SYMPTOMS,
            ChiefComplaint.HEAD_INJURY
        ]:
            return ESILevel.EMERGENT
        
        # Level 3: Urgent (needs resources)
        if patient.chief_complaint in [
            ChiefComplaint.ABDOMINAL_PAIN,
            ChiefComplaint.SHORTNESS_OF_BREATH
        ]:
            return ESILevel.URGENT
        
        # Level 4-5: Non-urgent
        if patient.chief_complaint == ChiefComplaint.OTHER:
            return ESILevel.NON_URGENT
        
        return ESILevel.SEMI_URGENT


class ClinicalDeteriorationPredictor:
    """Predict patient deterioration"""
    
    @staticmethod
    def risk_score(patient: Patient) -> float:
        """Calculate deterioration risk score (0-1)"""
        score = 0.0
        
        if patient.age > 75:
            score += 0.3
        elif patient.age > 65:
            score += 0.2
        
        if patient.chief_complaint in [
            ChiefComplaint.CHEST_PAIN,
            ChiefComplaint.STROKE_SYMPTOMS
        ]:
            score += 0.2
        
        return min(score, 1.0)