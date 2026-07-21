# src/clinical/clinical_guidelines.py
"""
Clinical Guidelines Integration
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)


class ClinicalGuidelines:
    """
    Clinical Guidelines with:
    - Evidence-based recommendations
    - Multi-source integration
    - Version management
    - Decision support
    """
    
    def __init__(self, guidelines_path: Optional[str] = None):
        self.guidelines_path = guidelines_path or "config/clinical_guidelines.json"
        self.guidelines = self._load_guidelines()
        
        logger.info("📋 Clinical Guidelines loaded")
    
    def _load_guidelines(self) -> Dict:
        """Load guidelines from file or create default"""
        
        path = Path(self.guidelines_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        
        return self._create_default_guidelines()
    
    def _create_default_guidelines(self) -> Dict:
        """Create default clinical guidelines"""
        return {
            "diabetes": {
                "diagnostic_criteria": {
                    "fasting_glucose": "≥ 126 mg/dL",
                    "hba1c": "≥ 6.5%",
                    "oral_glucose_tolerance": "≥ 200 mg/dL",
                },
                "primary_treatment": "Lifestyle modifications and Metformin",
                "alternative_treatments": [
                    "SGLT2 inhibitors",
                    "GLP-1 agonists",
                    "DPP-4 inhibitors",
                ],
                "monitoring": {
                    "hba1c": "Every 3-6 months",
                    "blood_pressure": "Every visit",
                    "lipid_profile": "Annually",
                },
                "lifestyle": [
                    "Diet: Low glycemic index, portion control",
                    "Exercise: 150 min/week moderate activity",
                    "Weight: 5-10% reduction if overweight",
                ],
                "referral_criteria": [
                    "HbA1c > 9%",
                    "Frequent hypoglycemia",
                    "Complications present",
                ],
            },
            "heart_disease": {
                "diagnostic_criteria": {
                    "ecg": "ST-T wave changes",
                    "stress_test": "Positive for ischemia",
                    "cardiac_enzymes": "Elevated",
                },
                "primary_treatment": "Aspirin and Statins",
                "alternative_treatments": [
                    "Beta-blockers",
                    "ACE inhibitors",
                    "Calcium channel blockers",
                ],
                "monitoring": {
                    "ecg": "As needed",
                    "lipid_profile": "Every 3-6 months",
                    "blood_pressure": "Every visit",
                },
                "lifestyle": [
                    "Diet: Heart-healthy, low sodium",
                    "Exercise: Cardiac rehabilitation",
                    "Smoking cessation",
                ],
                "referral_criteria": [
                    "Unstable angina",
                    "Heart failure symptoms",
                    "Complex arrhythmias",
                ],
            },
            "hypertension": {
                "diagnostic_criteria": {
                    "systolic": "≥ 140 mmHg",
                    "diastolic": "≥ 90 mmHg",
                },
                "primary_treatment": "ACE inhibitors and lifestyle changes",
                "alternative_treatments": [
                    "ARBs",
                    "Calcium channel blockers",
                    "Diuretics",
                ],
                "monitoring": {
                    "blood_pressure": "Home monitoring daily",
                    "renal_function": "Annually",
                    "electrolytes": "As needed",
                },
                "lifestyle": [
                    "Diet: DASH diet",
                    "Exercise: 150 min/week",
                    "Reduce sodium to < 1500mg/day",
                ],
                "referral_criteria": [
                    "Resistant hypertension",
                    "Secondary causes",
                    "End-organ damage",
                ],
            },
        }
    
    def get_guidelines(self, condition: str) -> Dict:
        """Get guidelines for a specific condition"""
        
        condition_key = condition.lower()
        return self.guidelines.get(condition_key, {})
    
    def get_diagnostic_criteria(self, condition: str) -> Dict:
        """Get diagnostic criteria for a condition"""
        
        guidelines = self.get_guidelines(condition)
        return guidelines.get("diagnostic_criteria", {})
    
    def get_treatment(self, condition: str) -> Dict:
        """Get treatment recommendations"""
        
        guidelines = self.get_guidelines(condition)
        return {
            "primary": guidelines.get("primary_treatment"),
            "alternatives": guidelines.get("alternative_treatments", []),
        }
    
    def get_monitoring_plan(self, condition: str) -> Dict:
        """Get monitoring recommendations"""
        
        guidelines = self.get_guidelines(condition)
        return guidelines.get("monitoring", {})
    
    def get_lifestyle_recommendations(self, condition: str) -> List[str]:
        """Get lifestyle recommendations"""
        
        guidelines = self.get_guidelines(condition)
        return guidelines.get("lifestyle", [])
    
    def check_referral_criteria(self, condition: str) -> List[str]:
        """Get referral criteria"""
        
        guidelines = self.get_guidelines(condition)
        return guidelines.get("referral_criteria", [])
    
    def save_guidelines(self):
        """Save guidelines to file"""
        
        path = Path(self.guidelines_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.guidelines, f, indent=2)
        
        logger.info(f"✅ Guidelines saved to {path}")
    
    def update_guidelines(self, condition: str, updates: Dict):
        """Update guidelines for a condition"""
        
        if condition not in self.guidelines:
            self.guidelines[condition] = {}
        
        self.guidelines[condition].update(updates)
        self.save_guidelines()