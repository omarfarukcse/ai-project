import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

from src.logger import get_logger
from src.config_manager import config_manager
from src.components.confidence_threshold import ConfidenceThresholdEnforcer

logger = get_logger(__name__)

class ClinicalDecisionEngine:
    def __init__(self):
        self.config = config_manager.config
        self.clinical_config = config_manager.get_validated('clinical')
        self.risk_threshold = self.clinical_config.high_risk_threshold
        self.moderate_threshold = self.clinical_config.moderate_risk_threshold
        self.confidence_enforcer = ConfidenceThresholdEnforcer()
        self.model = None
        self.preprocessor = None
    
    def initialize(self, model: Any, preprocessor: Any):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, patient_data: pd.DataFrame) -> Dict[str, Any]:
        patient_dict = patient_data.iloc[0].to_dict()
        patient_array = self.preprocessor.transform(patient_data.values)
        probability = self.model.predict_proba(patient_array)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        confidence_level, requires_review = self.confidence_enforcer.evaluate(probability)
        
        if probability >= self.risk_threshold:
            risk_status = "HIGH RISK"
            recommendation = "🚨 URGENT: Immediate clinical evaluation required"
            actions = ["Immediate clinical assessment", "Confirm with additional tests", "Consider specialist consultation"]
        elif probability >= self.moderate_threshold:
            risk_status = "MODERATE RISK"
            recommendation = "⚠️ Clinical evaluation recommended"
            actions = ["Schedule clinical assessment", "Review risk factors", "Consider preventive interventions"]
        else:
            risk_status = "LOW RISK"
            recommendation = "✅ Continue routine monitoring"
            actions = ["Continue preventive monitoring", "Maintain healthy lifestyle", "Schedule routine follow-up"]
        
        if requires_review:
            actions.append("🔍 Human review recommended")
        
        return {
            'risk_status': risk_status,
            'probability': float(probability),
            'prediction': 'Disease Present' if prediction == 1 else 'No Disease',
            'recommendation': recommendation,
            'actions': actions,
            'confidence': confidence_level.value,
            'requires_human_review': requires_review,
            'decision_timestamp': datetime.now().isoformat()
        }