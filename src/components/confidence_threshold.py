# src/components/confidence_threshold.py
from typing import Dict, Any, Tuple
from enum import Enum

from src.logger import get_logger

logger = get_logger(__name__)

class ConfidenceLevel(Enum):
    HIGH = "High Confidence"
    MODERATE = "Moderate Confidence"
    LOW = "Low Confidence"
    INCONCLUSIVE = "Inconclusive"

class ConfidenceThresholdEnforcer:
    """Enforce confidence thresholds for clinical safety"""
    
    def __init__(self, high_threshold: float = 0.80, 
                 moderate_threshold: float = 0.60,
                 low_threshold: float = 0.40):
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.low_threshold = low_threshold
    
    def evaluate(self, probability: float) -> Tuple[ConfidenceLevel, bool]:
        """
        Evaluate confidence level and determine if human review is required
        
        Returns:
            ConfidenceLevel, requires_human_review
        """
        if probability >= self.high_threshold:
            return ConfidenceLevel.HIGH, False
        elif probability >= self.moderate_threshold:
            return ConfidenceLevel.MODERATE, True
        elif probability >= self.low_threshold:
            return ConfidenceLevel.LOW, True
        else:
            return ConfidenceLevel.INCONCLUSIVE, True
    
    def get_action(self, confidence: ConfidenceLevel) -> Dict[str, Any]:
        """Get clinical action based on confidence level"""
        actions = {
            ConfidenceLevel.HIGH: {
                'action': 'APPROVE',
                'message': 'High confidence prediction - safe to act',
                'requires_review': False,
                'priority': 'LOW'
            },
            ConfidenceLevel.MODERATE: {
                'action': 'REVIEW',
                'message': 'Moderate confidence - physician review recommended',
                'requires_review': True,
                'priority': 'MEDIUM'
            },
            ConfidenceLevel.LOW: {
                'action': 'REVIEW_URGENT',
                'message': 'Low confidence - immediate physician review required',
                'requires_review': True,
                'priority': 'HIGH'
            },
            ConfidenceLevel.INCONCLUSIVE: {
                'action': 'REJECT',
                'message': 'Inconclusive - cannot make clinical decision',
                'requires_review': True,
                'priority': 'CRITICAL'
            }
        }
        
        return actions.get(confidence, {
            'action': 'REJECT',
            'message': 'Unknown confidence level',
            'requires_review': True,
            'priority': 'CRITICAL'
        })
    
    def create_review_ticket(self, patient: Dict, prediction: Dict,
                           confidence: ConfidenceLevel) -> Dict:
        """Create a review ticket for low confidence predictions"""
        return {
            'ticket_id': f"REVIEW_{int(time.time())}_{patient.get('patient_id', 'unknown')}",
            'patient': patient,
            'prediction': prediction,
            'confidence': confidence.value,
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING',
            'priority': self.get_action(confidence)['priority'],
            'action_required': self.get_action(confidence)['action'],
            'message': self.get_action(confidence)['message']
        }