# src/components/human_review.py
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

class HumanReviewSystem:
    """
    Human-in-the-loop system for low confidence predictions
    Flags predictions requiring human review
    """
    
    def __init__(self):
        self.review_queue = []
        self.review_history = []
        self.low_confidence_threshold = 0.3
        self.auto_review_timeout = 3600  # 1 hour
        
        # Load existing queue
        self._load_review_queue()
    
    def _load_review_queue(self):
        """Load existing review queue"""
        queue_path = Path("outputs/review_queue.json")
        if queue_path.exists():
            with open(queue_path, 'r') as f:
                self.review_queue = json.load(f)
            logger.info(f"Loaded {len(self.review_queue)} items from review queue")
    
    def _save_review_queue(self):
        """Save review queue"""
        queue_path = Path("outputs/review_queue.json")
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_path, 'w') as f:
            json.dump(self.review_queue, f, indent=2)
    
    def needs_review(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if a prediction needs human review
        """
        # Check confidence
        if prediction.get('confidence') == 'Low Confidence':
            return True
        
        # Check probability
        probability = prediction.get('probability', 0)
        if probability < self.low_confidence_threshold:
            return True
        
        # Check if it's a high risk case that needs verification
        if prediction.get('risk_status') == 'HIGH RISK':
            return True
        
        # Check for fallback
        if prediction.get('fallback', False):
            return True
        
        return False
    
    def add_to_review_queue(self, patient: Dict[str, Any], 
                           prediction: Dict[str, Any]) -> str:
        """Add a prediction to the review queue"""
        review_id = f"REV_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.review_queue)}"
        
        review_item = {
            'review_id': review_id,
            'patient': patient,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'priority': self._calculate_priority(prediction),
            'reviewer': None,
            'review_notes': None,
            'reviewed_at': None
        }
        
        self.review_queue.append(review_item)
        self._save_review_queue()
        
        logger.info(f"Added review item {review_id} to queue (Priority: {review_item['priority']})")
        return review_id
    
    def _calculate_priority(self, prediction: Dict) -> str:
        """Calculate priority based on risk and confidence"""
        risk = prediction.get('risk_status', 'LOW RISK')
        confidence = prediction.get('confidence', 'High Confidence')
        
        if risk == 'HIGH RISK' and confidence == 'Low Confidence':
            return 'CRITICAL'
        elif risk == 'HIGH RISK':
            return 'HIGH'
        elif confidence == 'Low Confidence' and risk != 'LOW RISK':
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_pending_reviews(self, priority: Optional[str] = None) -> List[Dict]:
        """Get pending reviews, optionally filtered by priority"""
        pending = [item for item in self.review_queue if item['status'] == 'pending']
        
        if priority:
            pending = [item for item in pending if item['priority'] == priority]
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        pending.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return pending
    
    def review_prediction(self, review_id: str, reviewer: str, 
                         notes: str, approved: bool) -> bool:
        """Review a prediction"""
        for item in self.review_queue:
            if item['review_id'] == review_id and item['status'] == 'pending':
                item['status'] = 'approved' if approved else 'rejected'
                item['reviewer'] = reviewer
                item['review_notes'] = notes
                item['reviewed_at'] = datetime.now().isoformat()
                
                # Move to history
                self.review_history.append(item)
                self.review_queue.remove(item)
                self._save_review_queue()
                
                logger.info(f"Review {review_id} completed by {reviewer}: {'APPROVED' if approved else 'REJECTED'}")
                return True
        
        logger.warning(f"Review {review_id} not found or already reviewed")
        return False
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get review statistics"""
        total_reviews = len(self.review_history) + len(self.review_queue)
        
        if not self.review_history:
            return {
                'total_reviews': total_reviews,
                'pending_count': len(self.review_queue),
                'reviewed_count': 0,
                'message': 'No completed reviews'
            }
        
        df = pd.DataFrame(self.review_history)
        
        return {
            'total_reviews': total_reviews,
            'pending_count': len(self.review_queue),
            'reviewed_count': len(self.review_history),
            'approval_rate': len(df[df['status'] == 'approved']) / len(df) * 100,
            'avg_review_time': self._calculate_avg_review_time(),
            'priority_distribution': df['priority'].value_counts().to_dict(),
            'recent_reviews': len(df.tail(100))
        }
    
    def _calculate_avg_review_time(self) -> float:
        """Calculate average review time in minutes"""
        if not self.review_history:
            return 0
        
        times = []
        for item in self.review_history:
            if item['reviewed_at'] and item['timestamp']:
                start = datetime.fromisoformat(item['timestamp'])
                end = datetime.fromisoformat(item['reviewed_at'])
                times.append((end - start).total_seconds() / 60)
        
        return sum(times) / len(times) if times else 0
    
    def get_pending_count(self) -> int:
        """Get number of pending reviews"""
        return len([item for item in self.review_queue if item['status'] == 'pending'])