# src/components/fallback_system.py
"""
Clinical Fallback System with Rule-Based Decisions
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.config_manager import get_config_manager

logger = get_logger(__name__)


class FallbackSystem:
    """
    Clinical Fallback System with Rule-Based Decision Making
    
    Features:
    - Clinical rule-based predictions when ML unavailable
    - Risk scoring based on clinical guidelines
    - Graceful degradation
    - Override support
    - Audit logging
    - Monitoring integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/fallback_rules.json"
        self.rules = self._load_rules()
        self._fallback_count = 0
        self._fallback_reasons: Dict[str, int] = {}
        
        logger.info("🔄 FallbackSystem initialized")
        logger.info(f"   Rules loaded: {self.config_path}")
        logger.info(f"   Enabled: {self.rules.get('enabled', True)}")
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load fallback rules from config"""
        try:
            config_manager = get_config_manager()
            rules = config_manager.load_config("fallback_rules")
            
            # If not found, try direct file
            if not rules:
                path = Path(self.config_path)
                if path.exists():
                    with open(path, 'r') as f:
                        rules = json.load(f)
                else:
                    logger.warning("Fallback rules not found, using defaults")
                    rules = self._get_default_rules()
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to load fallback rules: {str(e)}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default fallback rules"""
        return {
            "enabled": True,
            "triggers": {
                "model_unavailable": {"enabled": True, "priority": 1}
            },
            "clinical_rules": {
                "diabetes": {
                    "high_risk_factors": {
                        "glucose": {"threshold": 126, "weight": 3},
                        "bmi": {"threshold": 30, "weight": 2},
                        "age": {"threshold": 45, "weight": 1}
                    },
                    "risk_scoring": {
                        "high_risk": {"score_threshold": 5, "risk_level": "High Risk"},
                        "moderate_risk": {"score_threshold": 3, "risk_level": "Moderate Risk"},
                        "low_risk": {"score_threshold": 0, "risk_level": "Low Risk"}
                    }
                }
            },
            "fallback_decisions": {
                "default_prediction": {
                    "risk_score": 50,
                    "risk_level": "Moderate Risk",
                    "confidence": 0.0,
                    "reason": "ML model unavailable - using clinical rule-based fallback"
                },
                "safe_fallback": {
                    "enabled": True,
                    "defaults": {
                        "glucose": 100,
                        "bmi": 25,
                        "age": 50,
                        "blood_pressure": 75
                    }
                }
            },
            "audit": {
                "enabled": True,
                "log_all_fallbacks": True,
                "retention_days": 90
            }
        }
    
    # ============================================================================
    # 🚀 Fallback Prediction
    # ============================================================================
    
    def predict(
        self,
        data: Dict[str, Any],
        trigger_reason: str = "model_unavailable",
        override_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make fallback prediction using clinical rules
        
        Args:
            data: Patient data
            trigger_reason: Why fallback was triggered
            override_data: Manual override data
            
        Returns:
            Prediction result with risk assessment
        """
        
        if not self.rules.get("enabled", True):
            logger.warning("Fallback system disabled")
            return {
                "risk_score": 50,
                "risk_level": "Moderate Risk",
                "confidence": 0.0,
                "reason": "Fallback system disabled",
                "fallback_used": False,
            }
        
        # Increment counter
        self._fallback_count += 1
        self._fallback_reasons[trigger_reason] = (
            self._fallback_reasons.get(trigger_reason, 0) + 1
        )
        
        # Check for override
        if override_data:
            result = self._apply_override(override_data)
            if result:
                logger.info(f"✅ Fallback override applied for {trigger_reason}")
                return result
        
        # Apply safe defaults for missing data
        data = self._apply_safe_defaults(data)
        
        # Determine dataset type
        dataset_type = self._detect_dataset_type(data)
        
        # Calculate risk score
        result = self._calculate_risk_score(data, dataset_type)
        
        # Apply conservative adjustment
        if self.rules.get("fallback_decisions", {}).get("conservative_fallback", {}).get("enabled", True):
            result = self._apply_conservative_bump(result)
        
        # Add fallback metadata
        result.update({
            "fallback_used": True,
            "fallback_reason": trigger_reason,
            "fallback_timestamp": datetime.now().isoformat(),
            "fallback_version": self.rules.get("version", "1.0.0"),
        })
        
        # Log fallback
        self._log_fallback(data, result, trigger_reason)
        
        return result
    
    # ============================================================================
    # 🔧 Risk Score Calculation
    # ============================================================================
    
    def _detect_dataset_type(self, data: Dict[str, Any]) -> str:
        """Detect which dataset type the data belongs to"""
        
        diabetes_features = ["glucose", "bmi", "pregnancies", "insulin", "diabetes_pedigree"]
        heart_features = ["chol", "trestbps", "thalach", "exang", "oldpeak"]
        
        diabetes_count = sum(1 for f in diabetes_features if f in data)
        heart_count = sum(1 for f in heart_features if f in data)
        
        if diabetes_count > heart_count:
            return "diabetes"
        elif heart_count > diabetes_count:
            return "heart_disease"
        else:
            # Default to diabetes if equal
            return "diabetes"
    
    def _calculate_risk_score(self, data: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """Calculate risk score using clinical rules"""
        
        clinical_rules = self.rules.get("clinical_rules", {}).get(dataset_type, {})
        risk_factors = clinical_rules.get("high_risk_factors", {})
        risk_scoring = clinical_rules.get("risk_scoring", {})
        
        total_score = 0
        factor_details = []
        
        for factor_name, factor_config in risk_factors.items():
            if factor_name in data:
                value = data[factor_name]
                threshold = factor_config.get("threshold", 0)
                weight = factor_config.get("weight", 1)
                
                # Check if value exceeds threshold
                if value >= threshold:
                    total_score += weight
                    factor_details.append({
                        "factor": factor_name,
                        "value": value,
                        "threshold": threshold,
                        "weight": weight,
                        "status": "elevated"
                    })
                else:
                    factor_details.append({
                        "factor": factor_name,
                        "value": value,
                        "threshold": threshold,
                        "weight": weight,
                        "status": "normal"
                    })
        
        # Determine risk level
        risk_level = "Low Risk"
        risk_action = "Maintain healthy lifestyle"
        recommendations = []
        
        for level, config in risk_scoring.items():
            threshold = config.get("score_threshold", 0)
            if total_score >= threshold:
                risk_level = config.get("risk_level", "Low Risk")
                risk_action = config.get("action", "")
                recommendations = config.get("recommendations", [])
                break
        
        # Calculate confidence based on number of risk factors evaluated
        confidence = min(0.9, 0.5 + (len(factor_details) / 20))
        
        return {
            "risk_score": min(100, 30 + total_score * 10),  # Scale to 0-100
            "risk_level": risk_level,
            "confidence": confidence,
            "action": risk_action,
            "recommendations": recommendations,
            "total_score": total_score,
            "factor_details": factor_details,
            "dataset_type": dataset_type,
        }
    
    def _apply_safe_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safe defaults for missing data"""
        
        safe_defaults = self.rules.get("fallback_decisions", {}).get("safe_fallback", {}).get("defaults", {})
        result = data.copy()
        
        for key, default_value in safe_defaults.items():
            if key not in result or result[key] is None:
                result[key] = default_value
                logger.debug(f"Applied safe default for {key}: {default_value}")
        
        return result
    
    def _apply_conservative_bump(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative risk bump for patient safety"""
        
        conservative = self.rules.get("fallback_decisions", {}).get("conservative_fallback", {})
        risk_bump = conservative.get("risk_bump", 15)
        adjustment_map = conservative.get("risk_level_adjustment", {})
        
        # Bump risk score
        result["risk_score"] = min(100, result.get("risk_score", 50) + risk_bump)
        
        # Adjust risk level
        current_level = result.get("risk_level", "Low Risk")
        if current_level in adjustment_map:
            result["risk_level"] = adjustment_map[current_level]
        
        return result
    
    def _apply_override(self, override_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply manual override"""
        
        if not override_data:
            return None
        
        # Validate override
        risk_score = override_data.get("risk_score")
        risk_level = override_data.get("risk_level")
        
        if risk_score is not None and risk_level is not None:
            return {
                "risk_score": min(100, max(0, risk_score)),
                "risk_level": risk_level,
                "confidence": 0.9,
                "override": True,
                "override_reason": override_data.get("reason", "Manual override"),
                "override_timestamp": datetime.now().isoformat(),
                "override_by": override_data.get("user_id", "unknown"),
            }
        
        return None
    
    # ============================================================================
    # 📊 Fallback Decision Functions
    # ============================================================================
    
    def should_fallback(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if fallback should be triggered
        
        Args:
            context: Context with model status, data quality, etc.
            
        Returns:
            Tuple of (should_fallback, reason)
        """
        
        triggers = self.rules.get("triggers", {})
        
        # Check model availability
        if context.get("model_available") is False:
            if triggers.get("model_unavailable", {}).get("enabled", True):
                return True, "model_unavailable"
        
        # Check timeout
        if context.get("timeout_occurred", False):
            if triggers.get("model_timeout", {}).get("enabled", True):
                return True, "model_timeout"
        
        # Check data validation
        if context.get("invalid_input", False):
            if triggers.get("invalid_input", {}).get("enabled", True):
                return True, "invalid_input"
        
        # Check confidence
        confidence = context.get("confidence", 1.0)
        confidence_threshold = self.rules.get("thresholds", {}).get("fallback_on_confidence_below", 0.5)
        if confidence < confidence_threshold:
            if triggers.get("high_uncertainty", {}).get("enabled", True):
                return True, "high_uncertainty"
        
        # Check circuit breaker
        if context.get("circuit_breaker_open", False):
            if triggers.get("circuit_breaker_open", {}).get("enabled", True):
                return True, "circuit_breaker_open"
        
        return False, "normal_operation"
    
    # ============================================================================
    # 📝 Audit and Logging
    # ============================================================================
    
    def _log_fallback(self, data: Dict, result: Dict, reason: str):
        """Log fallback usage for audit"""
        
        audit_config = self.rules.get("audit", {})
        
        if not audit_config.get("enabled", True):
            return
        
        if audit_config.get("log_all_fallbacks", True):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "data_keys": list(data.keys()),
                "result": {
                    "risk_score": result.get("risk_score"),
                    "risk_level": result.get("risk_level"),
                    "confidence": result.get("confidence"),
                },
                "fallback_count": self._fallback_count,
            }
            
            if audit_config.get("include_input_data", True):
                log_entry["data"] = data
            
            if audit_config.get("include_prediction", True):
                log_entry["prediction"] = result
            
            # Log to file
            log_file = self.rules.get("monitoring", {}).get("fallback_log_file", "outputs/logs/fallback.log")
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to log fallback: {str(e)}")
            
            logger.info(f"📝 Fallback logged: {reason} (Count: {self._fallback_count})")
    
    # ============================================================================
    # 📊 Statistics
    # ============================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        
        return {
            "total_fallbacks": self._fallback_count,
            "fallback_reasons": self._fallback_reasons,
            "enabled": self.rules.get("enabled", True),
            "config_version": self.rules.get("version", "1.0.0"),
        }
    
    def reset_stats(self):
        """Reset fallback statistics"""
        self._fallback_count = 0
        self._fallback_reasons = {}
        logger.info("🔄 Fallback statistics reset")
    
    def reload_rules(self):
        """Reload fallback rules from config"""
        self.rules = self._load_rules()
        logger.info("✅ Fallback rules reloaded")
    
    def get_fallback_recommendations(self, risk_level: str, dataset_type: str) -> List[str]:
        """Get recommendations based on risk level"""
        
        clinical_rules = self.rules.get("clinical_rules", {}).get(dataset_type, {})
        risk_scoring = clinical_rules.get("risk_scoring", {})
        
        for level, config in risk_scoring.items():
            if config.get("risk_level") == risk_level:
                return config.get("recommendations", [])
        
        # Default recommendations
        return [
            "Schedule clinical evaluation",
            "Review patient history",
            "Consider additional testing",
        ]


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_fallback_system: Optional[FallbackSystem] = None


def get_fallback_system() -> FallbackSystem:
    """Get fallback system singleton"""
    global _fallback_system
    if _fallback_system is None:
        _fallback_system = FallbackSystem()
    return _fallback_system


def fallback_predict(data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function for fallback prediction"""
    system = get_fallback_system()
    
    if context:
        should_fallback, reason = system.should_fallback(context)
        if should_fallback:
            return system.predict(data, reason)
    
    return system.predict(data, "manual_override")