# src/validation/adversarial_protection.py
"""
Advanced Adversarial Protection for Clinical AI Systems

This module provides defense-in-depth security for clinical ML systems:
- Input validation and sanitization
- Injection attack prevention
- Anomaly detection
- Adversarial example detection
- Data poisoning prevention
- Safe fallback mechanisms

Features:
    - Multiple validation layers
    - Statistical outlier detection
    - Pattern-based injection detection
    - Type enforcement and coercion
    - Range clamping with clinical constraints
    - Anomaly scoring
    - Batch processing protection
    - Configurable security levels

Version: 3.0.0
"""

import re
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EllipticEnvelope

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


# ============================================================================
# 📋 Enums and Constants
# ============================================================================

class ThreatLevel(Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(Enum):
    """Security response actions"""
    ALLOW = "allow"
    WARN = "warn"
    SANITIZE = "sanitize"
    BLOCK = "block"
    QUARANTINE = "quarantine"


@dataclass
class SecurityResult:
    """Security validation result"""
    is_safe: bool
    threat_level: ThreatLevel = ThreatLevel.NONE
    action_taken: SecurityAction = SecurityAction.ALLOW
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    sanitized_data: Dict[str, Any] = field(default_factory=dict)
    threat_signatures: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "threat_level": self.threat_level.value,
            "action_taken": self.action_taken.value,
            "warnings": self.warnings,
            "errors": self.errors,
            "sanitized_data": self.sanitized_data,
            "threat_signatures": self.threat_signatures,
            "timestamp": self.timestamp.isoformat(),
        }


class AdversarialProtection:
    """
    Advanced Adversarial Protection for Clinical ML Systems
    
    Features:
    - Multi-layer input validation
    - Injection attack detection
    - Adversarial example detection
    - Statistical anomaly detection
    - Data poisoning prevention
    - Safe fallback mechanisms
    - Threat intelligence integration
    - Audit trail
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Clinical ranges by dataset
        self.clinical_ranges = self._get_clinical_ranges()
        self.clinical_expected = self._get_expected_ranges()
        
        # Injection patterns (comprehensive list)
        self.injection_patterns = self._get_injection_patterns()
        
        # Anomaly detection models
        self._anomaly_model = None
        self._reference_data = None
        self._feature_names = None
        
        # Threat signatures
        self.threat_signatures: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "sanitized_requests": 0,
            "warned_requests": 0,
            "threats_detected": {},
            "last_reset": datetime.now(),
        }
        
        logger.info("🔒 AdversarialProtection initialized")
        logger.info(f"   Clinical fields: {len(self.clinical_ranges)}")
        logger.info(f"   Injection patterns: {len(self.injection_patterns)}")
        logger.info(f"   Security level: {self.config.get('security_level', 'HIGH')}")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "security_level": "HIGH",  # LOW, MEDIUM, HIGH, CRITICAL
            "block_on_injection": True,
            "block_on_outlier": False,
            "sanitize_inputs": True,
            "clamp_to_expected": True,
            "detect_anomalies": True,
            "anomaly_threshold": 3.0,
            "max_requests_per_second": 1000,
            "enable_audit_log": True,
        }
    
    def _get_clinical_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get clinical value ranges"""
        return {
            # Diabetes features
            "pregnancies": (0, 20),
            "glucose": (0, 300),
            "blood_pressure": (0, 200),
            "skin_thickness": (0, 100),
            "insulin": (0, 1000),
            "bmi": (10, 60),
            "diabetes_pedigree": (0, 3),
            "age": (0, 120),
            
            # Heart disease features
            "sex": (0, 1),
            "cp": (0, 3),
            "trestbps": (0, 300),
            "chol": (0, 600),
            "fbs": (0, 1),
            "restecg": (0, 2),
            "thalach": (50, 250),
            "exang": (0, 1),
            "oldpeak": (0, 10),
            "slope": (0, 2),
            "ca": (0, 4),
            "thal": (0, 3),
            
            # Additional safety fields
            "age_days": (0, 43800),  # 120 years in days
            "height_cm": (100, 250),
            "weight_kg": (30, 300),
        }
    
    def _get_expected_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get expected clinical ranges (tighter than absolute ranges)"""
        return {
            "glucose": (70, 140),
            "blood_pressure": (60, 120),
            "bmi": (18.5, 30),
            "age": (18, 65),
            "chol": (120, 240),
            "thalach": (100, 200),
            "trestbps": (90, 140),
        }
    
    def _get_injection_patterns(self) -> List[Dict[str, str]]:
        """Get injection attack patterns with metadata"""
        return [
            # SQL Injection
            {"pattern": r"SELECT\s+.*\s+FROM", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"INSERT\s+INTO", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"DROP\s+TABLE", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"UNION\s+SELECT", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"DELETE\s+FROM", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"UPDATE\s+.*\s+SET", "type": "sql_injection", "severity": "CRITICAL"},
            {"pattern": r"OR\s+1\s*=\s*1", "type": "sql_injection", "severity": "HIGH"},
            {"pattern": r"OR\s+'1'\s*=\s*'1'", "type": "sql_injection", "severity": "HIGH"},
            {"pattern": r"';.*--", "type": "sql_injection", "severity": "CRITICAL"},
            
            # XSS Attacks
            {"pattern": r"<script>.*</script>", "type": "xss", "severity": "HIGH"},
            {"pattern": r"onerror\s*=", "type": "xss", "severity": "HIGH"},
            {"pattern": r"onload\s*=", "type": "xss", "severity": "HIGH"},
            {"pattern": r"javascript:", "type": "xss", "severity": "HIGH"},
            {"pattern": r"alert\s*\(", "type": "xss", "severity": "MEDIUM"},
            {"pattern": r"eval\s*\(", "type": "xss", "severity": "HIGH"},
            
            # Command Injection
            {"pattern": r";\s*exec\s*", "type": "command_injection", "severity": "CRITICAL"},
            {"pattern": r";\s*ping\s*", "type": "command_injection", "severity": "CRITICAL"},
            {"pattern": r";\s*curl\s*", "type": "command_injection", "severity": "CRITICAL"},
            {"pattern": r"\$\{.*\}", "type": "command_injection", "severity": "HIGH"},
            {"pattern": r"`.*`", "type": "command_injection", "severity": "HIGH"},
            {"pattern": r"\|.*sh", "type": "command_injection", "severity": "CRITICAL"},
            
            # Path Traversal
            {"pattern": r"\.\./", "type": "path_traversal", "severity": "HIGH"},
            {"pattern": r"\.\.\\", "type": "path_traversal", "severity": "HIGH"},
            {"pattern": r"/etc/passwd", "type": "path_traversal", "severity": "CRITICAL"},
            {"pattern": r"C:\\Windows", "type": "path_traversal", "severity": "CRITICAL"},
            
            # Data Poisoning
            {"pattern": r"NaN|Infinity|-Infinity", "type": "data_poisoning", "severity": "HIGH"},
            {"pattern": r"None|NULL|null", "type": "data_poisoning", "severity": "MEDIUM"},
            
            # Protocol Attacks
            {"pattern": r"gopher://", "type": "protocol_attack", "severity": "CRITICAL"},
            {"pattern": r"file://", "type": "protocol_attack", "severity": "HIGH"},
            {"pattern": r"data://", "type": "protocol_attack", "severity": "CRITICAL"},
            {"pattern": r"http://.*\?", "type": "protocol_attack", "severity": "LOW"},
        ]
    
    # ============================================================================
    # 🚀 Core Validation Methods
    # ============================================================================
    
    def validate_and_sanitize(
        self,
        data: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> SecurityResult:
        """
        Validate and sanitize input data with multi-layer security
        
        Args:
            data: Input data dictionary
            context: Additional context (source, user, etc.)
            
        Returns:
            SecurityResult with validation outcome
        """
        
        self.stats["total_requests"] += 1
        
        result = SecurityResult(
            is_safe=True,
            sanitized_data=data.copy(),
        )
        
        # Layer 1: Type Validation
        self._validate_types(data, result)
        
        # Layer 2: Range Validation
        self._validate_ranges(data, result)
        
        # Layer 3: Injection Detection
        self._detect_injections(data, result)
        
        # Layer 4: Anomaly Detection
        if self.config.get("detect_anomalies", True):
            self._detect_anomalies(data, result)
        
        # Layer 5: Context Validation
        if context:
            self._validate_context(data, context, result)
        
        # Apply actions based on threat level
        self._apply_security_actions(data, result)
        
        # Update statistics
        if result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.stats["blocked_requests"] += 1
        elif result.warnings:
            self.stats["warned_requests"] += 1
        if result.sanitized_data != data:
            self.stats["sanitized_requests"] += 1
        
        # Log threats
        if result.threat_signatures:
            for signature in result.threat_signatures:
                self.stats["threats_detected"][signature] = (
                    self.stats["threats_detected"].get(signature, 0) + 1
                )
        
        # Audit log
        if self.config.get("enable_audit_log", True):
            self._audit_log(data, result)
        
        return result
    
    # ============================================================================
    # 🔧 Validation Layers
    # ============================================================================
    
    def _validate_types(self, data: Dict, result: SecurityResult):
        """Layer 1: Type validation"""
        
        for key, value in data.items():
            # Check for None or null
            if value is None:
                result.warnings.append(f"Null value for {key}")
                continue
            
            # Expected types for clinical data
            expected_type = self._get_expected_type(key)
            
            if expected_type == "numeric":
                if not isinstance(value, (int, float)):
                    try:
                        result.sanitized_data[key] = float(value)
                        result.warnings.append(f"Converted {key} from {type(value).__name__} to float")
                    except (ValueError, TypeError):
                        result.errors.append(f"Invalid numeric value for {key}: {value}")
                        result.is_safe = False
                        result.threat_level = ThreatLevel.HIGH
                        
            elif expected_type == "string":
                if not isinstance(value, str):
                    try:
                        result.sanitized_data[key] = str(value)
                        result.warnings.append(f"Converted {key} from {type(value).__name__} to string")
                    except:
                        result.errors.append(f"Invalid string value for {key}: {value}")
                        result.is_safe = False
                        
            elif expected_type == "boolean":
                if isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes']:
                        result.sanitized_data[key] = True
                    elif value.lower() in ['false', '0', 'no']:
                        result.sanitized_data[key] = False
                    else:
                        result.errors.append(f"Invalid boolean value for {key}: {value}")
                        result.is_safe = False
                elif not isinstance(value, bool):
                    try:
                        result.sanitized_data[key] = bool(value)
                        result.warnings.append(f"Converted {key} from {type(value).__name__} to boolean")
                    except:
                        result.errors.append(f"Invalid boolean value for {key}: {value}")
                        result.is_safe = False
    
    def _validate_ranges(self, data: Dict, result: SecurityResult):
        """Layer 2: Range validation"""
        
        for key, value in data.items():
            if key not in self.clinical_ranges:
                continue
            
            if not isinstance(value, (int, float)):
                continue
            
            min_val, max_val = self.clinical_ranges[key]
            
            # Absolute range check
            if value < min_val or value > max_val:
                result.errors.append(
                    f"{key} value {value} outside clinical range [{min_val}, {max_val}]"
                )
                result.is_safe = False
                result.threat_level = ThreatLevel.HIGH
                continue
            
            # Expected range check (tighter bounds)
            if key in self.clinical_expected and self.config.get("clamp_to_expected", True):
                exp_min, exp_max = self.clinical_expected[key]
                
                if value < exp_min:
                    result.warnings.append(
                        f"{key} value {value} below expected range [{exp_min}, {exp_max}], clamped"
                    )
                    result.sanitized_data[key] = exp_min
                    
                elif value > exp_max:
                    result.warnings.append(
                        f"{key} value {value} above expected range [{exp_min}, {exp_max}], clamped"
                    )
                    result.sanitized_data[key] = exp_max
    
    def _detect_injections(self, data: Dict, result: SecurityResult):
        """Layer 3: Injection attack detection"""
        
        block_on_injection = self.config.get("block_on_injection", True)
        
        for key, value in data.items():
            if not isinstance(value, str):
                continue
            
            for pattern_info in self.injection_patterns:
                pattern = pattern_info["pattern"]
                pattern_type = pattern_info["type"]
                severity = pattern_info["severity"]
                
                if re.search(pattern, value, re.IGNORECASE):
                    threat_msg = f"Injection pattern detected in {key}: {pattern_type} ({severity})"
                    
                    if severity in ["CRITICAL", "HIGH"]:
                        result.errors.append(threat_msg)
                        result.is_safe = False
                        result.threat_level = ThreatLevel.CRITICAL
                        result.threat_signatures.append(f"{pattern_type}:{key}")
                        
                        if block_on_injection:
                            result.action_taken = SecurityAction.BLOCK
                    else:
                        result.warnings.append(threat_msg)
                        result.threat_signatures.append(f"{pattern_type}:{key}")
                    
                    # Sanitize by removing dangerous characters
                    if self.config.get("sanitize_inputs", True):
                        sanitized_value = self._sanitize_value(value)
                        if sanitized_value != value:
                            result.sanitized_data[key] = sanitized_value
                            result.warnings.append(f"Sanitized {key} for injection patterns")
    
    def _detect_anomalies(self, data: Dict, result: SecurityResult):
        """Layer 4: Statistical anomaly detection"""
        
        # Convert to array for statistical analysis
        numeric_values = []
        feature_keys = []
        
        for key, value in data.items():
            if key in self.clinical_ranges and isinstance(value, (int, float)):
                numeric_values.append(value)
                feature_keys.append(key)
        
        if not numeric_values:
            return
        
        # Check for extreme outliers
        z_scores = np.abs(stats.zscore(numeric_values))
        threshold = self.config.get("anomaly_threshold", 3.0)
        
        for idx, z_score in enumerate(z_scores):
            if z_score > threshold:
                result.warnings.append(
                    f"Anomaly detected in {feature_keys[idx]}: z-score = {z_score:.2f}"
                )
                result.threat_signatures.append(f"anomaly:{feature_keys[idx]}")
    
    def _validate_context(self, data: Dict, context: Dict, result: SecurityResult):
        """Layer 5: Context-aware validation"""
        
        # Source IP validation
        if "source_ip" in context:
            # Check for internal vs external
            source_ip = context["source_ip"]
            if source_ip.startswith("192.168.") or source_ip.startswith("10."):
                # Internal - lower risk
                pass
            else:
                # External - higher risk
                result.warnings.append("External source IP detected")
        
        # Rate limiting check
        if "rate_limit" in context and "request_count" in context:
            request_count = context["request_count"]
            max_requests = self.config.get("max_requests_per_second", 1000)
            if request_count > max_requests:
                result.errors.append("Rate limit exceeded")
                result.is_safe = False
                result.threat_level = ThreatLevel.HIGH
                result.action_taken = SecurityAction.BLOCK
        
        # User agent check
        if "user_agent" in context:
            user_agent = context["user_agent"]
            suspicious_agents = ["curl", "wget", "python-requests", "scrapy", "nmap"]
            if any(agent in user_agent.lower() for agent in suspicious_agents):
                result.warnings.append(f"Suspicious user agent: {user_agent}")
                result.threat_signatures.append("suspicious_user_agent")
    
    # ============================================================================
    # 🔧 Security Actions
    # ============================================================================
    
    def _apply_security_actions(self, data: Dict, result: SecurityResult):
        """Apply security actions based on threat level"""
        
        if result.threat_level == ThreatLevel.CRITICAL:
            result.action_taken = SecurityAction.BLOCK
            result.is_safe = False
            logger.critical(f"CRITICAL THREAT: {result.errors}")
            
        elif result.threat_level == ThreatLevel.HIGH:
            if self.config.get("block_on_outlier", False):
                result.action_taken = SecurityAction.BLOCK
                result.is_safe = False
            else:
                result.action_taken = SecurityAction.SANITIZE
                result.is_safe = True
            logger.warning(f"HIGH THREAT: {result.errors}")
            
        elif result.threat_level == ThreatLevel.MEDIUM:
            result.action_taken = SecurityAction.WARN
            result.is_safe = True
            logger.info(f"MEDIUM THREAT: {result.warnings}")
            
        elif result.warnings:
            result.action_taken = SecurityAction.WARN
            result.is_safe = True
    
    def _sanitize_value(self, value: str) -> str:
        """Sanitize a string value"""
        # Remove dangerous characters
        dangerous_chars = ["<", ">", "&", "'", "\"", ";", "`", "$", "|"]
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        return sanitized
    
    def _get_expected_type(self, key: str) -> str:
        """Get expected type for a field"""
        numeric_fields = list(self.clinical_ranges.keys())
        string_fields = ["name", "address", "email", "phone", "notes"]
        boolean_fields = ["exang", "fbs", "target", "smoking"]
        
        if key in numeric_fields:
            return "numeric"
        elif key in string_fields:
            return "string"
        elif key in boolean_fields:
            return "boolean"
        else:
            return "string"
    
    # ============================================================================
    # 🔧 Anomaly Detection Model Training
    # ============================================================================
    
    def train_anomaly_detector(self, reference_data: pd.DataFrame):
        """
        Train anomaly detection model on reference data
        
        Args:
            reference_data: Clean reference dataset
        """
        
        self._reference_data = reference_data
        self._feature_names = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit Elliptic Envelope for anomaly detection
        self._anomaly_model = EllipticEnvelope(
            contamination=0.01,
            random_state=42,
        )
        
        numeric_data = reference_data[self._feature_names].values
        self._anomaly_model.fit(numeric_data)
        
        logger.info(f"✅ Anomaly detector trained on {len(reference_data)} samples")
        logger.info(f"   Features: {len(self._feature_names)}")
    
    def detect_adversarial_examples(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect adversarial examples using trained anomaly detector
        
        Args:
            data: Input data to check
            
        Returns:
            DataFrame with anomaly flags
        """
        
        if self._anomaly_model is None or self._feature_names is None:
            logger.warning("Anomaly detector not trained")
            return pd.DataFrame()
        
        # Ensure only numeric features
        available_features = [f for f in self._feature_names if f in data.columns]
        if not available_features:
            return pd.DataFrame()
        
        numeric_data = data[available_features].values
        
        # Predict anomalies
        predictions = self._anomaly_model.predict(numeric_data)
        scores = self._anomaly_model.score_samples(numeric_data)
        
        # Create results
        results = pd.DataFrame({
            'is_adversarial': predictions == -1,
            'anomaly_score': scores,
        }, index=data.index)
        
        adversarial_count = results['is_adversarial'].sum()
        if adversarial_count > 0:
            logger.warning(f"🚨 Found {adversarial_count} potential adversarial examples")
        
        return results
    
    # ============================================================================
    # 🔧 Batch Protection
    # ============================================================================
    
    def protect_batch(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[SecurityResult]]:
        """
        Apply protection to batch data
        
        Args:
            data: Batch DataFrame
            
        Returns:
            Tuple of (sanitized_data, security_results)
        """
        
        results = []
        sanitized_rows = []
        
        for idx, row in data.iterrows():
            row_dict = row.to_dict()
            result = self.validate_and_sanitize(row_dict)
            results.append(result)
            
            if result.is_safe:
                sanitized_rows.append(result.sanitized_data)
            else:
                # Replace with safe defaults
                logger.warning(f"Row {idx} failed security check, using safe defaults")
                sanitized_rows.append(self._get_safe_defaults(row_dict))
        
        return pd.DataFrame(sanitized_rows), results
    
    def _get_safe_defaults(self, data: Dict) -> Dict:
        """Get safe defaults for a row"""
        safe_defaults = {}
        for key, value in data.items():
            if key in self.clinical_expected:
                exp_min, exp_max = self.clinical_expected[key]
                safe_defaults[key] = (exp_min + exp_max) / 2
            elif key in self.clinical_ranges:
                min_val, max_val = self.clinical_ranges[key]
                safe_defaults[key] = (min_val + max_val) / 2
            else:
                safe_defaults[key] = value
        return safe_defaults
    
    # ============================================================================
    # 🔧 Threat Intelligence
    # ============================================================================
    
    def update_threat_signatures(self, signatures: Dict[str, List[str]]):
        """Update threat signatures"""
        self.threat_signatures.update(signatures)
        logger.info(f"✅ Updated threat signatures: {len(signatures)} new signatures")
    
    def get_threat_intelligence(self) -> Dict:
        """Get threat intelligence report"""
        return {
            "total_requests": self.stats["total_requests"],
            "blocked_requests": self.stats["blocked_requests"],
            "sanitized_requests": self.stats["sanitized_requests"],
            "warned_requests": self.stats["warned_requests"],
            "block_rate": self.stats["blocked_requests"] / max(1, self.stats["total_requests"]),
            "threats_detected": self.stats["threats_detected"],
            "top_threats": sorted(
                self.stats["threats_detected"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "security_level": self.config.get("security_level", "HIGH"),
            "last_reset": self.stats["last_reset"].isoformat(),
        }
    
    # ============================================================================
    # 🔧 Audit Logging
    # ============================================================================
    
    def _audit_log(self, data: Dict, result: SecurityResult):
        """Log security events for audit"""
        
        if result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            log_entry = {
                "timestamp": result.timestamp.isoformat(),
                "threat_level": result.threat_level.value,
                "action_taken": result.action_taken.value,
                "threat_signatures": result.threat_signatures,
                "errors": result.errors,
                "warnings": result.warnings,
                "data_keys": list(data.keys()),
            }
            
            logger.warning(f"SECURITY EVENT: {json.dumps(log_entry)}")
            
            # In production, this would write to a secure audit log
            # e.g., to a dedicated audit database or SIEM
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "sanitized_requests": 0,
            "warned_requests": 0,
            "threats_detected": {},
            "last_reset": datetime.now(),
        }
        logger.info("🔄 Security statistics reset")
    
    def get_security_status(self) -> Dict:
        """Get overall security status"""
        return {
            "status": "active",
            "security_level": self.config.get("security_level", "HIGH"),
            "protection_layers": [
                "Type Validation",
                "Range Validation",
                "Injection Detection",
                "Anomaly Detection",
                "Context Validation",
                "Rate Limiting",
            ],
            "statistics": self.get_threat_intelligence(),
            "config": self.config,
        }


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_adversarial_protection: Optional[AdversarialProtection] = None


def get_adversarial_protection() -> AdversarialProtection:
    """Get adversarial protection singleton"""
    global _adversarial_protection
    if _adversarial_protection is None:
        _adversarial_protection = AdversarialProtection()
    return _adversarial_protection


def protect_input(data: Dict[str, Any], context: Optional[Dict] = None) -> SecurityResult:
    """Convenience function for input protection"""
    protection = get_adversarial_protection()
    return protection.validate_and_sanitize(data, context)


def protect_batch(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[SecurityResult]]:
    """Convenience function for batch protection"""
    protection = get_adversarial_protection()
    return protection.protect_batch(data)