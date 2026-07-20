# src/utils/log_sanitizer.py
import re
import json
from typing import Dict, Any, List, Optional
import hashlib
from src.logger import get_logger

logger = get_logger(__name__)

class LogSanitizer:
    """
    Sanitize logs to remove PII and sensitive information
    Implements field-level redaction and masking
    """
    
    def __init__(self):
        # PII patterns
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'address': r'\d{1,5}\s+[A-Za-z]+\s+[A-Za-z]+',
            'id': r'ID-\d{6,10}',
            'name': r'[A-Z][a-z]+\s+[A-Z][a-z]+',
            'medical_record': r'MRN-\d{6,10}'
        }
        
        # Sensitive fields to redact (exact matches)
        self.sensitive_fields = [
            'patient_id',
            'patient_name',
            'ssn',
            'address',
            'email',
            'phone',
            'medical_record_number',
            'date_of_birth',
            'zip_code',
            'full_name'
        ]
        
        self.salt = "CDSS_LOG_SALT_2024"  # For deterministic hashing
    
    def sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize dictionary data, removing PII and sensitive fields
        """
        sanitized = {}
        
        for key, value in data.items():
            # Skip sensitive fields
            if key.lower() in self.sensitive_fields:
                sanitized[key] = self._mask_value(value)
                continue
            
            # Recursively sanitize nested structures
            if isinstance(value, dict):
                sanitized[key] = self.sanitize(value)
                continue
            
            if isinstance(value, list):
                sanitized[key] = [self.sanitize_item(item) for item in value]
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def sanitize_item(self, item: Any) -> Any:
        """Sanitize a single item (for lists)"""
        if isinstance(item, dict):
            return self.sanitize(item)
        elif isinstance(item, str):
            return self._sanitize_string(item)
        else:
            return item
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string by removing PII patterns"""
        if not isinstance(text, str):
            return text
        
        sanitized = text
        
        # Apply PII pattern masking
        for pattern_name, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, sanitized)
            if matches:
                for match in matches:
                    hashed = self._hash_value(match)
                    sanitized = sanitized.replace(match, f"[{pattern_name.upper()}_{hashed[:8]}]")
        
        return sanitized
    
    def _mask_value(self, value: Any) -> str:
        """Mask a sensitive value"""
        if value is None:
            return None
        
        if isinstance(value, str):
            if len(value) > 8:
                return f"****{value[-4:]}"
            else:
                return "****"
        elif isinstance(value, (int, float)):
            return f"****{str(value)[-4:]}" if len(str(value)) > 4 else "****"
        else:
            return "****"
    
    def _hash_value(self, value: str) -> str:
        """Hash a value for consistent masking"""
        return hashlib.sha256(f"{value}{self.salt}".encode()).hexdigest()
    
    def sanitize_log_message(self, message: str) -> str:
        """Sanitize a log message string"""
        # Apply PII pattern masking
        for pattern in self.pii_patterns.values():
            message = re.sub(pattern, '[REDACTED]', message)
        
        return message
    
    def sanitize_request(self, request: Dict) -> Dict:
        """Sanitize API request data"""
        sanitized = self.sanitize(request)
        
        # Remove any authentication tokens
        if 'Authorization' in sanitized:
            sanitized['Authorization'] = '[REDACTED]'
        
        if 'access_token' in sanitized:
            sanitized['access_token'] = '[REDACTED]'
        
        if 'password' in sanitized:
            sanitized['password'] = '[REDACTED]'
        
        return sanitized

# Global sanitizer instance
log_sanitizer = LogSanitizer()

# Custom logging filter for PII removal
class PIIFilter:
    """Logging filter to remove PII from log records"""
    
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = log_sanitizer.sanitize_log_message(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                log_sanitizer.sanitize_log_message(str(arg)) 
                if isinstance(arg, str) else arg 
                for arg in record.args
            )
        
        return True