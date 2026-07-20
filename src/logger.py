import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import json
import traceback
import re

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'lineno': record.lineno,
        }
        
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_tb(record.exc_info[2])
            }
        
        return json.dumps(log_data)

class LoggerSetup:
    """Centralized logging configuration with correlation ID support"""
    
    _instance = None
    _correlation_id = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.log_dir = Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._configure_root_logger()
        self._initialized = True
    
    def _configure_root_logger(self):
        """Configure root logger with multiple handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = JSONFormatter()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with daily rotation
        log_file = self.log_dir / f"cdss_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=30, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = self.log_dir / f"cdss_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = TimedRotatingFileHandler(
            error_log_file, when='midnight', interval=1, backupCount=30, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Audit log
        audit_handler = TimedRotatingFileHandler(
            self.log_dir / "audit.log", when='midnight', interval=1, backupCount=90
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(file_formatter)
        root_logger.addHandler(audit_handler)
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        cls._correlation_id = correlation_id

def get_logger(name: str) -> logging.Logger:
    """Get configured logger with correlation ID"""
    logger = logging.getLogger(name)
    
    class CorrelationFilter(logging.Filter):
        def filter(self, record):
            record.correlation_id = LoggerSetup._correlation_id or 'N/A'
            return True
    
    logger.addFilter(CorrelationFilter())
    return logger

def set_correlation_id(correlation_id: str):
    LoggerSetup.set_correlation_id(correlation_id)