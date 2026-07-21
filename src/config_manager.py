# src/config_manager.py
"""
Enterprise Configuration Manager with YAML + JSON Support
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Advanced Configuration Manager with:
    - YAML + JSON support
    - Environment variable interpolation
    - Caching
    - Schema validation
    - Multiple config sources
    - Live reload
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Any] = {}
        self._loaded_files: Dict[str, float] = {}
        self._watch_enabled = False
        
        logger.info(f"🔧 ConfigManager initialized: {config_dir}")
        
    def load_config(self, name: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Load configuration file
        
        Args:
            name: Config file name (without extension)
            refresh: Force refresh cache
            
        Returns:
            Configuration dictionary
        """
        
        # Check cache
        if not refresh and name in self._config_cache:
            return self._config_cache[name]
        
        # Try YAML first, then JSON
        config_path = self.config_dir / f"{name}.yaml"
        if not config_path.exists():
            config_path = self.config_dir / f"{name}.yml"
        if not config_path.exists():
            config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {name}")
        
        # Load file
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            # Interpolate environment variables
            config = self._interpolate_env_vars(config)
            
            # Cache
            self._config_cache[name] = config
            self._loaded_files[name] = config_path.stat().st_mtime
            
            logger.info(f"✅ Loaded config: {name} ({config_path})")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config {name}: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None, config_name: str = "main") -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Dot-notation key (e.g., "models.xgboost.n_estimators")
            default: Default value if key not found
            config_name: Config file name
            
        Returns:
            Configuration value
        """
        
        config = self.load_config(config_name)
        
        # Navigate nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any, config_name: str = "main"):
        """
        Set configuration value (in memory only)
        
        Args:
            key: Dot-notation key
            value: Value to set
            config_name: Config file name
        """
        
        config = self.load_config(config_name)
        
        # Navigate nested keys
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        self._config_cache[config_name] = config
    
    def reload(self, config_name: str = "main"):
        """Force reload configuration"""
        self.load_config(config_name, refresh=True)
    
    def reload_all(self):
        """Reload all configurations"""
        for name in list(self._config_cache.keys()):
            self.load_config(name, refresh=True)
    
    def _interpolate_env_vars(self, obj: Any) -> Any:
        """Interpolate environment variables in config values"""
        
        if isinstance(obj, dict):
            return {k: self._interpolate_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_env_vars(v) for v in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # Extract variable name
            var_name = obj[2:-1]
            # Check for default value
            if ':' in var_name:
                var_name, default = var_name.split(':', 1)
                return os.getenv(var_name, default)
            return os.getenv(var_name, obj)
        else:
            return obj
    
    def get_config_path(self, name: str) -> Optional[Path]:
        """Get configuration file path"""
        for ext in ['.yaml', '.yml', '.json']:
            path = self.config_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None
    
    def list_configs(self) -> List[str]:
        """List all available configuration files"""
        configs = []
        for ext in ['.yaml', '.yml', '.json']:
            configs.extend([f.stem for f in self.config_dir.glob(f"*{ext}")])
        return sorted(set(configs))
    
    def get_section(self, section: str, config_name: str = "main") -> Dict[str, Any]:
        """Get a configuration section"""
        config = self.load_config(config_name)
        return config.get(section, {})
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.get('global.environment', 'development')
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.get_environment() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.get_environment() == 'development'
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('global.debug', False)
    
    def get_log_level(self) -> str:
        """Get configured log level"""
        return self.get('global.log_level', 'INFO').upper()


# ============================================================================
# 🔧 Pydantic Config Models
# ============================================================================

class RedisConfig(BaseModel):
    """Redis configuration model"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    connection_timeout: int = 1
    socket_timeout: int = 1
    retry_on_timeout: bool = True
    cache_ttl: int = 3600
    cache_key_prefix: str = "cdss:cache:"
    cache_enabled: bool = True
    
    @validator('port')
    def validate_port(cls, v):
        if v < 0 or v > 65535:
            raise ValueError('Port must be between 0 and 65535')
        return v


class APIConfig(BaseModel):
    """API configuration model"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    rate_limit: int = 100
    rate_limit_period: str = "minute"
    timeout: int = 30
    max_request_size: int = 10485760
    
    @validator('port')
    def validate_port(cls, v):
        if v < 0 or v > 65535:
            raise ValueError('Port must be between 0 and 65535')
        return v


class ModelConfig(BaseModel):
    """Model configuration model"""
    models: List[str] = ["logistic_regression", "random_forest", "xgboost"]
    search_type: str = "grid"
    cv_folds: int = 5
    primary_metric: str = "recall"
    secondary_metric: str = "specificity"
    improvement_threshold: float = 0.01
    calibration_method: str = "platt"
    calibration_cv_folds: int = 5


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get configuration manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(name: str = "main") -> Dict[str, Any]:
    """Convenience function to load config"""
    return get_config_manager().load_config(name)


def get(key: str, default: Any = None, config: str = "main") -> Any:
    """Convenience function to get config value"""
    return get_config_manager().get(key, default, config)


config_manager = get_config_manager()