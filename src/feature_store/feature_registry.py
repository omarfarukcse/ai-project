# src/feature_store/feature_registry.py
"""
Centralized Feature Registry with Version Control
"""

import json
import hashlib
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.config_manager import config_manager
from src.feature_store.feature_engineering import FeatureDefinition, FeatureType, FeatureStatus

logger = get_logger(__name__)


@dataclass
class FeatureVersion:
    """Feature version information"""
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    changes: List[str] = field(default_factory=list)
    author: str = "system"
    status: str = "active"


@dataclass
class FeatureCatalog:
    """Feature catalog with metadata"""
    features: Dict[str, FeatureDefinition] = field(default_factory=dict)
    version: str = "1.0.0"
    updated_at: datetime = field(default_factory=datetime.now)
    total_features: int = 0
    
    def add_feature(self, feature: FeatureDefinition):
        """Add feature to catalog"""
        self.features[feature.name] = feature
        self.total_features = len(self.features)
        self.updated_at = datetime.now()
    
    def remove_feature(self, name: str):
        """Remove feature from catalog"""
        if name in self.features:
            del self.features[name]
            self.total_features = len(self.features)
            self.updated_at = datetime.now()
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature by name"""
        return self.features.get(name)
    
    def list_features(
        self,
        feature_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureDefinition]:
        """List features with filters"""
        
        features = list(self.features.values())
        
        if feature_type:
            features = [f for f in features if f.feature_type.value == feature_type]
        
        if status:
            features = [f for f in features if f.status.value == status]
        
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]
        
        return features
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat(),
            "total_features": self.total_features,
            "features": {
                name: feature.to_dict()
                for name, feature in self.features.items()
            }
        }


class FeatureRegistry:
    """
    Centralized Feature Registry with:
    - Feature catalog management
    - Version control
    - Feature lineage
    - Validation rules
    - Dependency management
    - Audit trail
    """
    
    def __init__(
        self,
        catalog_path: Optional[str] = None,
        auto_save: bool = True
    ):
        self.catalog_path = catalog_path or "data/feature_store/catalog.json"
        self.auto_save = auto_save
        self.catalog = FeatureCatalog()
        self._versions: Dict[str, List[FeatureVersion]] = {}
        self._audit_log: List[Dict] = []
        
        # Load existing catalog
        self._load_catalog()
        
        logger.info(f"📚 FeatureRegistry initialized: {self.catalog_path}")
        logger.info(f"   Total features: {self.catalog.total_features}")
    
    def _load_catalog(self):
        """Load catalog from file"""
        path = Path(self.catalog_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                for name, feature_data in data.get('features', {}).items():
                    feature = FeatureDefinition(
                        name=name,
                        description=feature_data.get('description', ''),
                        feature_type=FeatureType(feature_data.get('feature_type', 'numerical')),
                        status=FeatureStatus(feature_data.get('status', 'active')),
                        version=feature_data.get('version', '1.0.0'),
                        created_at=datetime.fromisoformat(feature_data.get('created_at', datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(feature_data.get('updated_at', datetime.now().isoformat())),
                        owner=feature_data.get('owner', 'system'),
                        tags=feature_data.get('tags', []),
                        dependencies=feature_data.get('dependencies', [])
                    )
                    self.catalog.add_feature(feature)
                
                logger.info(f"✅ Loaded {len(self.catalog.features)} features from catalog")
                
            except Exception as e:
                logger.error(f"❌ Failed to load catalog: {str(e)}")
    
    def _save_catalog(self):
        """Save catalog to file"""
        if not self.auto_save:
            return
        
        try:
            path = Path(self.catalog_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.catalog.to_dict(), f, indent=2, default=str)
            
            logger.debug(f"✅ Catalog saved: {self.catalog_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save catalog: {str(e)}")
    
    # ============================================================================
    # 🚀 Feature Management
    # ============================================================================
    
    def register_feature(
        self,
        name: str,
        description: str,
        feature_type: Union[str, FeatureType] = "numerical",
        status: Union[str, FeatureStatus] = "active",
        owner: str = "system",
        tags: List[str] = None,
        dependencies: List[str] = None,
        validation_fn: Optional[Callable] = None,
    ) -> FeatureDefinition:
        """
        Register a new feature in the catalog
        
        Args:
            name: Feature name
            description: Feature description
            feature_type: Feature data type
            status: Feature status
            owner: Feature owner
            tags: Feature tags
            dependencies: Feature dependencies
            validation_fn: Validation function
            
        Returns:
            Registered FeatureDefinition
        """
        
        # Convert string to enum
        if isinstance(feature_type, str):
            feature_type = FeatureType(feature_type)
        if isinstance(status, str):
            status = FeatureStatus(status)
        
        # Check if feature already exists
        if name in self.catalog.features:
            raise ValueError(f"Feature {name} already exists")
        
        # Validate dependencies
        if dependencies:
            for dep in dependencies:
                if dep not in self.catalog.features:
                    raise ValueError(f"Dependency {dep} not found in catalog")
        
        # Create feature
        feature = FeatureDefinition(
            name=name,
            description=description,
            feature_type=feature_type,
            status=status,
            owner=owner,
            tags=tags or [],
            dependencies=dependencies or [],
            validation_fn=validation_fn,
        )
        
        # Add to catalog
        self.catalog.add_feature(feature)
        
        # Log audit
        self._log_audit(
            action="register",
            feature_name=name,
            details={"feature_type": feature_type.value}
        )
        
        # Save catalog
        self._save_catalog()
        
        logger.info(f"✅ Registered feature: {name}")
        
        return feature
    
    def update_feature(
        self,
        name: str,
        description: Optional[str] = None,
        status: Optional[Union[str, FeatureStatus]] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> FeatureDefinition:
        """Update existing feature"""
        
        if name not in self.catalog.features:
            raise ValueError(f"Feature {name} not found")
        
        feature = self.catalog.features[name]
        
        changes = []
        
        if description is not None:
            changes.append(f"description: {feature.description} -> {description}")
            feature.description = description
        
        if status is not None:
            if isinstance(status, str):
                status = FeatureStatus(status)
            changes.append(f"status: {feature.status.value} -> {status.value}")
            feature.status = status
        
        if tags is not None:
            changes.append(f"tags: {feature.tags} -> {tags}")
            feature.tags = tags
        
        if dependencies is not None:
            # Validate dependencies
            for dep in dependencies:
                if dep not in self.catalog.features:
                    raise ValueError(f"Dependency {dep} not found in catalog")
            changes.append(f"dependencies: {feature.dependencies} -> {dependencies}")
            feature.dependencies = dependencies
        
        feature.updated_at = datetime.now()
        
        # Version bump
        version_parts = feature.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        feature.version = '.'.join(version_parts)
        
        # Log audit
        self._log_audit(
            action="update",
            feature_name=name,
            details={"changes": changes}
        )
        
        # Save catalog
        self._save_catalog()
        
        logger.info(f"✅ Updated feature: {name} (v{feature.version})")
        
        return feature
    
    def archive_feature(self, name: str) -> bool:
        """Archive a feature"""
        
        if name not in self.catalog.features:
            raise ValueError(f"Feature {name} not found")
        
        feature = self.catalog.features[name]
        feature.status = FeatureStatus.ARCHIVED
        feature.updated_at = datetime.now()
        
        # Log audit
        self._log_audit(
            action="archive",
            feature_name=name,
        )
        
        # Save catalog
        self._save_catalog()
        
        logger.info(f"✅ Archived feature: {name}")
        
        return True
    
    def delete_feature(self, name: str) -> bool:
        """Delete a feature (dangerous)"""
        
        if name not in self.catalog.features:
            raise ValueError(f"Feature {name} not found")
        
        # Check if any feature depends on this
        for feature_name, feature in self.catalog.features.items():
            if name in feature.dependencies:
                raise ValueError(f"Feature {name} is a dependency of {feature_name}")
        
        self.catalog.remove_feature(name)
        
        # Log audit
        self._log_audit(
            action="delete",
            feature_name=name,
        )
        
        # Save catalog
        self._save_catalog()
        
        logger.warning(f"⚠️ Deleted feature: {name}")
        
        return True
    
    # ============================================================================
    # 🔧 Feature Query Methods
    # ============================================================================
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature by name"""
        return self.catalog.get_feature(name)
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[FeatureDefinition]:
        """Get features by type"""
        return self.catalog.list_features(feature_type=feature_type.value)
    
    def get_active_features(self) -> List[FeatureDefinition]:
        """Get all active features"""
        return self.catalog.list_features(status="active")
    
    def get_features_by_tag(self, tag: str) -> List[FeatureDefinition]:
        """Get features by tag"""
        return self.catalog.list_features(tags=[tag])
    
    def get_feature_lineage(self, name: str) -> Dict[str, Any]:
        """Get feature lineage (dependencies and dependents)"""
        
        if name not in self.catalog.features:
            return {}
        
        feature = self.catalog.features[name]
        
        # Find dependents
        dependents = []
        for f_name, f in self.catalog.features.items():
            if name in f.dependencies:
                dependents.append(f_name)
        
        return {
            "feature": name,
            "dependencies": feature.dependencies,
            "dependents": dependents,
            "version": feature.version,
            "status": feature.status.value,
        }
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature catalog statistics"""
        
        features = list(self.catalog.features.values())
        
        stats = {
            "total_features": len(features),
            "by_type": {},
            "by_status": {},
            "versions": {},
            "latest_version": self.catalog.version,
            "updated_at": self.catalog.updated_at.isoformat(),
        }
        
        for feature in features:
            # By type
            type_key = feature.feature_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1
            
            # By status
            status_key = feature.status.value
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1
            
            # Versions
            stats["versions"][feature.name] = feature.version
        
        return stats
    
    # ============================================================================
    # 📊 Feature Validation
    # ============================================================================
    
    def validate_feature(
        self,
        name: str,
        value: Any,
        context: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Validate a feature value"""
        
        if name not in self.catalog.features:
            return False, f"Feature {name} not found"
        
        feature = self.catalog.features[name]
        
        # Basic type validation
        if feature.feature_type == FeatureType.NUMERICAL:
            if not isinstance(value, (int, float)):
                return False, f"Expected numerical value, got {type(value)}"
        
        elif feature.feature_type == FeatureType.CATEGORICAL:
            if not isinstance(value, (str, int)):
                return False, f"Expected categorical value, got {type(value)}"
        
        elif feature.feature_type == FeatureType.BINARY:
            if value not in [0, 1, True, False]:
                return False, f"Expected binary value (0/1), got {value}"
        
        # Custom validation
        if feature.validation_fn:
            try:
                is_valid, message = feature.validation_fn(value, context)
                if not is_valid:
                    return False, message
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        
        return True, "Validation passed"
    
    # ============================================================================
    # 🔧 Audit & Logging
    # ============================================================================
    
    def _log_audit(self, action: str, feature_name: str, details: Dict = None):
        """Log audit entry"""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "feature_name": feature_name,
            "details": details or {},
        }
        
        self._audit_log.append(entry)
        
        # Limit audit log size
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
    
    def get_audit_log(self, feature_name: Optional[str] = None) -> List[Dict]:
        """Get audit log"""
        
        if feature_name:
            return [
                entry for entry in self._audit_log
                if entry["feature_name"] == feature_name
            ]
        
        return self._audit_log
    
    # ============================================================================
    # 🔧 Export Methods
    # ============================================================================
    
    def export_catalog(self, path: Optional[str] = None) -> str:
        """Export catalog to file"""
        
        export_path = path or self.catalog_path
        path_obj = Path(export_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w') as f:
            json.dump(self.catalog.to_dict(), f, indent=2, default=str)
        
        logger.info(f"✅ Catalog exported to {export_path}")
        
        return str(export_path)
    
    def generate_feature_documentation(self) -> str:
        """Generate feature documentation in markdown"""
        
        doc_lines = [
            "# Feature Catalog Documentation",
            "",
            f"**Version:** {self.catalog.version}",
            f"**Total Features:** {self.catalog.total_features}",
            f"**Last Updated:** {self.catalog.updated_at.isoformat()}",
            "",
            "## Feature List",
            "",
        ]
        
        for name, feature in self.catalog.features.items():
            doc_lines.extend([
                f"### {name}",
                f"- **Description:** {feature.description}",
                f"- **Type:** {feature.feature_type.value}",
                f"- **Status:** {feature.status.value}",
                f"- **Version:** {feature.version}",
                f"- **Owner:** {feature.owner}",
                f"- **Tags:** {', '.join(feature.tags) if feature.tags else 'None'}",
                f"- **Dependencies:** {', '.join(feature.dependencies) if feature.dependencies else 'None'}",
                "",
            ])
        
        return "\n".join(doc_lines)