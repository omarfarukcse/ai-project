# outputs/__init__.py
"""
Outputs Package - All Generated Artifacts

This package serves as the central location for all outputs:
- Figures: Visualizations and plots
- Logs: Application and system logs
- Reports: Clinical and system reports
- Data: Drift history, review queue, security audits

Structure:
    outputs/
    ├── figures/          # Generated plots and visualizations
    ├── logs/            # Application log files
    ├── reports/         # Clinical and system reports
    ├── __init__.py      # Package marker
    ├── drift_history.json    # Data drift tracking
    ├── review_queue.json     # Human review queue
    └── security_audit_report.json # Security audit results

Auto-generated files:
    - All files are auto-created by the system
    - No manual creation needed
    - Files are updated automatically
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Package version
__version__ = "3.0.0"

# Get the outputs directory
OUTPUTS_DIR = Path(__file__).parent

# Subdirectory paths
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
for dir_path in [FIGURES_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 📊 File Management Functions
# ============================================================================

def ensure_output_structure():
    """Ensure all output directories exist"""
    directories = [
        FIGURES_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return True


def get_figure_path(name: str, timestamp: bool = True) -> Path:
    """Get path for a figure file"""
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{ts}.png"
    else:
        filename = f"{name}.png"
    return FIGURES_DIR / filename


def get_log_path(name: str, timestamp: bool = True) -> Path:
    """Get path for a log file"""
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d")
        filename = f"{name}_{ts}.log"
    else:
        filename = f"{name}.log"
    return LOGS_DIR / filename


def get_report_path(name: str, timestamp: bool = True) -> Path:
    """Get path for a report file"""
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{ts}.json"
    else:
        filename = f"{name}.json"
    return REPORTS_DIR / filename


# ============================================================================
# 📊 Drift History Management
# ============================================================================

class DriftHistoryManager:
    """Manage drift detection history"""
    
    def __init__(self):
        self.file_path = OUTPUTS_DIR / "drift_history.json"
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create empty drift history file if not exists"""
        if not self.file_path.exists():
            self._write_data([])
    
    def _read_data(self) -> list:
        """Read drift history data"""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_data(self, data: list):
        """Write drift history data"""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_drift_record(self, record: Dict[str, Any]):
        """Add a drift detection record"""
        data = self._read_data()
        data.append({
            "timestamp": datetime.now().isoformat(),
            "record": record,
        })
        self._write_data(data)
    
    def get_history(self, limit: int = 100) -> list:
        """Get drift history"""
        data = self._read_data()
        return data[-limit:] if limit else data
    
    def get_latest_drift(self) -> Optional[Dict]:
        """Get latest drift record"""
        data = self._read_data()
        return data[-1] if data else None
    
    def clear_history(self):
        """Clear drift history"""
        self._write_data([])


# ============================================================================
# 📋 Review Queue Management
# ============================================================================

class ReviewQueueManager:
    """Manage human review queue"""
    
    def __init__(self):
        self.file_path = OUTPUTS_DIR / "review_queue.json"
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create empty review queue if not exists"""
        if not self.file_path.exists():
            self._write_data({"queue": [], "completed": [], "statistics": {}})
    
    def _read_data(self) -> Dict:
        """Read review queue data"""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"queue": [], "completed": [], "statistics": {}}
    
    def _write_data(self, data: Dict):
        """Write review queue data"""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_review_item(self, item: Dict[str, Any]):
        """Add item to review queue"""
        data = self._read_data()
        item["added_at"] = datetime.now().isoformat()
        item["status"] = "pending"
        item["review_id"] = f"REV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        data["queue"].append(item)
        self._write_data(data)
    
    def get_pending_reviews(self) -> list:
        """Get pending review items"""
        data = self._read_data()
        return [item for item in data["queue"] if item.get("status") == "pending"]
    
    def complete_review(self, review_id: str, result: Dict):
        """Complete a review"""
        data = self._read_data()
        
        # Find and remove from queue
        for i, item in enumerate(data["queue"]):
            if item.get("review_id") == review_id:
                item["completed_at"] = datetime.now().isoformat()
                item["status"] = "completed"
                item["result"] = result
                data["completed"].append(item)
                del data["queue"][i]
                
                # Update statistics
                stats = data.get("statistics", {})
                stats["total_completed"] = stats.get("total_completed", 0) + 1
                stats["last_completed"] = datetime.now().isoformat()
                data["statistics"] = stats
                
                self._write_data(data)
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get review statistics"""
        data = self._read_data()
        return data.get("statistics", {})


# ============================================================================
# 🔒 Security Audit Manager
# ============================================================================

class SecurityAuditManager:
    """Manage security audit reports"""
    
    def __init__(self):
        self.file_path = OUTPUTS_DIR / "security_audit_report.json"
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create empty security audit file if not exists"""
        if not self.file_path.exists():
            self._write_data({
                "audits": [],
                "statistics": {
                    "total_audits": 0,
                    "last_audit": None,
                    "critical_findings": 0,
                    "high_findings": 0,
                }
            })
    
    def _read_data(self) -> Dict:
        """Read security audit data"""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"audits": [], "statistics": {}}
    
    def _write_data(self, data: Dict):
        """Write security audit data"""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_audit_report(self, audit: Dict[str, Any]):
        """Add a security audit report"""
        data = self._read_data()
        
        audit["timestamp"] = datetime.now().isoformat()
        audit["audit_id"] = f"AUDIT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        data["audits"].append(audit)
        
        # Update statistics
        stats = data.get("statistics", {})
        stats["total_audits"] = stats.get("total_audits", 0) + 1
        stats["last_audit"] = datetime.now().isoformat()
        
        # Count findings by severity
        for finding in audit.get("findings", []):
            severity = finding.get("severity", "low").lower()
            if severity == "critical":
                stats["critical_findings"] = stats.get("critical_findings", 0) + 1
            elif severity == "high":
                stats["high_findings"] = stats.get("high_findings", 0) + 1
        
        data["statistics"] = stats
        self._write_data(data)
    
    def get_latest_audit(self) -> Optional[Dict]:
        """Get latest security audit"""
        data = self._read_data()
        audits = data.get("audits", [])
        return audits[-1] if audits else None
    
    def get_statistics(self) -> Dict:
        """Get security audit statistics"""
        data = self._read_data()
        return data.get("statistics", {})


# ============================================================================
# 🔧 Singleton Instances
# ============================================================================

_drift_manager = None
_review_manager = None
_audit_manager = None


def get_drift_manager() -> DriftHistoryManager:
    """Get drift history manager singleton"""
    global _drift_manager
    if _drift_manager is None:
        _drift_manager = DriftHistoryManager()
    return _drift_manager


def get_review_manager() -> ReviewQueueManager:
    """Get review queue manager singleton"""
    global _review_manager
    if _review_manager is None:
        _review_manager = ReviewQueueManager()
    return _review_manager


def get_audit_manager() -> SecurityAuditManager:
    """Get security audit manager singleton"""
    global _audit_manager
    if _audit_manager is None:
        _audit_manager = SecurityAuditManager()
    return _audit_manager


# ============================================================================
# 📦 Exports
# ============================================================================

__all__ = [
    # Directories
    "OUTPUTS_DIR",
    "FIGURES_DIR",
    "LOGS_DIR",
    "REPORTS_DIR",
    
    # Path helpers
    "get_figure_path",
    "get_log_path",
    "get_report_path",
    "ensure_output_structure",
    
    # Managers
    "DriftHistoryManager",
    "ReviewQueueManager",
    "SecurityAuditManager",
    "get_drift_manager",
    "get_review_manager",
    "get_audit_manager",
]