# scripts/setup_outputs.py
"""
Initialize Output Directories and Files
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from outputs import (
    ensure_output_structure,
    get_drift_manager,
    get_review_manager,
    get_audit_manager,
)
from outputs import FIGURES_DIR, LOGS_DIR, REPORTS_DIR


def setup_outputs():
    """Initialize all output directories and files"""
    
    print("📁 Setting up outputs directory...")
    
    # 1. Ensure directory structure
    ensure_output_structure()
    print("   ✅ Directories created")
    
    # 2. Initialize managers (creates files)
    drift_manager = get_drift_manager()
    review_manager = get_review_manager()
    audit_manager = get_audit_manager()
    
    print("   ✅ Files initialized:")
    print(f"      - drift_history.json")
    print(f"      - review_queue.json")
    print(f"      - security_audit_report.json")
    
    # 3. Add sample data (optional)
    add_sample_data()
    
    print("✅ Setup complete!")
    print(f"   Figures: {FIGURES_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"   Reports: {REPORTS_DIR}")


def add_sample_data():
    """Add sample data for testing"""
    
    # Sample drift record
    drift_manager = get_drift_manager()
    drift_manager.add_drift_record({
        "type": "data_drift",
        "severity": "low",
        "features": ["glucose", "bmi"],
        "score": 0.12,
        "message": "Minor drift detected in glucose and bmi"
    })
    
    # Sample review item
    review_manager = get_review_manager()
    review_manager.add_review_item({
        "patient_id": "P12345",
        "prediction": {
            "risk_score": 85,
            "risk_level": "High Risk",
            "confidence": 0.92
        },
        "reason": "High risk prediction requires clinician review",
        "priority": "high"
    })
    
    # Sample security audit
    audit_manager = get_audit_manager()
    audit_manager.add_audit_report({
        "findings": [
            {
                "severity": "low",
                "title": "API Rate Limit Configuration",
                "description": "Rate limit set to 100 requests/minute",
                "recommendation": "Consider increasing for production"
            }
        ],
        "summary": "Security audit completed. No critical findings."
    })


if __name__ == "__main__":
    setup_outputs()