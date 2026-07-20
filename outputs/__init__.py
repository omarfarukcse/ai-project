# outputs/__init__.py
"""
Outputs Package - All generated artifacts
"""

from pathlib import Path

# Create output directories
OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = Path("models")

# Ensure directories exist
for dir_path in [FIGURES_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

__all__ = [
    "OUTPUTS_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
]