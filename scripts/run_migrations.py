# scripts/run_migrations.py
import subprocess
import sys
from pathlib import Path
import os

from src.logger import get_logger

logger = get_logger(__name__)

def run_migration():
    """Run database migrations with rollback capability"""
    try:
        # Check if migrations directory exists
        if not Path("migrations").exists():
            logger.info("Initializing migrations...")
            subprocess.run(["alembic", "init", "migrations"], check=True)
        
        # Check current revision
        result = subprocess.run(
            ["alembic", "current"],
            capture_output=True,
            text=True
        )
        
        current_rev = result.stdout.strip()
        logger.info(f"Current migration: {current_rev if current_rev else 'None'}")
        
        # Generate migration if needed
        if "--autogenerate" in sys.argv:
            logger.info("Generating migration...")
            subprocess.run(["alembic", "revision", "--autogenerate", "-m", "auto_migration"], check=True)
        
        # Run migrations
        logger.info("Running migrations...")
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        
        logger.info("✅ Migrations completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Migration failed: {e}")
        logger.info("🔄 Rolling back...")
        subprocess.run(["alembic", "downgrade", "-1"], check=False)
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)