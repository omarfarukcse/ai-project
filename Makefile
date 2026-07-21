# Makefile
# Complete Makefile for CDSS Healthcare System

.PHONY: help install test lint format train deploy docker-build docker-up docker-down clean
.PHONY: dev-install dev-test dev-lint dev-format dev-docs dev-serve dev-load-test
.PHONY: all security coverage docs

# ============================================================================
# 🏠 Default target
# ============================================================================
.DEFAULT_GOAL := help

# ============================================================================
# 📚 Help
# ============================================================================
help:
	@echo "🏥 CDSS Healthcare Makefile"
	@echo ""
	@echo "📦 Installation & Setup:"
	@echo "  install          Install production dependencies"
	@echo "  dev-install      Install all dependencies (including dev)"
	@echo "  pre-commit       Install pre-commit hooks"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test             Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-load        Run load tests with Locust"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint             Run linters (flake8, black, mypy)"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security checks (bandit, safety)"
	@echo ""
	@echo "🤖 Model Operations:"
	@echo "  train            Train models"
	@echo "  deploy           Deploy to production"
	@echo "  promote          Promote model from staging to production"
	@echo "  validate         Validate current production model"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      View container logs"
	@echo "  docker-status    Show container status"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  monitor-up       Start monitoring stack"
	@echo "  monitor-down     Stop monitoring stack"
	@echo "  docs             Build documentation"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean            Clean artifacts and caches"
	@echo "  clean-all        Deep clean (including venv)"

# ============================================================================
# 📦 Installation
# ============================================================================
install:
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Production dependencies installed"

dev-install:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .
	pre-commit install
	@echo "✅ Development dependencies installed"

pre-commit:
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

# ============================================================================
# 🧪 Testing
# ============================================================================
test:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing
	@echo "✅ Unit tests completed"

test-integration:
	pytest tests/integration/ -v
	@echo "✅ Integration tests completed"

test-load:
	locust -f tests/load/locustfile.py --host=http://localhost:8000
	@echo "✅ Load tests started"

test-all: test test-integration
	@echo "✅ All tests completed"

coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

# ============================================================================
# 🔍 Code Quality
# ============================================================================
lint:
	@echo "🔍 Running linters..."
	flake8 src/ tests/ --count --max-complexity=10 --max-line-length=127 --statistics
	black --check src/ tests/
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting completed"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Formatting completed"

security:
	@echo "🔒 Running security checks..."
	bandit -r src/ -f json -o bandit-report.json || true
	safety check -r requirements.txt
	@echo "✅ Security checks completed"

# ============================================================================
# 🤖 Model Operations
# ============================================================================
train:
	python scripts/train.py
	@echo "✅ Training completed"

train-force:
	python scripts/train.py --force
	@echo "✅ Forced training completed"

deploy:
	@echo "🚀 Deploying model to production..."
	python scripts/deploy.py --version $$(python -c "import json; print(json.load(open('models/production/metadata.json'))['version'])")
	@echo "✅ Deployment completed"

promote:
	@echo "🚀 Promoting model to production..."
	python scripts/promote_model.py --version $$(python -c "import json; print(json.load(open('models/staging/metadata.json'))['version'])")
	@echo "✅ Promotion completed"

validate:
	python scripts/validate.py --stage production
	@echo "✅ Validation completed"

register:
	@echo "📦 Registering model..."
	python scripts/register_model.py --model-path models/staging/model.pkl --auto-version
	@echo "✅ Registration completed"

# ============================================================================
# 🐳 Docker
# ============================================================================
docker-build:
	docker-compose -f docker-compose.prod.yml build
	@echo "✅ Docker images built"

docker-up:
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Services started"

docker-down:
	docker-compose -f docker-compose.prod.yml down
	@echo "✅ Services stopped"

docker-restart: docker-down docker-up
	@echo "✅ Services restarted"

docker-logs:
	docker-compose -f docker-compose.prod.yml logs -f

docker-status:
	docker-compose -f docker-compose.prod.yml ps

docker-clean:
	docker-compose -f docker-compose.prod.yml down -v
	@echo "✅ Docker volumes cleaned"

# ============================================================================
# 📊 Monitoring
# ============================================================================
monitor-up:
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3000 (admin/admin)"
	@echo "   Loki: http://localhost:3100"

monitor-down:
	docker-compose -f docker-compose.monitoring.yml down
	@echo "✅ Monitoring stack stopped"

# ============================================================================
# 📚 Documentation
# ============================================================================
docs:
	@echo "📚 Building documentation..."
	cd docs && make html
	@echo "✅ Documentation built in docs/_build/html/"

docs-serve:
	cd docs && make livehtml
	@echo "📚 Documentation server started at http://localhost:8000"

# ============================================================================
# 🧹 Cleanup
# ============================================================================
clean:
	@echo "🧹 Cleaning artifacts..."
	rm -rf outputs/figures/* outputs/reports/* outputs/logs/*
	rm -rf models/staging/* models/production/* models/archived/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	@echo "✅ Cleanup completed"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf venv/
	rm -rf .venv/
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .eggs/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	@echo "✅ Deep cleaning completed"

# ============================================================================
# 🚀 Development Server
# ============================================================================
dev-serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
	@echo "🚀 Development server started at http://localhost:8000"

dev-serve-dashboard:
	streamlit run dashboard/app.py --server.port 8501
	@echo "🚀 Dashboard started at http://localhost:8501"

# ============================================================================
# 🔄 All-in-One Commands
# ============================================================================
all: install lint test
	@echo "✅ All checks passed"

ci: install lint test test-integration
	@echo "✅ CI pipeline passed"

prod: install train deploy
	@echo "✅ Production deployment complete"

# ============================================================================
# 📊 Database
# ============================================================================
db-migrate:
	alembic upgrade head
	@echo "✅ Database migration complete"

db-rollback:
	alembic downgrade -1
	@echo "✅ Database rollback complete"

# ============================================================================
# 🔧 Utilities
# ============================================================================
shell:
	ipython

notebook:
	jupyter notebook

lab:
	jupyter lab