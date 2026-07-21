# Makefile
# Complete Makefile for CDSS Healthcare

.PHONY: help install dev-install test test-integration test-load test-all lint format security train train-force deploy promote register validate docker-build docker-up docker-down docker-logs docker-status docker-clean clean clean-all dev-serve dev-serve-dashboard monitor-up monitor-down docs

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
	@echo "  test-all         Run all tests"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint             Run linters (flake8, black, mypy)"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security checks (bandit, safety)"
	@echo ""
	@echo "🤖 Model Operations:"
	@echo "  train            Train models"
	@echo "  train-force      Force retraining"
	@echo "  deploy           Deploy to production"
	@echo "  promote          Promote model from staging to production"
	@echo "  validate         Validate current production model"
	@echo "  register         Register model in staging"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      View container logs"
	@echo "  docker-status    Show container status"
	@echo "  docker-clean     Clean Docker volumes"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  monitor-up       Start monitoring stack"
	@echo "  monitor-down     Stop monitoring stack"
	@echo "  docs             Build documentation"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean            Clean artifacts and caches"
	@echo "  clean-all        Deep clean (including venv)"
	@echo ""
	@echo "🚀 Development:"
	@echo "  dev-serve        Start development server"
	@echo "  dev-serve-dashboard Start dashboard"

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

test-integration:
	pytest tests/integration/ -v

test-load:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

test-all: test test-integration

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

train-force:
	python scripts/train.py --force

deploy:
	@echo "🚀 Deploying model to production..."
	python scripts/deploy.py --version $$(python -c "import json; print(json.load(open('models/production/metadata.json'))['version'])")

promote:
	@echo "🚀 Promoting model to production..."
	python scripts/promote_model.py --version $$(python -c "import json; print(json.load(open('models/staging/metadata.json'))['version'])")

validate:
	python scripts/validate.py --stage production

register:
	@echo "📦 Registering model..."
	python scripts/register_model.py --model-path models/staging/model.pkl --auto-version

# ============================================================================
# 🐳 Docker
# ============================================================================
docker-build:
	docker-compose -f docker-compose.yml build

docker-up:
	docker-compose -f docker-compose.yml up -d

docker-down:
	docker-compose -f docker-compose.yml down

docker-logs:
	docker-compose -f docker-compose.yml logs -f

docker-status:
	docker-compose -f docker-compose.yml ps

docker-clean:
	docker-compose -f docker-compose.yml down -v

# ============================================================================
# 📊 Monitoring
# ============================================================================
monitor-up:
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3000 (admin/admin)"

monitor-down:
	docker-compose -f docker-compose.monitoring.yml down

# ============================================================================
# 📚 Documentation
# ============================================================================
docs:
	@echo "📚 Building documentation..."
	cd docs && make html

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

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf venv/ .venv/ htmlcov/ .tox/ .eggs/ build/ dist/ *.egg-info

# ============================================================================
# 🚀 Development Server
# ============================================================================
dev-serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

dev-serve-dashboard:
	streamlit run dashboard/app.py --server.port 8501