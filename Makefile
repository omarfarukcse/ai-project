# Makefile
.PHONY: help install test lint format train deploy docker-build docker-up docker-down clean

help:
	@echo "CDSS Healthcare Makefile"
	@echo ""
	@echo "Commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  train        Train models"
	@echo "  deploy       Deploy to production"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start services"
	@echo "  docker-down  Stop services"
	@echo "  clean        Clean artifacts"

install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

test-load:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

lint:
	flake8 src/ tests/ --count --max-complexity=10 --max-line-length=127 --statistics
	black --check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

train:
	python scripts/train.py

deploy:
	python scripts/deploy.py --version $$(python -c "import json; print(json.load(open('models/production/metadata.json'))['version'])")

docker-build:
	docker-compose -f docker-compose.prod.yml build

docker-up:
	docker-compose -f docker-compose.prod.yml up -d

docker-down:
	docker-compose -f docker-compose.prod.yml down

clean:
	rm -rf outputs/figures/* outputs/reports/* outputs/logs/*
	rm -rf models/staging/* models/production/* models/archived/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pytest_cache" -exec rm -rf {} +