# 🏥 CDSS Healthcare - Clinical Decision Support System

## FAANG-Level Explainable AI for Healthcare

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85%2B-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-orange.svg)](https://mlflow.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.27%2B-blue.svg)](https://kubernetes.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/omarfarukcse/ai-project/actions/workflows/ci.yml/badge.svg)](https://github.com/omarfarukcse/ai-project/actions/workflows/ci.yml)
[![CD](https://github.com/omarfarukcse/ai-project/actions/workflows/cd.yml/badge.svg)](https://github.com/omarfarukcse/ai-project/actions/workflows/cd.yml)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Training & Registry](#-model-training--registry)
- [Deployment](#-deployment)
- [Monitoring & Observability](#-monitoring--observability)
- [Security](#-security)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**CDSS Healthcare** is a **production-grade, explainable AI-powered Clinical Decision Support System** designed to assist healthcare professionals in predicting disease risk from patient clinical data. Built with enterprise scalability, reliability, and clinical safety at its core.

### 🏥 Why CDSS?

| Aspect | Description |
|--------|-------------|
| **Clinical Accuracy** | High-performance models optimized for **recall (sensitivity)** to minimize false negatives |
| **Explainability** | SHAP-based transparent predictions with **natural language explanations** |
| **Clinical Safety** | Human-in-the-loop, fallback systems, and **bias monitoring** |
| **Enterprise Ready** | Kubernetes-native, auto-scaling, **canary deployments** |
| **Regulatory Compliance** | HIPAA-inspired security, audit logging, and **governance** |

### Supported Diseases

- **Diabetes** (Pima Indians Diabetes Dataset)
- **Heart Disease** (UCI Heart Disease Dataset)

---

## ✨ Key Features

### 🔬 Clinical AI Features

| Feature | Description |
|---------|-------------|
| **Disease Risk Prediction** | Diabetes and Heart Disease prediction with >85% recall |
| **Explainable AI** | SHAP-based global and local explanations |
| **Natural Language Explanations** | Clinician-friendly prediction explanations |
| **Risk Scoring** | Low/Moderate/High risk classification |
| **Batch Prediction** | Process up to 1000 patients simultaneously |
| **Real-time Streaming** | Server-Sent Events for live predictions |

### 🚀 Production Features

| Feature | Description |
|---------|-------------|
| **High Performance** | <100ms cached, <500ms uncached predictions |
| **Auto-scaling** | Kubernetes HPA with 2-10 pods |
| **Canary Deployments** | Zero-risk model rollouts with traffic splitting |
| **Circuit Breaker** | Fault tolerance for external services |
| **Fallback System** | Clinical rule-based fallback when ML unavailable |
| **Distributed Caching** | Redis-based multi-level caching |
| **Async Tasks** | Celery for SHAP explanations, reports, batch |

### 📊 Monitoring & Observability

| Feature | Description |
|---------|-------------|
| **Metrics** | Prometheus with 20+ custom metrics |
| **Dashboards** | Grafana with pre-built CDSS dashboards |
| **Logging** | Loki + Promtail for centralized logging |
| **Tracing** | Tempo for distributed tracing |
| **Alerting** | Slack, Email, PagerDuty integration |
| **Drift Detection** | Automated data and model drift monitoring |
| **Bias Monitoring** | Fairness and demographic parity checks |

### 🔒 Security

| Feature | Description |
|---------|-------------|
| **Authentication** | JWT-based authentication with refresh tokens |
| **Authorization** | RBAC with fine-grained permissions |
| **Rate Limiting** | Distributed token bucket rate limiting |
| **Encryption** | AES-256-GCM data encryption |
| **Network Security** | Kubernetes network policies |
| **Audit Logging** | Complete audit trail for all actions |
| **Input Validation** | Schema validation with adversarial protection |

### 🛠️ MLOps Features

| Feature | Description |
|---------|-------------|
| **Model Registry** | MLflow with staging/production/archived |
| **Data Versioning** | DVC for dataset version control |
| **Pipeline Orchestration** | Airflow DAGs for automated retraining |
| **CI/CD** | GitHub Actions with canary deployments |
| **Feature Store** | Online/Offline feature serving |
| **Model Promotion** | Automated promotion with validation |
| **Rollback** | One-click rollback to previous models |

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │  Streamlit  │ │ React Web   │ │ Mobile Apps │ │ CLI/API     │        │
│  │  Dashboard  │ │ UI          │ │             │ │ Clients     │        │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  Ingress/ALB  │  Rate Limiter  │  SSL/TLS  │  Load Balancer    │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API SERVICES LAYER                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │Predict  │ │ Batch   │ │Streaming│ │Explain  │ │Metrics  │ │Admin    ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BUSINESS LOGIC LAYER                                │
│  ┌───────────────────────┐ ┌───────────────────────────┐                 │
│  │   Inference Pipeline   │ │    Training Pipeline       │                 │
│  │  • Validation          │ │  • Data Ingestion         │                 │
│  │  • Feature Fetch       │ │  • Preprocessing          │                 │
│  │  • Model Predict       │ │  • Model Training         │                 │
│  │  • Explainability      │ │  • Evaluation             │                 │
│  │  • Drift Detection     │ │  • Calibration            │                 │
│  │  • Fallback            │ │  • Registry               │                 │
│  └───────────────────────┘ └───────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA & MODEL LAYER                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │Feature Store│ │Model Registry│ │   Data      │ │  Caching    │        │
│  │  • Online   │ │  • Staging   │ │  • Raw      │ │  • Redis    │        │
│  │  • Offline  │ │  • Production│ │  • Processed│ │  • Memory   │        │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INFRASTRUCTURE & OBSERVABILITY                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │Kubernetes│ │  Istio  │ │Prometheus│ │ Grafana │ │  Loki   │ │  Tempo  ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → API Gateway → Middleware Stack → Route Handler
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Redis Cache  │
                            │  (Hit → Fast) │
                            └───────────────┘
                                    │ (Miss)
                                    ▼
                        ┌─────────────────────┐
                        │  Inference Pipeline │
                        └─────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  Feature    │ │   Model     │ │  SHAP       │
            │  Fetch      │ │   Predict   │ │  Explain    │
            └─────────────┘ └─────────────┘ └─────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │   Response Builder  │
                        └─────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   Async Tasks │
                            │  • Audit Log  │
                            │  • Report     │
                            └───────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technology | Version |
|----------|------------|---------|
| **Language** | Python | 3.8+ |
| **Web Framework** | FastAPI | 0.85+ |
| **ML Framework** | Scikit-learn, XGBoost | 1.0+, 1.5+ |
| **Explainability** | SHAP | 0.40+ |
| **Model Registry** | MLflow | 2.0+ |
| **Task Queue** | Celery | 5.2+ |
| **Cache** | Redis | 7.0+ |
| **Database** | PostgreSQL | 15+ |
| **Container** | Docker | 20.10+ |
| **Orchestration** | Kubernetes | 1.27+ |
| **Service Mesh** | Istio | 1.18+ |
| **Monitoring** | Prometheus, Grafana | 2.45+, 9.5+ |
| **Logging** | Loki | 2.8+ |
| **Tracing** | Tempo | 2.1+ |
| **CI/CD** | GitHub Actions | Latest |
| **IaC** | Terraform | 1.5+ |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Kubernetes cluster (for production)
- Redis (for caching)
- PostgreSQL (for metadata)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/omarfarukcse/ai-project.git
cd ai-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Setup data directory
python scripts/setup_outputs.py

# Train model
make train

# Start development server
make dev-serve

# Open dashboard
make dev-serve-dashboard
```

### Docker Quick Start

```bash
# Build images
make docker-build

# Start services
make docker-up

# Check status
make docker-status

# View logs
make docker-logs

# Stop services
make docker-down
```

---

## 📦 Installation

### Development Installation

```bash
# Install with all dependencies
make dev-install

# Or manually
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

### Production Installation

```bash
# Using Kubernetes
kubectl apply -f infrastructure/kubernetes/ -n cdss

# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

---

## 💻 Usage

### API Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pregnancies": 6,
        "glucose": 148,
        "bmi": 33.6,
        "age": 50
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "patients": [
            {"glucose": 148, "bmi": 33.6},
            {"glucose": 120, "bmi": 28.5}
        ]
    }
)
```

### CLI Commands

```bash
# Train model
cdss-train

# Deploy to production
cdss-deploy

# Validate model
cdss-validate

# Register model
cdss-register

# Promote model
cdss-promote
```

### Streamlit Dashboard

```bash
# Run dashboard
streamlit run dashboard/app.py

# Access at http://localhost:8501
```

---

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single patient risk prediction |
| `POST` | `/predict/batch` | Batch prediction for multiple patients |
| `GET` | `/predict/stream` | Streaming predictions (SSE) |
| `GET` | `/explanations/{patient_id}` | SHAP explanations |
| `GET` | `/explanations/{patient_id}/waterfall` | Waterfall plot |
| `GET` | `/explanations/{patient_id}/force` | Force plot |
| `GET` | `/feature-importance` | Global feature importance |
| `GET` | `/model-metrics` | Model performance metrics |
| `GET` | `/drift-report` | Drift detection report |
| `GET` | `/health` | System health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/auth/login` | Authentication |
| `POST` | `/auth/refresh` | Token refresh |
| `POST` | `/auth/logout` | Logout |
| `GET` | `/admin/status` | Admin status (admin only) |
| `POST` | `/admin/promote` | Promote model (admin only) |
| `POST` | `/admin/rollback` | Rollback model (admin only) |

### Authentication Example

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Admin123!"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use token
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{"glucose": 148, "bmi": 33.6}'
```

---

## 🤖 Model Training & Registry

### Training Pipeline

```bash
# Train with default settings
make train

# Train specific model
python scripts/train.py --models xgboost

# Train with custom config
python scripts/train.py --config config/training_config.yaml

# Force retraining
make train-force
```

### Model Registration

```bash
# Register with auto-version
python scripts/register_model.py \
    --model-path models/staging/model.pkl \
    --auto-version

# Register with specific version
python scripts/register_model.py \
    --model-path models/staging/model.pkl \
    --version v1.0.0 \
    --features glucose bmi age
```

### Model Promotion

```bash
# Promote to production
python scripts/promote_model.py --version v1.0.0

# Promote with canary
python scripts/promote_model.py \
    --version v1.0.0 \
    --canary \
    --canary-percentage 0.1

# Rollback
python scripts/promote_model.py --rollback
```

---

## 🚢 Deployment

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/ -n cdss

# Check deployment status
kubectl rollout status deployment/cdss-api -n cdss

# Scale deployment
kubectl scale deployment cdss-api --replicas=5 -n cdss

# Canary deployment
./scripts/canary-deploy.sh v3.0.0 5 300

# Rollback
kubectl rollout undo deployment/cdss-api -n cdss
```

### Terraform (AWS)

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan infrastructure
terraform plan

# Apply infrastructure
terraform apply

# Destroy infrastructure
terraform destroy
```

---

## 📊 Monitoring & Observability

### Access Monitoring Tools

| Tool | URL | Credentials |
|------|-----|-------------|
| **Prometheus** | http://localhost:9090 | None |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Loki** | http://localhost:3100 | None |
| **Tempo** | http://localhost:3200 | None |
| **AlertManager** | http://localhost:9093 | None |

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cdss_requests_total` | Total requests | - |
| `cdss_request_latency_seconds` | Request latency | >2s (P95) |
| `cdss_model_recall` | Model recall | <0.7 |
| `cdss_drift_score` | Data drift score | >0.2 |
| `cdss_error_rate` | Error rate | >5% |
| `system_cpu_percent` | CPU usage | >80% |
| `system_memory_percent` | Memory usage | >85% |

---

## 🔒 Security

### Authentication & Authorization

```yaml
# Default users (change in production)
admin:
  username: admin
  password: Admin123!
  roles: [admin, ml_engineer, clinician]
  permissions: ["*"]

ml_engineer:
  username: ml_engineer
  password: MlEngineer123!
  roles: [ml_engineer]
  permissions: [train, evaluate, deploy, monitor]

clinician:
  username: clinician
  password: Clinician123!
  roles: [clinician]
  permissions: [predict, view_reports]
```

### Rate Limiting

```yaml
# Default rate limits
rate_limit: 100 requests/minute
burst_multiplier: 1.5

# Per user roles
admin: 1000/minute
ml_engineer: 500/minute
clinician: 100/minute
```

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
make test

# Integration tests
make test-integration

# Load tests
make test-load

# All tests
make test-all
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

### Load Testing

```bash
# Start Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Access at http://localhost:8089
# Test scenarios:
# - 1 user: 10 requests
# - 10 users: 100 requests
# - 100 users: 1000 requests
```

---

## 📁 Project Structure

```
cdss_healthcare_ai/
├── src/                          # Source Code
│   ├── api/                      # API Layer
│   ├── pipelines/                # ML Pipelines
│   ├── components/               # ML Components
│   ├── feature_store/            # Feature Store
│   ├── orchestration/            # Orchestration
│   ├── monitoring/               # Monitoring
│   ├── security/                 # Security
│   ├── caching/                  # Caching
│   ├── async_tasks/              # Async Tasks
│   ├── validation/               # Validation
│   └── utils/                    # Utilities
├── config/                       # Configuration
├── data/                         # Data Layer
├── models/                       # Model Registry
├── infrastructure/               # Infrastructure
├── observability/                # Observability
├── tests/                        # Tests
├── scripts/                      # Automation
├── outputs/                      # Outputs
├── .github/workflows/            # CI/CD
├── docker-compose.yml            # Docker Compose
├── requirements.txt              # Dependencies
├── pyproject.toml                # Project Config
├── Makefile                      # Automation
└── .env                         # Environment Variables
```

---

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Make your changes**
4. **Run pre-commit hooks**
   ```bash
   pre-commit run --all-files
   ```
5. **Run tests**
   ```bash
   make test
   ```
6. **Submit a pull request**

### Commit Convention

```bash
# Format
<type>(<scope>): <subject>

# Types
feat: New feature
fix: Bug fix
docs: Documentation
style: Code style
refactor: Code refactor
test: Tests
chore: Maintenance

# Example
feat(api): add batch prediction endpoint
fix(model): correct recall calculation
docs(readme): update deployment guide
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

- **AI Healthcare Team** - [@omarfarukcse](https://github.com/omarfarukcse)
- **ML Engineering** - [@omarfarukcse](https://github.com/omarfarukcse)
- **Clinical Informatics** - [@omarfarukcse](https://github.com/omarfarukcse)

---

## 📞 Support

- **Documentation**: [https://docs.cdss-healthcare.com](https://docs.cdss-healthcare.com)
- **Issues**: [GitHub Issues](https://github.com/omarfarukcse/ai-project/issues)
- **Email**: omarfarukcse@gmail.com

---

## 🌟 Acknowledgments

- **Open Source Community** - For the amazing tools and libraries
- **Clinical Partners** - For domain expertise and validation
- **Healthcare Providers** - For real-world testing and feedback

---

## 📊 Badges

[![CI](https://github.com/omarfarukcse/ai-project/actions/workflows/ci.yml/badge.svg)](https://github.com/omarfarukcse/ai-project/actions/workflows/ci.yml)
[![CD](https://github.com/omarfarukcse/ai-project/actions/workflows/cd.yml/badge.svg)](https://github.com/omarfarukcse/ai-project/actions/workflows/cd.yml)
[![Coverage](https://codecov.io/gh/omarfarukcse/ai-project/branch/main/graph/badge.svg)](https://codecov.io/gh/omarfarukcse/ai-project)
[![Docker Pulls](https://img.shields.io/docker/pulls/omarfarukcse/cdss-api.svg)](https://hub.docker.com/r/omarfarukcse/cdss-api)

---

## 🔗 Quick Links

- [API Documentation](https://api.cdss-healthcare.com/docs)
- [Dashboard](https://dashboard.cdss-healthcare.com)
- [Monitoring](https://monitoring.cdss-healthcare.com)
- [Metrics](https://metrics.cdss-healthcare.com)

---

**Built with ❤️ for Healthcare AI**
