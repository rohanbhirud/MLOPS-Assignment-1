# MLOps Assignment – End-to-End Pipeline Guide
This repository demonstrates an end-to-end MLOps pipeline for heart disease prediction, compliant with assignment requirements. It includes data acquisition, EDA, model training, MLflow tracking, Docker containerization, Kubernetes deployment, and a CI/CD pipeline.

---

## 🚀 Quick Setup & Execution (Master Runbook)
Follow these steps to run the entire project from scratch.

### Prerequisites
- **Python 3.12+** & **pip**
- **Docker**
- **kubectl** & **Minikube** (or Docker Desktop K8s)

### 1. Model Development (Local)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare data
python data/download_data.py --output data/raw/heart.csv

# 3. Exploratory Data Analysis (EDA)
# Outputs plots to reports/figures/
python -m src.eda --data data/raw/heart.csv --out reports/figures

# 4. Train Models
# Trains LogReg & RandomForest, saves metrics to reports/, and logs to MLflow (mlruns/)
python -m src.model_train --data data/raw/heart.csv --models-dir models --reports-dir reports
```

### 2. Run API Locally
```bash
# Start FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test Endpoints:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

### 3. Docker Containerization
```bash
# Build the image
docker build -t heart-api .

# Run container
docker run -p 8000:8000 heart-api

# (The API is now accessible at http://127.0.0.1:8000)
```

---

## ☸️ Kubernetes Deployment
Deploy the Dockerized API to a Kubernetes cluster (Minikube example).

### 1. Build Image in Cluster
```bash
# Point local Docker client to Minikube's Docker daemon
eval "$(minikube docker-env)"

# Build the image inside Minikube
docker build -t heart-api .
```

### 2. Apply Manifests
```bash
# Deploy Service and Deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get svc heart-api
```

### 3. Access the Service
```bash
# Method A: Port Forward (Simplest)
kubectl port-forward svc/heart-api 8000:8000

# Method B: Minikube Service URL
minikube service heart-api --url
```

---

## 📊 Monitoring & Logging
The system uses **Prometheus** format metrics logging.

- **Metrics Endpoint**: `GET /metrics`
    - Exposes `api_requests_total` and `api_request_latency_seconds`.
    - Scrape this endpoint using a Prometheus server.
- **Logs**:
    - View container logs via Kubernetes: `kubectl logs -f deploy/heart-api`

---

## 🛠 Project Structure
- **src/**: Source code for data cleaning, training, and API.
- **data/**: Scripts for downloading and storing raw CSVs.
- **models/**: Serialized model artifacts (`.joblib`).
- **k8s/**: Kubernetes deployment manifests.
- **.github/workflows/**: CI/CD pipeline configuration.
- **mlruns/**: MLflow experiment tracking history.
- **reports/**: Generated plots and JSON metrics.

## 📝 Design Decisions
- **Model**: Logistic Regression was chosen for production for its interpretability and low latency.
- **Preprocessing**: Feature scaling (`StandardScaler`) and encoding (`OneHotEncoder`) are handled via Scikit-Learn pipelines.
- **CI/CD**: GitHub Actions pipeline handles linting (`ruff`), testing (`pytest`), training, and artifact delivery on every push.
