# Tomato Leaf Health Classifier — MLOps Pipeline

## Problem Statement

Binary classification of tomato leaf images as **HEALTHY** or **UNHEALTHY**
(diseased / pest-infested). Non-tomato images are automatically rejected.

## Architecture

PlantDoc → DVC Pipeline → MLflow Experiments → Champion Model → FastAPI → Docker → Kubernetes

## Quick Start

### Predict (Docker)

```bash
docker pull <USERNAME>/tomato-health-api:latest
docker run -p 8000:8000 <USERNAME>/tomato-health-api:latest

curl -X POST http://localhost:8000/predict \
  -F "file=@tomato_leaf.jpg"
```

### Reproduce Training

```bash
git clone https://github.com/erickanangwee/tomato-health-classifier.git
 && cd tomato-health-classifier
pip install -r requirements-train.txt
dvc pull        # OR: dvc repro  (re-trains from scratch)
```

### Run MLflow UI

```bash
mlflow server --port 5000
# Open http://localhost:5000
```

## API Endpoints

| Method | Path     | Description                     |
| ------ | -------- | ------------------------------- |
| GET    | /health  | Liveness check + model status   |
| GET    | /docs    | Interactive Swagger UI          |
| GET    | /classes | List output classes + label map |
| POST   | /predict | Upload leaf image → prediction  |

## Models Compared

| Model               | Tuning | Tracking |
| ------------------- | ------ | -------- |
| Logistic Regression | Optuna | MLflow   |
| Random Forest       | Optuna | MLflow   |
| XGBoost             | Optuna | MLflow   |

## Project Layout

See `IMPLEMENTATION_GUIDE.md` for full step-by-step setup instructions.
