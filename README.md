# G224 MP126 — College Canteen Lunch Rush (Patna)

## Group & Problem
| Field | Value |
|---|---|
| **Group ID** | G224 |
| **Problem ID** | MP126 |
| **Faculty** | Dr. Upasna Joshi |
| **Batch** | 23 |

## Members
| Enrolment No. | Name | Role |
|---|---|---|
| S24CSEU0246 | Yash Mudgal | Tests, CI placeholders |
| S24CSEU1362 | Ekta Mishra | Params, config, requirements |
| S24CSEU1347 | Parv | Repo setup, DVC, data checks, findings |

## Project Description
The Patna Institute Canteen Committee needs to predict how many lunch plates (`plates_sold`) will be sold between 12–3 pm each day. This is a **regression** problem evaluated with **MAE**. The pipeline covers data versioning (DVC), a 7-point data audit, MLflow experiment tracking, a FastAPI prediction service, Docker containerisation, GitHub Actions CI, Kubernetes deployment, drift monitoring, and a final holdout prediction.

## Data SHA-256 (required for MLflow `data_sha256` parameter)
```
train.csv : 61d09d93327626578cf1806cb690c5d9bd3527719620496f4630bfcd5f25735e
later.csv : ba35ef6463591e749f8dea24f7765b97d91b8aec95562f9d43caffdc2f3faafc
```

## How to Run the Pipeline & Tests
```bash
# Run data quality checks (Stage 1):
python -m mp126.run_checks

# Run data cleaning pipeline (Stage 2):
python -m mp126.clean

# Run MLflow model training & registry (Stage 2):
python -m mp126.train

# Run pytest (does NOT need the CSVs):
pytest -q
```

## How to Run the Service
```bash
# Start the FastAPI prediction server:
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Build the Docker image:
docker build -t mp126-api:1.0 .

# Deploy to Kubernetes:
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## Repo Layout
```
README.md                    ← this file
requirements.txt             ← pinned libraries
params.yaml                  ← paths & thresholds (read with yaml.safe_load)
src/mp126/__init__.py
src/mp126/config.py          ← loads params.yaml
src/mp126/data_checks.py     ← 7 check functions
src/mp126/run_checks.py      ← CLI runner
src/mp126/clean.py           ← data cleaning pipeline
src/mp126/train.py           ← MLflow experiment tracking & model registry
app/main.py                  ← FastAPI service: POST /predict, GET /health
models/model.joblib          ← champion GradientBoosting model
models/encoder.joblib        ← fitted OrdinalEncoder
Dockerfile                   ← python:3.12-slim container image
.dockerignore                ← keeps image small
.github/workflows/ci.yml     ← pytest on every push
k8s/deployment.yaml          ← Deployment (2 replicas, probes)
k8s/service.yaml             ← Service mp126-svc (80 → 8000)
tests/test_data_checks.py    ← 15 pytest tests (no CSV access)
tests/test_clean.py          ← 13 cleaning pipeline tests
tests/test_api.py            ← 11 TestClient API tests
data/raw/train.csv.dvc       ← DVC pointer
data/raw/later.csv.dvc       ← DVC pointer
data/raw/holdout_features.csv← committed directly (no target)
.dvc/config                  ← DVC remote config
reports/findings.yaml        ← filled data_issues section
reports/data_audit.md        ← evidence narrative for each check
```

## Stage Progress
- [x] Stage 1 — Repo, DVC, 7 data checks, findings.yaml
- [x] Stage 2 — Cleaning pipeline, MLflow tracking, model registry, unit tests
- [x] Stage 3 — FastAPI, Docker, GitHub Actions, Kubernetes, JSON logging
- [ ] Stage 4 — Drift monitoring, retraining, holdout predictions
