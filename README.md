# MLOps End‑to‑End: TACO Garbage Detection

This is a personal effort to learn MLOps and go beyond just training/inferencing models in research/academia. The model training part and getting great object detection results is _not_ the motto of this project; it serves primarily as a placeholder to demonstrate the **complete operations lifecycle**.

A production‐grade MLOps pipeline for object detection using the TACO (Trash Annotations in Context) dataset, built with open‑source tools.

## 🏗 Architecture and Tech Stack
- **Data Versioning**: DVC (Google Cloud Storage backend)  
- **Experiment Tracking**: MLflow (Local server with artifact logging)  
- **Model**: Ultralytics YOLO (wrapped in a custom MLflow pyfunc)  
- **Serving**: FastAPI (Containerized with Docker)  
- **CI/CD**: GitHub Actions (Linting, Testing, Docker Build & Push)  
- **Monitoring & Observability**:  
  - Service Health: Prometheus & Grafana  
  - Model Drift: Evidently AI  

## 🚀 Phases Completed
### Phase 1 & 2: Data Engineering
- Set up DVC to track the TACO dataset, using Google Cloud Storage as the remote backend.  
- Created robust preprocessing scripts to convert COCO‑style annotations to YOLO format and split data, handling inconsistent filenames in the raw dataset.  
- Implemented `verify_labels.py` to visually audit bounding box correctness before training.  

### Phase 3: Experimentation & Training
- Integrated Ultralytics YOLO with MLflow.  
- Developed a custom `UltralyticsWrapper` (MLflow pyfunc) to encapsulate pre‑processing (resizing) and post‑processing (NMS), ensuring the served model accepts raw images.  
- Disabled conflicting autologgers to ensure clean, manual registration of “production‑ready” model artifacts.  

### Phase 4: Deployment (API)
- Built a FastAPI application that loads the model from MLflow’s `@staging` alias dynamically at startup.  
- Containerized the API using Docker, optimizing the image by installing only necessary system dependencies (e.g., `libgl1` for OpenCV).  
- Configured Docker networking (bridge/host) to allow containers to communicate with the local MLflow model registry.  

### Phase 5: CI/CD Pipeline
- Configured GitHub Actions for Continuous Integration: Linting (`black`) and unit testing (`pytest`).  
- Configured Data integrity checks: pulling data via DVC from GCS and verifying checksums.  
- Configured Continuous Delivery: Automated build and push of the API Docker image to GitHub Container Registry (GHCR) on merge to `main`.  

### Phase 6: Monitoring & Observability
- **Service Monitoring**: Instrumented FastAPI with Prometheus metrics. Deployed a `docker‑compose` stack running Grafana and Prometheus to track request latency, error rates in real time.  
- **Model Monitoring**: Implemented an Evidently AI pipeline. The API logs inference data, and a drift detection script compares production traffic against a reference dataset to alert on data drift (e.g., sudden drops in confidence).  

## 🛠️ Quick Start (Local)
**Prerequisites**: Docker, Python 3.12, MLflow  
1. Start MLflow Server:  
   ```bash
   mlflow server --host 0.0.0.0 --port 5001      --allowed‑hosts "localhost,127.0.0.1,host.docker.internal"
   ```  
2. Train & Register Model:  
   ```bash
   python src/training/train.py
   ```  
   Then go to http://localhost:5001, and assign the `@staging` alias to your newly logged model version.  
3. Launch Production Stack:  
   ```bash
   docker-compose up -d --build
   ```  
4. Access Services:  
   - API Docs: http://localhost:8000/docs  
   - Prometheus UI: http://localhost:9090  
   - Grafana: http://localhost:3000 (Data source: `http://prometheus:9090`)  

## 📊 Monitoring
**Drift detection** can be run manually (or scheduled via cron/GitHub Actions):  
```bash
python src/monitoring/generate_reference.py  # Run once to create baseline  
python src/monitoring/monitor_drift.py       # Run periodically to check for drift  
```

## ⌨️ Common Project Workflows
### Data Versioning (DVC)
When you modify the `data/prepared` directory (e.g., re‑running the split script):  
```bash
# 1. Tell DVC to track the changed directory  
dvc add data/prepared  
# 2. Commit the .dvc pointer file  
git add data/prepared.dvc  
git commit -m "Updated dataset (e.g., fixed labels)"  
# 3. Upload the data to remote storage (GCS)  
dvc push  
```
On a new machine (or CI environment):  
```bash
dvc pull
```

### Code Quality & CI
Before pushing a pull request:  
```bash
# 1. Format code  
black .  
# 2. Run tests  
pytest  
```

### Data Verification
To visually check that your bounding boxes look correct after a data split:  
```bash
python src/utils/verify_labels.py  # Saves ~5 random images+boxes to `data/verification_output/`
```

## ✅ To‑Do
- Create a script that can run the **entire pipeline** including monitoring drift and retriggering training if observed.  
- Note: One of the major but subtle problems I encountered was getting MLflow to talk to the Docker image/FastAPI part. The `--allowed‑hosts` flag is important here. It’s possible there’s an easier solution when dealing with multiple tools; we cannot always track and add each of them in the flag args.  

---
