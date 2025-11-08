# MLOps End-to-End: TACO Garbage Detection

This is an effort at learning MLOps, to go beyond training/inferencing models in research/academia. The model training part and getting great results with object detection is not a moto of this project. It can be said as only a placeholder. 

A production-grade MLOps pipeline for object detection using the TACO dataset. This project demonstrates a complete lifecycle from data ingestion to monitored deployment, built with open-source tools.

## 🏗 Architecture and Tech Stack

* **Data Versioning:** DVC (Google Cloud Storage backend)
* **Experiment Tracking:** MLflow (Local server with artifact logging)
* **Model:** YOLOv11 (Ultralytics) wrapped in a custom MLflow `pyfunc`
* **Serving:** FastAPI (Containerized with Docker)
* **CI/CD:** GitHub Actions (Linting, Testing, Docker Build & Push)
* **Monitoring:**
    * *Service Health:* Prometheus & Grafana
    * *Model Drift:* Evidently AI

## 🚀 Phases Completed

### Phase 1 & 2: Data Engineering
* Set up DVC to track the TACO dataset, using **Google Cloud Storage** as the remote backend.
* Created robust preprocessing scripts to convert COCO annotations to YOLO format and split data, handling inconsistent filenames in the raw dataset. Depended on LLMs for this. 
* Implemented `verify_labels.py` to visually audit bounding box correctness before training.

### Phase 3: Experimentation & Training
* Integrated **Ultralytics YOLOv11** with **MLflow**.
* Developed a custom `UltralyticsWrapper` (MLflow `pyfunc`) to encapsulate pre-processing (resizing) and post-processing (NMS), ensuring the served model accepts raw images.
* Disabled conflicting autologgers to ensure clean, manual registration of "production-ready" model artifacts.

### Phase 4: Deployment (API)
* Built a **FastAPI** application that loads the model from MLflow's `@staging` alias dynamically at startup.
* Containerized the API using **Docker**, optimizing the image by installing only necessary system dependencies (`libgl1`) for OpenCV.
* Configured Docker networking to allow containers to communicate with the local MLflow model registry.

### Phase 5: CI/CD Pipeline
* Configured **GitHub Actions** for Continuous Integration:
    * Linting (`black`) and unit testing (`pytest`).
    * Data integrity checks (pulling data from GCS and verifying checksums).
* Configured Continuous Delivery:
    * Automated build and push of the API Docker image to **GitHub Container Registry (GHCR)** on merge to main.

### Phase 6: Monitoring & Observability
* **Service Monitoring:** Instrumented FastAPI with **Prometheus** metrics. Deployed a `docker-compose` stack running Grafana and Prometheus to track request latency and error rates in real-time.
* **Model Monitoring:** Implemented an **Evidently AI** pipeline. The API logs inference data, and a drift detection script compares production traffic against a reference dataset to alert on data drift (e.g., sudden drops in confidence).

## 🛠️ Quick Start (Local)

**Prerequisites:** Docker, Python 3.12, MLflow

1.  **Start MLflow Server:**
    ```bash
    mlflow server --host 0.0.0.0 --port 5001 --allowed-hosts "localhost,127.0.0.1,host.docker.internal"
    ```

2.  **Train & Register Model:**
    ```bash
    python src/training/train.py
    ```
    *(Go to http://localhost:5001 and assign the `@staging` alias to your new model version)*

3.  **Launch Production Stack:**
    ```bash
    docker-compose up -d --build
    ```

4.  **Access Services:**
    * API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
    * Prometheus: [http://localhost:9090](http://localhost:9090)
    * Grafana: [http://localhost:3000](http://localhost:3000)
    * For adding prometheus datasource in Grafana, add as http://prometheus:9090


## Monitoring

Drift detection can be run manually (or scheduled):
```bash
python src/monitoring/generate_reference.py  # Run once to create baseline
python src/monitoring/monitor_drift.py       # Run periodically to check for drift
```

## ToDO

Create a script that can run the entire pipeline inculuding monitoring drift and retrigger training if observed.

Note: One of the major but silly problems I encountered was getting MlFlow to talk to Docker image/FastAPI part. The  --allowed-hosts flag is important here. Its possible there's an easier solution when dealing with multiple tools as we cannot track and add each of them in the flag args. ToDO again.
