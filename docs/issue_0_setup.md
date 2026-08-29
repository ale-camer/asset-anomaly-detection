# Day 0 Setup & Project Tracking Checklist

## 📋 Overview

- **Project**: P-06 Asset Anomaly Detection (MLOps)
- **Phase**: Day 0 - Scaffolding, Repository & GitHub Roadmap Setup
- **Target Branch**: `feature/day-0-setup`
- **Base Branch**: `develop`
- **GitHub Repository**: [ale-camer/asset-anomaly-detection](https://github.com/ale-camer/asset-anomaly-detection)

---

## 🛑 Day 0 Golden Rule Verification

- [x] **Zero Business Logic**: No extractors, loaders, models, or pipeline execution code written in Day 0.
- [x] **Scaffolding Only**: Created empty directories with `.gitkeep`, config files, and documentation.
- [x] **Isolated Virtual Environment**: `.venv` created and verified.
- [x] **Git Flow Initialized**: `main` created with initial commit, `develop` branched, working on `feature/day-0-setup`.
- [x] **English Only**: All configs, docs, commits, issues, and tracking written in English.

---

## 🛠 Deliverables Checklist

| Deliverable | Status | Details |
| :--- | :---: | :--- |
| **Git Initialization** | ✅ Completed | Initial commit on `main`, `develop` created, checked out `feature/day-0-setup` |
| **Virtual Environment** | ✅ Completed | `.venv` isolated Python 3.14 environment created |
| **Configuration Files** | ✅ Completed | `.gitignore`, `.env.example`, `pyproject.toml` |
| **Directory Scaffolding** | ✅ Completed | `src/`, `dags/`, `tests/`, `infra/`, `docs/`, `data/` with `.gitkeep` |
| **Project Documentation** | ✅ Completed | `README.md` with system architecture and tech stack |
| **Setup Tracking Doc** | ✅ Completed | `docs/issue_0_setup.md` |
| **GitHub Remote Repo** | ✅ Completed | Public repository [ale-camer/asset-anomaly-detection](https://github.com/ale-camer/asset-anomaly-detection) |
| **GitHub Milestones (5)** | ✅ Completed | M1 (5 issues), M2 (4 issues), M3 (5 issues), M4 (4 issues), M5 (4 issues) |
| **GitHub Issues (22)** | ✅ Completed | 22 atomic issues created and linked across milestones |

---

## 🗺 Milestones & Issues Master Plan

### Milestone 1: Data Ingestion & Storage Architecture
- [ ] **Issue #1**: Setup Data Ingestion Configuration & Base Connector Interface
- [ ] **Issue #2**: Implement Financial & Crypto Market Data Ingestion Client
- [ ] **Issue #3**: Implement Pydantic Validation Schemas for Raw Ingested Market Data
- [ ] **Issue #4**: Implement Local & Lake Storage Layer (Parquet Sink)
- [ ] **Issue #5**: Implement Unit & Integration Tests for Data Ingestion Module

### Milestone 2: Feature Engineering & Processing Pipeline
- [ ] **Issue #6**: Implement Time-Series Windowing & Rolling Statistics Transformers
- [ ] **Issue #7**: Implement Volatility, Momentum & Technical Anomaly Feature Extractors
- [ ] **Issue #8**: Implement Feature Persistence Layer & Schema Validation
- [ ] **Issue #9**: Implement Unit Tests & Data Quality Verification for Feature Pipeline

### Milestone 3: Anomaly Detection Modeling & MLflow Registry
- [ ] **Issue #10**: Implement Baseline Unsupervised Anomaly Detector (Isolation Forest)
- [ ] **Issue #11**: Implement Deep Learning / Autoencoder Anomaly Detector
- [ ] **Issue #12**: Setup MLflow Experiment Tracking, Parameter & Metric Logging
- [ ] **Issue #13**: Implement Anomaly Scoring Thresholding & Evaluation Metrics
- [ ] **Issue #14**: Implement Model Registry Packaging, Artifact Serialization & Unit Tests

### Milestone 4: Orchestration & MLOps Infrastructure
- [ ] **Issue #15**: Configure Docker Compose Infrastructure (Postgres, MinIO, MLflow, Airflow)
- [ ] **Issue #16**: Implement Airflow DAG for Scheduled Ingestion & Feature Transformation
- [ ] **Issue #17**: Implement Airflow DAG for Automated Model Retraining & Validation
- [ ] **Issue #18**: End-to-End Orchestration Testing with Mock Data & Healthchecks

### Milestone 5: Serving, Real-Time API & Monitoring
- [ ] **Issue #19**: Implement FastAPI Anomaly Scoring Inference Endpoint
- [ ] **Issue #20**: Implement Data Drift & Concept Drift Monitoring (Evidently / Prometheus)
- [ ] **Issue #21**: Implement Streamlit Real-Time Anomaly Dashboard & Alerting UI
- [ ] **Issue #22**: Setup GitHub Actions CI/CD Pipeline & Documentation Finalization
