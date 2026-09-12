# IndusTSFM Productization V1 — TPT-like Industrial Intelligence Platform

## Goal

Evolve the existing leakage-safe TSFM research platform into a product-facing industrial time-series intelligence system without destroying the current research protocols.

The product principle is **task-centric rather than model-centric**:

```text
Project
→ Data Source
→ Data Audit
→ Task Definition
→ Analysis / Shift / OOD
→ Model Routing
→ Zero-shot / Adaptation / Fine-tuning
→ Forecast / Anomaly / Diagnosis
→ Deployment Application
```

The current `industrial_tsfm` research runners remain the numerical core. Product-facing code lives under `industrial_tsfm.platform` and must not silently change train/validation/test boundaries.

## V1 scope

V1 is deliberately smaller than a full TPT clone. It establishes the product skeleton that future enterprise datasets can plug into.

### 1. Project and task contracts

A project owns one or more data sources and one or more explicit tasks. Supported task contracts are defined up front for forecasting, anomaly detection, fault diagnosis, RUL, regression and classification. V1 executes forecasting through the existing IndusTSFM engine; the other contracts establish stable extension boundaries.

### 2. Data Audit as a first-class product feature

Before any model is chosen, the platform reports:

- row and variable counts;
- timestamp validity, duplicates and ordering problems;
- missingness and degenerate/constant tags;
- usable numeric variables;
- high-correlation pairs;
- target availability;
- a transparent readiness score and its penalties.

The audit does not silently clean data. Cleaning and resampling decisions remain explicit follow-up actions.

### 3. Product UI artifact

V1 generates a dependency-free HTML report with a future product navigation structure:

```text
Overview
Data
Analysis
Models
Applications
Deployment
```

The first report exposes project metadata, data health, task definition, variable status and the recommended model-input set. Later iterations will replace the static artifact with a service-backed Web UI.

### 4. GitHub Actions product smoke

`.github/workflows/platform-smoke.yml` validates the new product path on every relevant push/PR and can also be triggered manually. It:

1. installs the repository in the existing dev environment;
2. runs platform-specific tests;
3. generates a synthetic industrial project;
4. runs Data Audit;
5. builds the HTML product artifact;
6. uploads the generated project/audit/report as a workflow artifact.

No external model checkpoint or private industrial dataset is required for CI.

## Next milestones

### V1.1 — Model Router

Wrap the existing validation-only `adaptation_engine` behind a product-facing routing contract. Inputs should include task type, target support budget, latency/parameter constraints and shift score. Outputs must retain `target_labels_used: false` semantics where applicable.

### V1.2 — Data connectors

Add connector interfaces for CSV/Parquet first, then SQL, MQTT and OPC-UA. Connectors normalize metadata and schema but do not hide provenance.

### V1.3 — Application runtime

Turn a validated model/task pair into a local inference application with a stable request/response schema, replay mode, prediction history and model/version metadata.

### V1.4 — Anomaly engine integration

Port reusable pieces from `industrial-anomaly-detection` behind a task adapter rather than copying the whole repository. Initial target capabilities: anomaly score, thresholding, event detection, sensor attribution and latency profiling.

### V2 — Analytics and decision layer

Add lag analysis, feature importance, causal-candidate analysis, calibrated UQ, what-if rollouts and constrained optimization.

### V3 — Agent orchestration

Only after the numerical tools are stable, add an LLM agent that invokes Data Audit, analysis, model routing, forecasting, anomaly diagnosis and optimization as tools. The LLM must not replace numerical detection or control logic.

## Non-goals

- training a billion-parameter foundation model from scratch;
- claiming one TSFM is universally best;
- using target-test labels for automatic product decisions;
- treating a chat interface as the product core;
- direct closed-loop control before explicit safety and constraint layers exist.

## Current first slice

The first product slice adds:

- `src/industrial_tsfm/platform/contracts.py`
- `src/industrial_tsfm/platform/data_audit.py`
- `src/industrial_tsfm/platform/report.py`
- `scripts/run_platform_smoke.py`
- `tests/test_platform.py`
- `.github/workflows/platform-smoke.yml`

This gives the repository a concrete path from research benchmark to an industrial-data product while keeping the existing TSFM experiments intact.
