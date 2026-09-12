# IndusTSFM Productization V1 — Industrial Time-Series Intelligence Platform

## Product goal

Evolve the leakage-safe IndusTSFM research platform into a task-centric industrial time-series intelligence product without weakening the current evaluation boundaries.

```text
Project
→ Data Source
→ Data Audit
→ Analytics / Shift / Variable Discovery
→ Task Definition
→ AutoModel / Adaptation Routing
→ Forecast / Anomaly
→ Application Runtime
→ Deployment
```

The product layer lives under `industrial_tsfm.platform`. Existing research runners remain the numerical core and are not silently rewritten by the product layer.

## Implemented V1 surface

### Project, data, and workspace

- explicit `ProjectSpec`, `DataSourceSpec`, and `TaskDefinition` contracts;
- local CSV, Parquet, and in-memory connectors;
- file provenance including resolved location, row/column metadata, and SHA-256 for local files;
- local JSON workspace for persistent project specifications and derived analysis artifacts;
- declared SQL, MQTT, and OPC-UA connector contracts that fail closed until live connection/security semantics are implemented.

### Data Audit

The platform reports timestamp validity/order, duplicate rows, missingness, constant variables, usable numeric variables, configured target availability, high-correlation pairs, and a transparent readiness score. Audit does not silently impute, resample, or reorder data.

### Industrial analytics

V1 now exposes four transparent analytics tools:

1. **Lagged association** — ranks historical/simultaneous feature-to-target associations over non-negative lags. Future feature values are never shifted backward into the target time step. This is association, not causality.
2. **Chronological distribution shift** — compares an early reference prefix with a late target suffix using source-standardized mean drift and dispersion change. Configured target labels are excluded by default.
3. **Cross-source distribution shift** — compares an explicit reference data source with a second target data source. The target source may be unlabeled; reference statistics do not use target data.
4. **Variable candidate discovery** — combines lag association, observed coverage, and a transparent shift penalty into a screening score. It is not causal discovery and is not model-derived feature importance.

The primary shift score is RMS source-standardized mean shift, matching the existing adaptation-engine convention.

### AutoModel / adaptation routing

The product router wraps the existing validation-only adaptation engine. It ranks implemented model candidates from validation evidence while honoring parameter, latency, and adaptation budgets. Supported strategies include zero-shot inference, target scaling, residual calibration, supervised few-shot fine-tuning, and PEFT where the selected model contract permits it.

`routing.auto_shift=true` can feed an unlabeled chronological shift score into strategy selection. An explicit `routing.shift_score` always takes precedence, so automatic shift evidence never silently overrides a user-supplied value.

### Anomaly triage

The first deployment-safe anomaly path uses PCA-SPE fitted on the configured normal training prefix, with train-score quantile thresholding, event counting, sensor contribution ranking, and lightweight pattern triage. Test labels are not used for threshold selection. Sensor contributions are triage evidence, not causal root-cause claims.

### Application runtime

`PlatformApplication` materializes project metadata, data provenance, audit evidence, validation evidence, model route, and deterministic offline replay. Replay preserves input order and performs no wall-clock sleep, hidden sorting, interpolation, or resampling.

### API and Studio UI

FastAPI provides a local product service and `IndusTSFM Studio` provides a dependency-free dashboard for Projects, Analytics, Model Hub, Applications, Data Sources, and Runtime status.

Important endpoints include:

```text
GET  /health
GET  /v1/capabilities
GET  /v1/projects
POST /v1/projects
POST /v1/projects/{project_id}/audit/{source_name}
POST /v1/projects/{project_id}/analysis/lag
POST /v1/projects/{project_id}/analysis/shift
POST /v1/projects/{project_id}/analysis/shift-sources
POST /v1/projects/{project_id}/analysis/discover
POST /v1/projects/{project_id}/route/{task_name}
POST /v1/projects/{project_id}/applications/{task_name}
POST /v1/projects/{project_id}/anomaly/{task_name}
GET  /v1/applications
```

The CLI entry point is `industrial-tsfm-platform` with `serve`, `capabilities`, and `list` commands.

## Evidence boundaries

The product layer keeps these rules explicit:

- model selection remains validation-only;
- automatic shift analysis does not use target labels;
- lag analysis never uses future feature values;
- variable discovery is a heuristic candidate screen, not a causal claim;
- anomaly thresholds are not selected on final target/test labels;
- deterministic numerical tools remain below any future LLM Agent layer;
- closed-loop OT control is not enabled in V1.

## CI contract

`.github/workflows/platform-smoke.yml` uses synthetic process data and no private checkpoint. It exercises project creation, audit, model routing, anomaly triage, lag analysis, distribution shift, variable discovery, replay, API imports, CLI capabilities, HTML reporting, and artifact generation. The existing multi-Python research CI remains independent.

## Next product boundary

The deterministic local V1 layer is now close to a freeze candidate. The next material architecture choice is the first live industrial connector:

- **OPC-UA first** emphasizes PLC/SCADA/process-industry integration and structured tag browsing;
- **MQTT first** emphasizes lightweight edge telemetry, gateways, and publish/subscribe deployment.

After the first live connector, the next planned slices are streaming inference/history, richer fault-diagnosis adapters, calibrated uncertainty, what-if/constrained optimization, and only then LLM Agent orchestration over the deterministic tools.
