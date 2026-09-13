# Live Diagnosis V1

Live Diagnosis V1 is the diagnostic-triage product slice above the frozen read-only OPC-UA runtime. It turns a previously fitted anomaly artifact into operator-facing alert severity and sensor-contribution evidence without introducing online fitting, threshold tuning, causal root-cause claims, or OT control.

## Runtime path

```text
Project / anomaly task
  -> DiagnosisDeploymentSpec
  -> offline-fitted PCASPEArtifact
  -> active OPC-UA runtime / WindowProvider
  -> LiveDiagnosisApplication
  -> anomaly score + persistence
  -> deterministic triage severity
  -> statistical sensor contributors
  -> Studio / API
```

The live layer does not call `fit_pca_spe_artifact`, update the PCA basis, update the scaler, change the anomaly threshold, consume target labels, or write to the industrial source.

## Deployment contract

`DiagnosisDeploymentSpec` records the workspace project ID, source, anomaly task, feature order, runtime context size, contributor count and optional artifact reference.

A deployment is accepted only when:

- the configured task is an `anomaly_detection` task;
- feature columns are explicit and unique;
- the artifact scope is `offline_training_or_calibration_only`;
- `target_labels_used = false`;
- runtime context and contributor counts are positive.

`register_prepared_live_diagnosis(...)` then checks that the frozen artifact feature order exactly matches the deployment, that the live OPC-UA runtime belongs to the same project and source, and that the runtime is actively running. Runtime state is checked again on every inference. Stopped, failed, starting, or reconnecting runtimes fail closed rather than producing a diagnosis from a stale buffer.

## Detector and missing-feature semantics

The V1 detector is the existing frozen PCA-SPE reconstruction-error detector. Its training medians, scaler statistics, PCA basis and anomaly threshold are fixed before live registration.

The underlying frozen scorer fills a missing feature value with the **training median stored in the artifact** before scaling and PCA reconstruction. This behavior is deterministic and does not use future or live-window statistics. It is intentionally different from the Live Forecasting V1 strict multivariate path, which does not fill missing rows.

Because asynchronous OPC-UA callbacks may not form perfectly aligned multivariate rows, this median fallback is surfaced here as part of the diagnosis contract rather than treated as invisible preprocessing. A future industrial deployment may replace this with a stricter plant-specific alignment contract when synchronized acquisition is available.

## Triage semantics

Live Diagnosis V1 reports:

- latest anomaly status;
- anomaly score and frozen threshold;
- score-to-threshold ratio;
- trailing anomalous-point persistence;
- anomaly count in the current window;
- deterministic severity band: `normal`, `warning`, `high`, or `critical`;
- top sensor reconstruction-error contributors.

Severity is an **operator triage heuristic**, not a calibrated failure probability. The default escalation bands use score ratio and persistence and are explicitly stored in the request metadata.

## Noncausal boundary

The per-sensor contributor values indicate which variables contribute most to the PCA reconstruction error for the current evidence window. They do **not** establish physical causality, fault origin, or root cause.

Every diagnosis record therefore carries:

```text
claim_scope = statistical_reconstruction_contribution_not_causal_root_cause
```

The UI repeats this boundary. A future causal or physics-based diagnosis module would require separate evidence and validation rather than relabeling reconstruction contributions as causes.

## Product API

The live FastAPI composition exposes:

- `GET /v1/diagnosis/capabilities`
- `POST /v1/projects/{project_id}/diagnosis/{task_name}/deployment`
- `GET /v1/diagnosis/applications`
- `GET /v1/diagnosis/applications/{application_id}`
- `POST /v1/diagnosis/applications/{application_id}/infer`
- `DELETE /v1/diagnosis/applications/{application_id}`

HTTP artifact/model registration is intentionally disabled in V1. The service consumes an already prepared artifact in-process; arbitrary requests cannot trigger PCA fitting or threshold selection.

## Studio

IndusTSFM Studio shows Live Diagnosis application/project/source/task/runtime binding, feature set, latest severity, inference count and observation cutoff. Operators can run inference and inspect the complete diagnosis JSON.

The operator boundary is explicit: statistical reconstruction contribution only, no causal root-cause proof, no online fitting, no threshold updates, no artifact adaptation and no control action.

## OPC-UA safety boundary

Live Diagnosis V1 does not change the connector security model. Production OPC-UA code remains browse/read/subscribe only. No diagnosis result can write a tag, setpoint, PLC command, or closed-loop action.

Writes in the asyncua smoke server are test-fixture operations used only to simulate changing PLC telemetry.

## CI and real live smoke

The real smoke path is extended to:

```text
server on
-> subscribe
-> data changes
-> disconnect
-> reconnect/resubscribe
-> LiveBuffer
-> WindowProvider
-> Live Forecasting inference
-> frozen PCA-SPE Live Diagnosis inference
-> forecast + diagnosis artifacts
```

The smoke verifies that diagnosis consumes a live window and preserves the no-online-training, frozen-threshold, no-control and noncausal-claim boundaries.

## Deliberately out of scope

- causal root-cause identification;
- online PCA fitting or adaptation;
- online threshold tuning;
- supervised fault classification from target-stream labels;
- OPC-UA writes or closed-loop control;
- optimization/control decisions;
- MQTT;
- Agent actions.
