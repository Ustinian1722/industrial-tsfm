# Live Forecasting V1

Live Forecasting V1 is the first online forecasting product slice above the frozen read-only OPC-UA runtime. It deliberately reuses the existing validation-only model-routing and research/evaluation boundaries instead of introducing a second online model-selection path.

## Runtime path

```text
Project / Task
  -> validation-only Model Route
  -> ForecastingDeploymentSpec
  -> prepared/frozen ForecastModel
  -> OPC-UA LiveRuntime / WindowProvider
  -> LiveForecastingApplication
  -> point forecast
  -> optional frozen conformal interval
  -> Studio / API
```

The live layer does **not** call `fit`, re-run AutoModel, consume future target labels, or recalibrate uncertainty from the live stream.

## Asynchronous OPC-UA semantics

OPC-UA data-change callbacks for different tags are not assumed to arrive as synchronized process rows. The default `per_tag_univariate` mode therefore forecasts each requested tag from its own arrival-order observation stream.

This avoids manufacturing a multivariate sample by silently aligning timestamps, forward filling, interpolating, resampling, or sorting observations. Bad-quality samples are rejected by default. Nonchronological evidence is rejected rather than repaired.

For models that require genuinely multivariate aligned inputs, `strict_multivariate` mode requires complete rows supplied by the upstream runtime/data contract. Missing values are not filled by the forecasting layer.

## Forecast timestamps

A future timestamp is emitted only when the observed context has a strictly regular positive cadence. If cadence is irregular or cannot be established, the forecast value is still allowed but its future timestamp is `null`.

This is intentional: the platform must not imply a sampling schedule that was not observed or configured.

## Deployment contract

`ForecastingDeploymentSpec` can only be built from a route satisfying:

- `route_ready = true`;
- `selection_evidence = validation_only`;
- `target_labels_used = false`;
- forecasting task with explicit context length and horizon.

The persisted deployment also records the workspace `project_id`. Runtime binding checks both that project ID and the configured source name, preventing a live runtime from another project from being attached accidentally.

The deployment artifact stores references and metadata. It does not load a model or checkpoint itself.

`register_prepared_live_forecast(...)` binds an already prepared model to an OPC-UA runtime through `WindowProvider`. Model preparation remains outside the live inference boundary. Registration requires an actively running runtime, and the provider re-checks runtime state on every inference. Stopped, failed, starting, or reconnecting runtimes fail closed instead of emitting a forecast from a stale buffer.

## Uncertainty quantification

Live Forecasting V1 supports a frozen split-conformal artifact using absolute residuals. Calibration is performed from an explicit offline calibration residual set with a finite-sample conformal quantile. The artifact stores per-target, per-horizon interval radii.

At runtime:

- interval radii are immutable;
- `online_updates = false`;
- no live residuals are used to tune coverage;
- target/horizon shape mismatches are rejected.

This is empirical predictive coverage, not a claim of physical uncertainty decomposition.

## Product API

The live FastAPI composition exposes:

- `GET /v1/forecasting/capabilities`
- `POST /v1/projects/{project_id}/forecasting/{task_name}/deployment`
- `GET /v1/forecasting/applications`
- `GET /v1/forecasting/applications/{application_id}`
- `POST /v1/forecasting/applications/{application_id}/infer`
- `DELETE /v1/forecasting/applications/{application_id}`

HTTP model registration is intentionally disabled in V1. This prevents an arbitrary request from triggering model fitting, adaptation, or model selection inside the online service.

## Studio

IndusTSFM Studio shows live forecasting application status, source/runtime binding, model metadata, mode, targets, UQ state, inference count, last observed timestamp, latency, and an explicit inference preview.

The operator UI also states the relevant safety/evaluation boundaries: no online training, no online model selection, no online UQ calibration, and no invented future timestamps for irregular cadence.

## OPC-UA safety boundary

Nothing in Live Forecasting V1 changes the OPC-UA security model. Product code remains browse/read/subscribe only. There is no tag write, setpoint, PLC command, or closed-loop control API.

Writes in the local asyncua smoke server are test-fixture operations used only to simulate changing PLC values. They are not client/product capabilities.

## CI boundary

The real asyncua smoke validates:

```text
server on
-> subscribe
-> data changes
-> server disconnect
-> runtime reconnecting
-> server returns
-> resubscribe
-> post-reconnect data
-> LiveBuffer
-> WindowProvider
-> LiveForecastingApplication
-> forecast artifact
```

CI uses a dependency-free deterministic persistence model as a contract test double. It does not download or initialize a TSFM checkpoint.

## Deliberately out of scope

- OPC-UA writes or closed-loop control;
- online model training/adaptation;
- online AutoModel/model selection;
- online conformal recalibration;
- MQTT;
- optimization/control;
- Agent actions.
