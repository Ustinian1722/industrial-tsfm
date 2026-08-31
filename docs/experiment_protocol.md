# Phase-1 C-MAPSS forecasting protocol

This is the fixed protocol for the first result table. Any change to a value below creates a new experiment configuration and must not overwrite an existing result directory.

## Data contract

The loader expects whitespace-delimited C-MAPSS files with 26 numeric columns:

`unit_id, cycle, setting_1..setting_3, sensor_1..sensor_21`.

Rows are sorted by `(unit_id, cycle)` after loading. The loader checks finite values, strictly increasing cycle values within each unit, and the presence of the expected train/test/RUL files. RUL is loaded and validated for provenance but is not used by the Phase-1 forecasting target.

The model feature set is all three operating settings plus all 21 sensors. A later ablation may define a sensor-only feature set, but it must be a separate configuration.

## Entity split

The official C-MAPSS train and test files are treated as separate entity pools. Numeric IDs are qualified with their source file:

```text
train_unit_001, train_unit_002, ...
test_unit_001, test_unit_002, ...
```

The official train pool is split once with `entity_split_seed=2026` into 80% model-train and 20% validation entities. The official test pool is never used for training, scaling, early stopping, or model selection. The split manifest is written to every run directory.

## Forecast windows

For an entity with values `x[0:T]`, a window ending at origin `t` is:

* context: `x[t-context_length:t]`
* target: `x[t:t+horizon]`

Training uses every valid origin at `train_window_stride`. Validation and test use only the final valid origin per entity, so long trajectories cannot dominate the headline table. Entities shorter than `context_length + horizon` are reported as skipped rather than padded with invented values.

## Preprocessing

The standardizer is fitted once on all raw feature rows belonging to model-train entities. Its means and scales are frozen before validation/test windows are transformed. There is no per-test-entity fitting, no use of RUL labels in scaling, and no target-window statistics in model selection. Predictions are inverse transformed before metrics are computed.

The zero-shot label means that TimesFM and Chronos parameters are not updated using C-MAPSS. The frozen train-only scaler is shared with every model so scale handling is visible and reproducible; model-internal context normalization, if enabled by an official implementation, is recorded as part of that model's inference behavior. Phase 1 disables TimesFM's optional internal normalization because the external train-only scale is explicit.

## Model/evaluation matrix

| Model | C-MAPSS training | Input contract | Headline mode |
|---|---:|---|---|
| Naive | none | multivariate context, repeat last value | baseline |
| PatchTST | supervised on model-train entities | multivariate context | trained baseline |
| TimesFM | none | independent univariate series per channel | zero-shot |
| Chronos original | none | independent univariate series per channel | zero-shot |
| Chronos-2 | none | native multivariate window, p50 point forecast | zero-shot |

Every model receives the same context/horizon and the same externally transformed features. This does not claim that original Chronos or TimesFM are natively multivariate; the adapter limitation is intentionally recorded. Chronos-2 is kept as a separate native-multivariate row and is not substituted for the historical original-Chronos row.

## Metrics

All metrics use original units:

* MAE and RMSE over all forecast points and channels.
* MASE using the model-train entity one-step naive denominator per feature.
* Source-standardized normalized MAE/RMSE, dividing each feature's error by the
  frozen model-train standard deviation so channels and industrial domains can
  be compared without refitting on target data.
* Macro-entity MAE/RMSE, averaging each entity's score equally.
* Per-horizon and per-feature MAE/RMSE/MASE for failure analysis.

Compute fields are measured around actual model calls: total/trainable parameter count, train seconds, inference seconds, and peak CUDA memory where applicable.

## Artifacts

Each run writes:

```text
results/<run_id>/
  config.resolved.yaml
  manifest.json
  split_manifest.json
  scaler.json
  dataset_summary.json
  result_table.csv
  validation_result_table.csv
  selection.json
  generalization_gap.csv
  metrics_detail.csv
  predictions/<model>_seed<seed>.npz
```

If an optional model is unavailable, the default is to fail rather than emit a fake metric. `--allow-missing-models` can be used only for infrastructure checks; the run manifest then marks the model as unavailable and the result is not a complete Phase-1 table.
