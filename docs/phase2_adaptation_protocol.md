# Phase-2 target-support adaptation protocol

Phase 2 extends the fixed Phase-1 FD001 forecasting task without changing the final evaluation target. The purpose is to measure how much target-domain information helps, while keeping the final target horizon unavailable to every adaptation operation.

## Fixed evaluation

The dataset, entity split, feature set, context length, horizon, source scaler, MASE denominator, and final-window evaluation are inherited from [experiment_protocol.md](experiment_protocol.md): all three settings plus all 21 sensors, context length 16, horizon 8, and one final valid window for each of the 100 official test entities. The source standardizer is fitted on the 80 model-train entities and remains the metric reference.

For each test entity, the final forecast starts at its last valid origin. The target support prefix is restricted to rows with `cycle < final_origin`; therefore it includes the available context but none of the final future target rows. One short entity cannot form a historical context-plus-horizon support window. It remains in the final evaluation pool but is excluded from the support candidate pool. In the recorded FD001 run this leaves 99 support candidates and 1 support-ineligible test entity.

## Support budgets

Support candidates are permuted once with `support_seed=2026`; every budget takes the first `n` entities from that same permutation, so budgets are nested rather than independently cherry-picked. The default budgets are 0%, 1%, 5%, 10%, 25%, 50%, and 100% of the 99 eligible candidates. Historical support windows use the same context and horizon as the final task and `support_window_stride=32`.

The final target horizon is never used to fit a scaler, fine-tune a model, or estimate a calibration bias. Every run writes `support_manifest.json`, including support IDs, evaluation origins, observed-prefix row counts, and support-window counts.

## Adaptation variants

* `naive / none` is a single persistence reference using the source scaler.
* `patchtst / few_shot_finetune` starts from the Phase-1 source-trained PatchTST checkpoint and continues supervised training only on target support windows under the frozen source scaler. The adaptation epochs, batch size, learning rate, and weight decay are fixed in YAML; no final-target validation or target hyperparameter search is performed.
* `timesfm`, `chronos`, and `chronos2` remain frozen zero-shot models. At nonzero budgets they receive a target-support scaler fitted only on observed prefixes. This is a controlled normalization experiment; Chronos-2 is natively multivariate while the TimesFM/original-Chronos adapters remain channel-wise.
* `target_residual_calibration` fits a feature-wise additive residual bias from frozen-model forecasts on historical support windows and applies that bias to the final forecast. It is explicitly post-hoc output calibration, not parameter-efficient fine-tuning.

Every nonzero support budget reports both `all_test` and `heldout_target` metrics. The latter excludes support entities and is the more conservative cross-entity transfer view. At 100% support, the held-out view contains only the short ineligible entity, so it must not be interpreted as a stable estimate.

## Artifacts and audit

Each Phase-2 run contains:

* `support_manifest.json`: support budgets, origins, candidate/ineligible pools, and observed-prefix accounting;
* `source_scaler.json` and `scalers/support_*.json`: exact fit scopes and statistics;
* `predictions/*.npz`: raw-unit targets, predictions, entity IDs, origins, features, and support IDs;
* `calibrators/*.json`: residual-bias coefficients and their support scope;
* `result_table.csv` and `metrics_detail.csv`: aggregate and granular metrics;
* `manifest.json`: package versions, device, raw-file hashes, shapes, and the no-final-target-use declaration.

Run the artifact-only audit after a real run:

```powershell
python scripts/audit_phase2_run.py results/<phase2_run_id>
```

The audit recomputes raw-file hashes, final origins, observed prefixes, support scalers, target shapes, and metrics from the saved NPZ files. It does not treat a missing model or an unavailable checkpoint as a benchmark result.

## PEFT boundary

The official TimesFM LoRA example uses the Transformers checkpoint `google/timesfm-2.5-200m-transformers`, PEFT, raw values with the model's internal instance normalization, and a context length that is a multiple of 32. Phase 1/2 use the official PyTorch forecast checkpoint `google/timesfm-2.5-200m-pytorch` with an explicit external scaler, so the two interfaces are not silently mixed. TimesFM PEFT is the next adaptation milestone and requires its own context-32 protocol and adapter-specific audit. Chronos fine-tuning likewise remains a later, heavier experiment based on the official training scripts.
