# Phase-3 TimesFM Transformers + PEFT protocol

Phase 3 is intentionally separate from the Phase-1/2 PyTorch forecast adapter. It follows the official TimesFM 2.5 Transformers training interface and uses a LoRA adapter rather than silently mixing checkpoint families.

## Official interface

The implementation follows the [official TimesFM LoRA fine-tuning example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning):

* checkpoint: `google/timesfm-2.5-200m-transformers`;
* model class: `TimesFm2_5ModelForPrediction`;
* LoRA: `target_modules="all-linear"`, rank 4, alpha 8, dropout 0.05;
* raw values are passed to the model because TimesFM 2.5 performs internal instance normalization;
* the context length is 32, satisfying the official example's multiple-of-32 requirement;
* each channel is represented as an independent one-dimensional series, then predictions are reshaped back to `[batch, horizon, feature]`.

The adapter records total and trainable parameters, package versions, raw-file hashes, input contract, fit/inference time, and saved adapter paths. In the current environment the LoRA profile has 232,672,192 total parameters and 1,382,912 trainable parameters.

## C-MAPSS protocol

The source and target split follows the Phase-1 entity protocol. Final evaluation uses one final context-32/horizon-8 window per valid official test entity. Four test entities are too short for this longer final context and are reported as skipped; no padding is introduced. Of the 96 evaluated entities, 95 can also provide at least one historical support window before the final target origin. Support budgets are nested at 0%, 1%, 5%, and 10% of those 95 candidates, with `support_window_stride=32`.

The source adapter is trained only from source model-train entities. Each target adapter is initialized from the base Transformers checkpoint and trained only on its selected target support windows. Both use raw values and the internal TimesFM normalization. The final target horizon is never used for adapter fitting, and `support_manifest.json` records the origin cutoff and support membership.

The first profile is deliberately bounded (`max_train_series=512`, one epoch, one seed) to establish a real end-to-end PEFT path on the available GPU. It is an engineering/scaling profile, not a final claim about the best adapter budget.

## Recorded profile

The authoritative local run is `results/20260817T162540Z_906dfc042b/`.

| adapter | support | evaluation | MAE | MASE |
|---|---:|---|---:|---:|
| base zero-shot | 0% | all valid test entities (96) | 0.70163 | 0.55966 |
| source LoRA | source train | all valid test entities (96) | 0.69314 | 0.55704 |
| target LoRA | 1% | all valid test entities (96) | 0.70111 | 0.55958 |
| target LoRA | 5% | all valid test entities (96) | 0.70027 | 0.55925 |
| target LoRA | 10% | all valid test entities (96) | 0.70129 | 0.55945 |

The source adapter improves this bounded profile, while the three target-support adapters are effectively near the base profile. These observations are preliminary: they use one seed, one epoch, and a small series budget. The target-support heldout rows are also recorded for cross-entity inspection and should be preferred when asking whether adaptation transfers beyond the support entities.

## Audit

```powershell
python scripts/audit_peft_run.py results/<peft_run_id>
```

The audit recomputes the final windows and raw target arrays, checks skipped and support pools, verifies that support windows end before each final origin, recomputes metrics from every NPZ artifact, and verifies each LoRA adapter config and weight file. A successful audit prints `PEFT_AUDIT_OK`.
