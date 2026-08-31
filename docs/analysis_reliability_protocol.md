# Failure analysis, OOD, and reliability protocol

Post-run analysis operates only on an already audited run and reconstructs its contexts from the resolved config and raw-file hashes. It never refits a model or uses target labels to choose preprocessing, model parameters, or intervals.

## Failure analysis artifacts

Run:

```powershell
python scripts/analyze_run.py results/<cross_regime_or_domain_run_id>
python scripts/audit_analysis.py results/<cross_regime_or_domain_run_id>
```

The files are written under the run’s `analysis/` directory:

- `feature_error_summary.csv`: raw-unit MAE/RMSE, signed bias, and source-train MASE by model, target, and feature.
- `horizon_error_summary.csv`: error and bias at each forecast step.
- `entity_error_summary.csv`: per-entity error, window count, and OOD score.
- `ood_feature_summary.csv`: source-standardized deviation by target and feature.
- `ood_window_summary.csv`: context-only OOD scores by target entity and origin.
- `context_representation_summary.csv`: a two-dimensional PCA projection of standardized target contexts, with PCA fitted only on source-train contexts. This is an interpretable context representation, not a hidden state extracted from a foundation model.
- `error_by_ood_bin.csv`: target-window errors grouped into within-target OOD quartiles.

The OOD score is the RMS of the source-train standardized context values. The scaler is fitted on the source model-train entities/years only. A very large score can be scientifically meaningful under a fixed source scale—for example, a target operating setting that is nearly constant in the source regime—but should not be interpreted as a calibrated probability of failure.

## Reliability intervals

`reliability_summary.csv` and `reliability_feature_summary.csv` apply a 90% source-validation absolute-residual interval to each saved target prediction. The residual calibration set is a Naive last-value forecast on all rolling windows from the source validation entities/year. Quantiles use the higher empirical quantile; the interval is deliberately labeled `source_validation_naive_absolute_residual` because it is not a model-specific probabilistic forecast and no target calibration is performed.

## Current observations

For the authoritative C-MAPSS cross-regime run, FD002 and FD004 have median context RMS OOD scores around 3,197 and 3,244 under the source FD001 scale, while FD003 is around 1.14. The source-calibrated intervals cover only about 1.9–2.7% of FD002/FD004 PatchTST/TimesFM windows and about 15.7–16.0% for Chronos, making the domain shift visible as severe under-coverage. FD003 coverage is much closer to nominal for the frozen models (about 90.7–92.6%). These intervals are diagnostic, not deployment guarantees.

For the authoritative UCI Gas Turbine run, the target context RMS OOD score has median 0.97 and ranges from about 0.71 to 2.09. Coverage of the same source-validation naive interval is 83.9% for Naive, 90.8% for PatchTST, 89.5% for TimesFM, and 88.2% for Chronos. Error increases across the highest OOD quartile for every model; the observed OOD–MAE Spearman correlations are approximately 0.23–0.30. These are one-seed diagnostic associations, not calibrated risk estimates.
