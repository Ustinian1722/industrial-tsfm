# C-MAPSS cross-regime and cross-device protocol

This protocol tests whether a model trained on one C-MAPSS operating regime transfers to other regimes without fitting on the target data. It uses the NASA C-MAPSS aircraft-engine simulator data; the official NASA resource and catalog entries are linked from the project research record.

## Split and preprocessing

- Source: FD001 training entities only. The source entity split remains 80% model-train and 20% validation, with entity split seed `2026`.
- Targets: official test entities from FD002, FD003, and FD004, evaluated independently.
- Context/horizon: 16 observed rows followed by an 8-row forecast.
- Evaluation origin: the final valid origin of each target test entity; entities shorter than `context + horizon` are reported as skipped.
- Features: the 3 operating settings plus the 21 sensor channels.
- Scaling: one standard scaler fitted only on source model-train entities. Target rows are transformed with that fixed scaler and are never used to estimate preprocessing statistics.
- Training: Naive has no fit; PatchTST is trained on source model-train windows; TimesFM, original Chronos, and Chronos-2 are frozen zero-shot models. No target labels, target validation rows, RUL files, or target statistics are used during model fitting.
- Reproducibility: one seed (`42`) in the authoritative run. Every prediction file stores targets, predictions, entity IDs, origins, and feature names; the manifest stores raw-file hashes and resolved package information.

The “cross-device” component is the strict entity separation: source engine IDs and target engine IDs are disjoint. The “cross-regime” component is the separate FD002/FD003/FD004 operating-condition and fault-pattern distribution. This is a generalization profile, not a claim that the target test set represents a production deployment distribution.

## Run

After the four C-MAPSS subsets have been fetched:

```powershell
python -m industrial_tsfm.cross_regime_cli --config configs/cmapss_cross_regime.yaml
python scripts/audit_cross_regime_run.py results/<cross_regime_run_id>
```

For a quick local check, select the inexpensive baselines:

```powershell
python -m industrial_tsfm.cross_regime_cli --config configs/cmapss_cross_regime.yaml --models naive,patchtst
```

## Authoritative local profile

The current full profile is `results/20260817T163507Z_9a55c0ca99/`. Its audit completed with `CROSS_REGIME_AUDIT_OK`.

| Target | Valid entities | Skipped entities | Model | MAE | RMSE | MASE |
|---|---:|---:|---|---:|---:|---:|
| FD002 | 256 | 3 | Naive | 64.6193 | 142.2993 | 1387.807 |
| FD002 | 256 | 3 | PatchTST | 82.1675 | 171.2481 | 1955.919 |
| FD002 | 256 | 3 | TimesFM | 52.4953 | 111.2372 | 1113.013 |
| FD002 | 256 | 3 | Chronos | 57.9771 | 129.5597 | 1339.523 |
| FD003 | 100 | 0 | Naive | 0.8535 | 2.3182 | 0.8719 |
| FD003 | 100 | 0 | PatchTST | 0.7560 | 1.9636 | 1.9295 |
| FD003 | 100 | 0 | TimesFM | 0.7341 | 1.9990 | 0.7438 |
| FD003 | 100 | 0 | Chronos | 0.7902 | 2.1901 | 0.8573 |
| FD004 | 242 | 6 | Naive | 62.8121 | 138.5151 | 1338.1548 |
| FD004 | 242 | 6 | PatchTST | 83.7359 | 172.8058 | 1993.9846 |
| FD004 | 242 | 6 | TimesFM | 51.5366 | 109.4009 | 1079.8157 |
| FD004 | 242 | 6 | Chronos | 55.7999 | 124.3836 | 1286.7254 |

These are preliminary one-seed profiles. FD002 and FD004 show a large source-only scale/regime shift, so their raw-unit errors are not directly comparable with FD003. In this run TimesFM and Chronos are more robust than the source-trained PatchTST and Naive baselines on those shifted subsets, while no model is uniformly best across every metric and subset. The result motivates explicit domain normalization, operating-condition conditioning, and OOD/reliability analysis; it does not establish a universal ranking.

## Provenance

- [NASA Dashlink C-MAPSS resource](https://c3.ndc.nasa.gov/dashlink/resources/139/)
- [NASA Data.gov C-MAPSS catalog entry](https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data)
- [Phase-1 forecasting protocol](experiment_protocol.md)
