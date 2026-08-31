# Tennessee Eastman cross-domain protocol

This profile adds a process-industry domain without changing the existing
forecasting interface. The source is the University of Washington Tennessee
Eastman Challenge Archive, which provides IDV disturbance recordings and a
[format description](https://depts.washington.edu/control/LARRY/TE/IDVs/format.txt).
The archive page describes each disturbance as a 50-hour simulation sampled
every 10 minutes; `y.dat` contains measured outputs and `t.dat` contains time.

## Fixed data contract

`scripts/fetch_tennessee_eastman.py` downloads the official `idv1.zip` through
`idv15.zip` URLs only after `--accept-source` is supplied. It extracts the
fixed `y.dat` and `t.dat` members, checks shapes and monotonic time, and writes
`source_manifest.json` with archive/member SHA256 hashes and provenance. Raw
archives are ignored by Git and are not redistributed because the archive page
does not state an SPDX license.

The default feature set is the 41 measured variables in `y.dat`, named
`xmeas_001`–`xmeas_041`. The ten derived outputs remain available through
`feature_set: all_outputs`, but that is a separate configuration choice.

## Pre-registered split

| Role | Disturbances | Use |
|---|---|---|
| Source train | IDV1–IDV8 | scaler fit, supervised model fit |
| Validation | IDV9–IDV10 | early stopping and validation-only model selection |
| Target test | IDV11–IDV15 | final zero-shot/generalization report |

The split is by disturbance/entity, not by random rows. The source standard
scaler and MASE denominators are fitted only on IDV1–IDV8. Each target entity
uses context 32 and horizon 8 at its final valid origin. No target fitting,
target scaling, calibration, or selection uses the final target horizon in the
cross-domain runner.

## Model matrix

The first profile compares Naive, PatchTST, and Moirai-2. Moirai-2 uses the
official Uni2TS wide multivariate predictor and p50 point quantile through the
`Moirai2Model` adapter. The official code is available at
<https://github.com/SalesforceAIResearch/uni2ts>; the small checkpoint is
<https://huggingface.co/Salesforce/moirai-2.0-R-small>. Uni2TS 2.0.0 declares
an older PyTorch range than this workspace, so the run records package
versions and keeps Moirai as an optional dependency.

## Research-hardening artifacts

Every current cross-domain run writes:

* raw MAE/RMSE, MASE, source-standardized normalized MAE/RMSE, and macro-entity variants;
* `validation_result_table.csv` and `selection.json`, where selection uses validation only;
* `shift_summary.json`, containing source-to-target feature mean shifts and dispersion ratios;
* `generalization_gap.csv` and `generalization_gap_summary.csv`, joining validation and target metrics by model/seed;
* prediction arrays, resolved config, split/scaler manifests, package versions, and raw hashes.

The three-seed reference command is:

```powershell
$env:HF_HOME = "D:\IndustrialTSFM\hf_cache"
python scripts/fetch_tennessee_eastman.py --accept-source
python -m industrial_tsfm.cross_domain_cli `
  --config configs/tennessee_eastman_cross_domain_multiseed.yaml `
  --models naive,patchtst,moirai2
python scripts/audit_cross_domain_run.py results/<run_id>
python scripts/analyze_run.py results/<run_id>
python scripts/audit_analysis.py results/<run_id>
```

The current three-seed artifact is
`results/20260817T193124Z_1263d3db69/`. Its validation-only selector chose
Moirai-2, while the held-out target table favored PatchTST; this is retained as
an explicit generalization-gap result rather than treated as a contradiction.
