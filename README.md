# IndusTSFM: Industrial Time-Series Foundation Model Adaptation & Generalization

中文版：[README_zh.md](README_zh.md)

[Apache-2.0](LICENSE) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Release checklist](docs/open_source_release_checklist.md)

IndusTSFM is a reproducible platform for studying time-series foundation
models on industrial multivariate sequences. It deliberately excludes battery
data. The central question is how pretrained models behave under target-data,
entity, operating-regime, domain, and compute shifts, and how their forecasts
can be adapted and assessed.

## Current milestone: Adaptation & Generalization Platform

The repository now contains a complete, configuration-driven and auditable
platform rather than a single benchmark script. Its core loop is:

Phase 1 provides the frozen baseline loop:

`C-MAPSS -> train-only scaling -> entity-safe windows -> Naive/PatchTST/TimesFM/Chronos/Chronos-2 -> forecast -> MAE/RMSE/MASE -> CSV/JSON artifacts`

Phase 2 adds a separate, leak-audited target-support loop:

`target observed prefix -> nested support budgets -> target scaling / PatchTST fine-tuning / residual calibration -> final-window evaluation`

Phase 3 adds a separate official TimesFM Transformers+PEFT loop:

`raw C-MAPSS windows -> TimesFM 2.5 Transformers + LoRA -> source/target adapter -> final-window evaluation`

The cross-regime milestone adds a strict source-to-target generalization loop:

`FD001 source entities -> FD002/FD003/FD004 target entities -> source-only scaling -> four-model comparison -> per-subset audit`

The cross-domain milestone adds a non-battery industrial benchmark:

`UCI Gas Turbine 2011/2012 -> 2013 validation -> 2014/2015 rolling target evaluation -> source-only scaling -> audit`

The first process-domain extension adds Tennessee Eastman IDV1–15 and the
native multivariate Moirai-2.0-R-small adapter:

`IDV1–8 source -> IDV9–10 validation -> IDV11–15 target -> validation-only selection -> shift/gap audit`

The mechanical-vibration extension adds an opt-in NASA/IMS ingestion contract:

`raw vibration files -> per-file RMS/std/kurtosis/crest factor -> bearing entities -> 1st/2nd/3rd-test source/validation/target split`

The platform also exposes a lightweight automatic adaptation layer:

`validation evidence + target support amount + shift/OOD score + compute budget -> model ranking -> zero-shot / scaling / calibration / fine-tuning / PEFT decision`

This repository is prepared for public development. Original project code is
licensed under Apache-2.0. Datasets and pretrained checkpoints are not
redistributed here; fetch them from their upstream sources and follow their
individual terms. CI and tests use synthetic fixtures and do not download
external data or model weights.

The public project name is **IndusTSFM**. The Python distribution name
(`industrial-tsfm`), import package (`industrial_tsfm`), and existing CLI names
remain unchanged for compatibility.

The first task is sensor/operating-condition forecasting rather than RUL regression. C-MAPSS `RUL_FD001.txt` is retained for a later predictive-maintenance task, but is not used as a forecasting target because future sensor ground truth and RUL are different tasks. The exact protocol is documented in [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Setup

The current workspace has no checked-in data. NASA's catalog currently marks the original C-MAPSS download as unavailable, so the repository provides a fetch helper that retrieves an unmodified copy from a public mirror and records the source in the run manifest. Do not commit the raw data.

```powershell
python -m pip install -e ".[dev]"
python scripts/fetch_cmapss_fd001.py --accept-mirror
python scripts/fetch_cmapss_fd001.py --accept-mirror --subsets FD001,FD002,FD003,FD004
python -m pytest -q
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast.yaml
python -m industrial_tsfm.adapt_cli --config configs/cmapss_fd001_adaptation.yaml
python scripts/audit_phase2_run.py results/<phase2_run_id>
python -m industrial_tsfm.peft_cli --config configs/cmapss_fd001_timesfm_peft.yaml
python scripts/audit_peft_run.py results/<peft_run_id>
python -m industrial_tsfm.cross_regime_cli --config configs/cmapss_cross_regime.yaml
python scripts/audit_cross_regime_run.py results/<cross_regime_run_id>
python scripts/fetch_gas_turbine.py --accept-mirror
python -m industrial_tsfm.cross_domain_cli --config configs/gas_turbine_cross_domain.yaml
python scripts/audit_cross_domain_run.py results/<cross_domain_run_id>
python scripts/analyze_run.py results/<cross_regime_or_domain_run_id>
python scripts/audit_analysis.py results/<cross_regime_or_domain_run_id>
```

If the default user cache is on a small system drive, place model downloads on a data drive before the first TSFM run:

```powershell
$env:HF_HOME = "D:\IndustrialTSFM\hf_cache"
$env:HF_HUB_CACHE = "D:\IndustrialTSFM\hf_cache\hub"
$env:HF_HUB_DISABLE_XET = "1"
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast.yaml
```

Install optional integrations only when you need them:

```powershell
python -m pip install -e ".[tsfm]"    # TimesFM and Chronos
python -m pip install -e ".[moirai]"  # Moirai-2 / Uni2TS
python -m pip install -e ".[peft]"    # TimesFM Transformers + LoRA
```

The model checkpoints are downloaded lazily from their official Hugging Face model identifiers on first use. The default Phase-1 identifiers are:

* TimesFM: `google/timesfm-2.5-200m-pytorch`
* Chronos original: `amazon/chronos-t5-tiny`
* Chronos-2: `amazon/chronos-2` (native multivariate, p50 point forecast)
* Moirai-2: `Salesforce/moirai-2.0-R-small` (optional Uni2TS wide multivariate adapter)

For a fast local verification without the large pretrained checkpoints, run:

```powershell
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast.yaml --models naive,patchtst
```

To run the newly added native multivariate model alone:

```powershell
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast.yaml --models chronos2
```

For a three-seed Phase-1 profile with automatic mean/std/95% CI aggregation:

```powershell
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast_multiseed.yaml
```

For the three-seed Tennessee Eastman process-domain profile:

```powershell
python scripts/fetch_tennessee_eastman.py --accept-source
python -m industrial_tsfm.cross_domain_cli --config configs/tennessee_eastman_cross_domain_multiseed.yaml --models naive,patchtst,moirai2
python scripts/audit_cross_domain_run.py results/<te_run_id>
python scripts/analyze_run.py results/<te_run_id>
```

For the opt-in NASA/IMS bearing profile, record provenance first and download
the large archive only when local storage is available:

```powershell
python scripts/fetch_ims_bearings.py --accept-source --metadata-only
python scripts/fetch_ims_bearings.py --accept-source
python -m industrial_tsfm.cross_domain_cli --config configs/ims_bearings_cross_domain.yaml --models naive,patchtst,moirai2
python scripts/audit_cross_domain_run.py results/<ims_run_id>
```

The feature and split contract is documented in
[docs/ims_bearings_protocol.md](docs/ims_bearings_protocol.md); the raw archive
is ignored by Git and is not downloaded by tests.

Cross-domain and cross-regime runs also write validation-only `selection.json`,
source-to-target `shift_summary.json`, and validation-to-target
`generalization_gap.csv`. Phase-2 adaptation writes a label-free
`strategy_recommendations.json` artifact.

Build the static demo view from a validated run:

```powershell
python scripts/run_adaptation_engine.py results/<run_id> --target-fraction 0.10 --support-windows 20
python scripts/build_industrial_demo.py results/<run_id>
python scripts/audit_demo.py results/<run_id>/demo/index.html
```

The adaptation engine ranks models from the validation table only, derives a
shift score from `shift_summary.json`, and chooses a support-aware strategy.
It records the selected model, strategy, evidence, resource constraints, and
`target_labels_used: false` in `engine_decision.json`. It is a transparent
decision layer over the existing runners, not a hidden test-set optimizer.

## Repository layout

* `src/industrial_tsfm/data`: C-MAPSS/IMS parsing, entity split, window construction, and train-only preprocessing.
* `src/industrial_tsfm/models`: a common forecasting interface plus Naive, PatchTST, TimesFM, original Chronos, native multivariate Chronos-2, and optional Moirai-2 adapters.
* `src/industrial_tsfm/adaptation.py`: target-support budgets, target scaling, few-shot PatchTST fine-tuning, and post-hoc residual calibration.
* `src/industrial_tsfm/peft_experiment.py`: official TimesFM Transformers+PEFT LoRA source and target-support adapters.
* `src/industrial_tsfm/cross_regime.py` and `src/industrial_tsfm/cross_domain.py`: strict generalization runners for C-MAPSS regime shifts, UCI Gas Turbine, Tennessee Eastman, and the opt-in IMS feature profile.
* `src/industrial_tsfm/selection.py`: validation-only model ranking, generalization gaps, and support-aware strategy recommendations.
* `src/industrial_tsfm/adaptation_engine.py`: validation/shift/budget-based model and strategy decision API.
* `src/industrial_tsfm/evaluation`: leakage-safe metrics and artifact generation.
* `configs`: versioned experiment settings.
* `docs`: research decisions, extension priorities, and the fixed Phase-1/Phase-2/Phase-3/cross-regime/cross-domain/analysis/selection protocols.
* `tests`: synthetic unit/integration smoke tests only; synthetic data are never reported as research evidence.

No benchmark result is hard-coded. A result table is valid only after a run has produced the raw inputs, resolved configuration, model versions, split manifest, metrics, and predictions in one output directory.

The maintainer workspace contains audited reference profiles for Phase 1,
Phase 2/PEFT, C-MAPSS cross-regime, UCI Gas Turbine, and the three-seed
Tennessee Eastman profile. Result directories are intentionally ignored from
the public repository because they can contain large predictions and source
manifests. Recreate a profile locally with the commands above, then run the
corresponding audit and `scripts/build_industrial_demo.py`.

The latest regression status is `22 passed`, Ruff clean, and `compileall` clean. The test suite uses synthetic data and does not download raw IMS or model checkpoints.
