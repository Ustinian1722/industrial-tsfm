# UCI Gas Turbine cross-domain protocol

This is the non-battery industrial-domain profile. It evaluates the same forecasting interface on the UCI Gas Turbine CO/NOx Emission Data Set, which contains hourly multivariate measurements from 2011–2015. The UCI protocol specifies the first three years for training/cross-validation and the last two years for testing; this implementation makes that boundary explicit as source years, validation year, and target years.

## Data and provenance

- Dataset: UCI Gas Turbine CO and NOx Emission Data Set, dataset 551, DOI `10.24432/C5WC95`.
- Features: `AT`, `AP`, `AH`, `AFDP`, `GTEP`, `TIT`, `TAT`, `TEY`, `CDP`, `CO`, and `NOX`.
- The `year` field is used as the entity/split key and is not forecast as a sensor feature.
- Source years: 2011 and 2012.
- Validation year: 2013.
- Target years: 2014 and 2015.
- The local UCI archive endpoint was unavailable from this workspace because of a TLS handshake failure. The reproducible run therefore uses the public [skforecast mirror CSV](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/turbine_emission.csv), which identifies the UCI dataset as its source. `scripts/fetch_gas_turbine.py --accept-mirror` records the mirror URL, upstream UCI URL, source SHA256, row counts, and the derived yearly-file hashes in `data/raw/gas_turbine/source_manifest.json`.
- The mirror provides chronological timestamps for the otherwise date-free UCI rows. The fetcher uses those timestamps only to split the combined file into yearly entities; it does not impute or alter sensor values.

The official UCI metadata reports 36,733 instances, 11 sensor measures, no missing values, and the first-three/last-two-year protocol. The downloaded mirror has the same 36,733 rows and passes the loader’s finite-value and per-year checks. The run is a mirror-backed reproduction, not a claim that the unavailable UCI archive was downloaded locally.

## Forecasting and leakage boundary

- Context/horizon: 32 observed rows followed by an 8-row forecast.
- Source train windows use stride 1 from 2011–2012; 2013 supplies validation windows for supervised PatchTST selection.
- Target evaluation uses non-overlapping-ish rolling origins every 168 rows (one week) across 2014–2015, plus the final valid origin for each target year. This gives more than one evaluation window per target year while preserving chronological order.
- One standard scaler is fitted only on 2011–2012 source entities. Validation and target values are transformed with that fixed scaler.
- Naive has no fit; PatchTST is trained on source windows; TimesFM, original Chronos, and Chronos-2 are frozen zero-shot models. No 2014/2015 values are used for fitting, preprocessing, model selection, or calibration.
- Metrics are raw-unit MAE/RMSE, source-train MASE, source-standardized normalized MAE/RMSE, macro-entity variants, and horizon/feature detail rows. Prediction arrays and all split/origin metadata are retained for audit. New runs also write validation-only selection, shift, and generalization-gap artifacts.

## Run

```powershell
python scripts/fetch_gas_turbine.py --accept-mirror
python -m industrial_tsfm.cross_domain_cli --config configs/gas_turbine_cross_domain.yaml
python scripts/audit_cross_domain_run.py results/<cross_domain_run_id>
```

For a cheap smoke run:

```powershell
python -m industrial_tsfm.cross_domain_cli --config configs/gas_turbine_cross_domain.yaml --models naive,patchtst
```

## Interpretation boundary

This profile is a chronological domain-shift test within one gas-turbine dataset and an external industrial benchmark for the pretrained TSFMs. It is not a cross-plant test, because the UCI data describe one plant, and it does not establish universal model rankings. Results should be read alongside the C-MAPSS cross-regime profile and later OOD/reliability analyses.

## Authoritative local profile

The full one-seed profile is `results/20260817T170355Z_a1f73fd9bc/`; its audit completed with `CROSS_DOMAIN_AUDIT_OK`. It contains 64 target windows from two target-year entities (52 regular origins plus the final origin in 2014, and 10 regular origins plus the final origin in 2015).

| Model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| Naive | 5.1793 | 10.4331 | 2.9372 |
| PatchTST | 3.7881 | 7.1021 | 2.3375 |
| TimesFM | 4.0434 | 8.4874 | 2.3144 |
| Chronos | 4.3671 | 9.2032 | 2.4346 |

PatchTST has the lowest raw MAE/RMSE in this profile, while TimesFM has the lowest source-train MASE by a small margin. These are preliminary one-seed, source-year-to-target-year profiles, not a universal ranking or a claim of cross-plant transfer.

## Sources

- [UCI dataset 551 metadata](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- [UCI DOI record](https://doi.org/10.24432/C5WC95)
- [Mirror dataset documentation](https://skforecast.org/latest/user_guides/datasets#turbine_emission)
- [Mirror CSV](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/turbine_emission.csv)
