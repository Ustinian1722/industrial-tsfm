# IndusTSFM extension roadmap: architecture, data, and fair comparison

This document records extension decisions after auditing the existing Phase
1/2/3, cross-regime, cross-domain, OOD, reliability, and artifact-audit loops.
Those protocols remain authoritative; a new model or dataset must not silently
change their split, scaling, or target-horizon rules.

## Model selection

| Priority | Model | Architecture/interface difference | Official checkpoint/code | License | Decision |
|---|---|---|---|---|---|
| P0 | Chronos-2 | Encoder-only, native multivariate/group attention, quantile output | `amazon/chronos-2`, `amazon-science/chronos-forecasting` | Apache-2.0 | Added first. The installed `chronos-forecasting` package already exposes `Chronos2Pipeline`. |
| P1 | Moirai 2.0-R-Small | Native multivariate patch/quantile forecaster through the GluonTS predictor | `Salesforce/moirai-2.0-R-small`, Salesforce `uni2ts` | CC BY-NC 4.0 model card; Uni2TS code Apache-2.0 | First added architecture. The 45.6 MB small checkpoint is practical, but Uni2TS 2.0.0 declares `torch<2.5`; keep it in an isolated optional environment and record that boundary. |
| P1 | MOMENT-1 | General-purpose encoder foundation model with forecasting task head and representation APIs | `AutonLab/MOMENT-1-large`, official `moment` repository | MIT | Candidate after Moirai or as a representation/OOD extension; requires a dedicated `momentfm` adapter and a fixed forecast-head contract. |
| P2 | Time-MoE | Decoder-only sparse mixture-of-experts, up to 2.4B parameters and 4096 sequence length | Official `Time-MoE/Time-MoE` repository and released checkpoints | Apache-2.0 code | Defer until a resource-bounded base checkpoint and a Transformers-version-isolated environment are validated. |
| P2 | Chronos original | Univariate tokenized autoregressive baseline | `amazon/chronos-t5-tiny` | Existing protocol | Keep unchanged for continuity; do not replace it with Chronos-2 in historical tables. |

Chronos-2 was the first model extension in the original environment. Moirai-2
is the first newly added architecture in the current extension profile. Both
create a meaningful comparison against the channel-wise TimesFM/Chronos
adapters because they can receive an industrial window as one multivariate
task. Their point forecast is the official p50 quantile; quantile outputs are
retained in model metadata for later reliability work.

## Dataset selection

| Priority | Dataset | What it adds | Main protocol risk | Decision |
|---|---|---|---|---|
| P1 | Tennessee Eastman Process | Process-industry dynamics, manipulated/process variables, operating modes and disturbance regimes | The University of Washington archive contains several simulator/code releases and does not state an SPDX license on the archive page | Implemented as an immutable IDV archive profile: official IDV1–15 ZIP URLs, extracted-member hashes, 10-minute/50-hour format checks, and a pre-registered IDV1–8 → IDV9–10 → IDV11–15 split. Raw archives remain local and non-redistributed. |
| P2 | IMS Bearings (NASA/IMS) | Mechanical vibration and run-to-failure behavior with a public NASA catalog entry | The official ZIP is about 1.06 GB; raw high-frequency files are not directly comparable to hourly/sensor forecasting; entity must be bearing/run and feature extraction must be fixed before splitting | Contract implemented: streaming per-file RMS/std/kurtosis/crest-factor features, bearing entities, 1st/2nd/3rd-test condition split, provenance fetcher, and loader tests. Raw benchmark download/run remains opt-in. |
| P2 | Paderborn Bearing | Rich vibration/current signals and multiple operating conditions | The official data are CC BY-NC 4.0 and very large; commercial/demo redistribution is restricted | Keep as a later non-commercial research profile, not a default download. |

The Tennessee Eastman implementation uses only the University of Washington
Challenge Archive's IDV1–IDV15 disturbance recordings. The archive format says
that each IDV is a 50-hour simulation sampled every 10 minutes and stores
measured outputs in `y.dat`; the fetcher records source URLs, archive hashes,
member hashes, shapes, sampling metadata, and the archive's non-SPDX-license
warning. A GitHub mirror or Kaggle copy is not accepted as a benchmark source.

## Fair benchmark contract

Every new model on an existing domain must use:

1. The same entity-safe windows, context length, forecast horizon, source-only
   scaler, MASE denominator, and final target origins as the corresponding
   existing protocol.
2. A model-specific input-contract field recording whether channels are jointly
   modeled, independently modeled, or treated as covariates.
3. Point metrics in raw units (MAE/RMSE), MASE, source-standardized normalized
   error, and
   macro-entity metrics. Quantile models additionally save p10/p50/p90 when the
   official interface provides them.
4. Runtime evidence: total/trainable parameters, fit and inference seconds,
   peak GPU memory, checkpoint ID, package versions, and model metadata.
5. The same seed list and no target-label model selection. A result is not
   promoted to a headline table until all requested seeds complete and the
   artifact audit recomputes its predictions and metrics.

The runners write `result_summary.csv` next to `result_table.csv`. It contains
`n_seeds`, mean, sample standard deviation, and a normal-approximation 95% CI
for each reported metric. When only one seed is present, the standard
deviation is zero for bookkeeping but the CI fields are blank by design.

For Chronos-2 specifically, the first comparison is a bounded FD001 smoke and
then the full cross-regime/cross-domain profiles. It is reported as a native
multivariate model, not merged into the historical univariate Chronos row.

## Source links

- Chronos-2 model card: <https://huggingface.co/amazon/chronos-2>
- Chronos official code: <https://github.com/amazon-science/chronos-forecasting>
- Moirai/Uni2TS official code: <https://github.com/SalesforceAIResearch/uni2ts>
- Moirai 2.0-R-Small model card: <https://huggingface.co/Salesforce/moirai-2.0-R-small>
- MOMENT model card: <https://huggingface.co/AutonLab/MOMENT-1-large>
- Time-MoE official code: <https://github.com/Time-MoE/Time-MoE>
- Tennessee Eastman provenance archive: <https://depts.washington.edu/control/>
- NASA IMS Bearings catalog: <https://catalog.data.gov/dataset/ims-bearings>
- NASA PCoE data-set repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Paderborn Bearing DataCenter: <https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter>

## Current extension status

The first extension implementation is complete at the code level and has a
verified three-seed reference run:

- `Moirai2Model` uses the official Uni2TS wide multivariate predictor and p50
  quantile while preserving the platform's fixed-window interface.
- `load_tennessee_eastman` implements the official IDV archive format and the
  pre-registered source/validation/target disturbance split.
- `scripts/fetch_tennessee_eastman.py` requires explicit source acceptance and
  writes a provenance/checksum manifest; it does not put raw data in Git.
- `cross_domain_cli` and `audit_cross_domain_run.py` use the same dataset
  contract for Gas Turbine and Tennessee Eastman.
- Cross-domain and cross-regime runners write validation-only `selection.json`,
  source-to-target `shift_summary.json`, and validation-to-target
  `generalization_gap.csv` artifacts.
- Phase 2 supports Moirai-2 zero-shot target scaling and residual calibration,
  and writes a label-free `strategy_recommendations.json` policy artifact.
- `load_ims_bearings` and `scripts/fetch_ims_bearings.py` define the mechanical
  vibration ingestion boundary without silently downloading or redistributing
  the large raw archive. The exact feature and split contract is in
  `docs/ims_bearings_protocol.md`.

The current TE reference is
`results/20260817T193124Z_1263d3db69/`; its cross-domain and post-run analysis
audits pass on the same immutable raw file set. It remains a research profile,
not a universal ranking: the validation selector chose Moirai-2 while the
held-out target table favored PatchTST, making the generalization gap an
explicit result.
