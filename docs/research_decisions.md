# IndusTSFM research decisions recorded before implementation

## Dataset and task

NASA describes C-MAPSS as a simulator for realistic commercial turbofan engine data and describes the shared data as run-to-failure simulations with multiple operational conditions/fault modes and recorded sensor channels. The commonly distributed FD001 files have engine/unit identifiers, cycle, three operating settings, and 21 sensor columns. The loader validates the exact 26-column layout instead of silently accepting malformed rows.

The first loop is **multivariate sensor/operating-condition forecasting**. A forecast example contains a past context window of all selected settings/sensors and the next `H` observed values from the same engine. The official C-MAPSS test trajectories are truncated before failure; they therefore do not provide future sensor observations after the final test row. We use their observed prefixes for within-prefix forecasting evaluation and keep the provided RUL labels out of this loop. This keeps the target aligned with the model APIs and avoids conflating forecasting with RUL estimation.

## Split and leakage policy

* All windows are generated inside one entity; windows never cross engines.
* C-MAPSS official `train_*.txt` and `test_*.txt` entities remain disjoint. The official training entities are split into model-train and validation entities using a fixed seeded permutation.
* PatchTST sees only windows from model-train entities. Validation is used only for early stopping.
* The standard scaler is fitted only on raw values from model-train entities. Validation and official test values are transformed with those frozen statistics.
* Model selection, hyperparameter selection, and early stopping never inspect official test targets.
* The source-qualified entity key (`train_unit_###` or `test_unit_###`) is used so numeric unit IDs from two official files cannot accidentally be treated as the same entity.

## Model choices

* **Naive** is persistence: repeat the last context value for every future step.
* **PatchTST** is a supervised baseline implementing the official paper's relevant core choices: fixed-length temporal patches, a Transformer encoder, and channel-independent shared weights. It is trained only on C-MAPSS model-train entities.
* **TimesFM** uses the current official PyTorch checkpoint `google/timesfm-2.5-200m-pytorch`. The official API accepts a list of univariate input series, so the adapter forecasts each selected channel independently and restores the `[batch, horizon, channel]` shape. No target-domain gradient update is performed.
* **Chronos** retains the original univariate `amazon/chronos-t5-tiny` checkpoint rather than silently substituting Chronos-2. The original official pipeline accepts a batch of univariate contexts and returns sample paths; the adapter reports their median. Channels are processed in fixed inference batches to bound sampling memory. This keeps an explicit apples-to-apples univariate zero-shot baseline for historical Phase-1 tables.
* **Chronos-2** uses the official `amazon/chronos-2` checkpoint through `Chronos2Pipeline`. Each window is passed as one native multivariate task with `cross_learning=false`, and the p50 quantile is used as the point forecast. Its results are a separate model row because its input contract is not equivalent to the channel-wise TimesFM/original-Chronos adapters.
* **Moirai-2.0-R-small** uses the official Uni2TS wide multivariate predictor and p50 quantile. It is an optional adapter because the released Uni2TS 2.0.0 dependency boundary declares an older PyTorch range than the current workspace; the run manifest records the actual package versions and input contract.

## Metrics and compute

Metrics are computed after inverse transforming predictions to the original feature units. The result table includes aggregate MAE/RMSE/MASE, source-standardized normalized MAE/RMSE, macro-entity variants, horizon-wise metrics, trainable/total parameters, train time, inference time, and peak GPU memory when available. MASE uses the mean absolute one-step difference computed on model-train entities only; normalized metrics divide by the frozen source standard deviation per feature. Raw predictions and per-horizon/per-feature metrics are saved so the table can be audited.

Validation-only model selection is written separately from target results. The
cross-regime and cross-domain runners also emit source-to-target shift
summaries and validation-to-target generalization gaps. Phase 2 emits a
support-aware, label-free strategy recommendation while retaining all
configured adaptation variants for later observed-prefix cross-validation.

## Phase-2 adaptation boundary

Phase 2 treats target support as a controlled resource rather than folding target trajectories into the source training pool. The final target horizon remains untouched; only target rows before each final origin can influence a target scaler, a support-window fine-tune, or a residual calibrator. Nested support budgets and a separate heldout-target scope make the adaptation curve inspectable.

PatchTST fine-tuning is implemented first because the source-trained compact model can be restored and continued without changing its input contract. TimesFM and Chronos are evaluated with frozen parameters under source-vs-target scaling and optional feature-wise residual calibration. The latter is reported as post-hoc output calibration, not PEFT. Official TimesFM LoRA uses a separate Transformers checkpoint and context-length contract, so PEFT is deferred to a dedicated subsequent protocol instead of mixing checkpoints in this table.

## Phase-3 PEFT boundary

The dedicated PEFT protocol now uses `google/timesfm-2.5-200m-transformers`, context length 32, raw values, and the official `TimesFm2_5ModelForPrediction` loss path. Source and target-support adapters are separate runs initialized from the base checkpoint; their adapter files are saved and audited. This is not combined with the Phase-1 PyTorch checkpoint or its external scaler.

## Cross-regime and cross-domain boundary

The C-MAPSS cross-regime runner trains on FD001 source entities and evaluates official FD002/FD003/FD004 test entities independently. The source scaler is frozen, target statistics are not estimated, and the strict source/test entity separation is retained. FD002 and FD004 are intentionally treated as operating-regime shifts rather than pooled with FD001; their large raw-unit errors are evidence to analyze, not values to normalize away after seeing the target.

The UCI Gas Turbine runner follows the repository’s first-three-years/last-two-years protocol as source 2011–2012, validation 2013, and target 2014–2015. The local UCI endpoint failed TLS negotiation, so the run uses a public skforecast mirror with an explicit URL, upstream UCI URL, source hash, yearly hashes, and row-count manifest. The mirror’s added timestamps are used only to recover the intended yearly entity boundary from the date-free chronological rows.

## Representation, OOD, and reliability boundary

Post-run analysis fits a two-dimensional PCA only on source-train standardized contexts. It reports target projections, feature/horizon/entity failures, source-standardized OOD scores, and OOD-stratified error. Reliability intervals are source-validation naive absolute-residual intervals applied to saved predictions; they are labeled diagnostic, not model-specific probabilistic forecasts or deployment guarantees. No target labels are used to fit the scaler, PCA, interval quantile, or model.

## Source evidence

* NASA Dashlink dataset description: <https://c3.ndc.nasa.gov/dashlink/resources/139/>
* NASA Data.gov current availability notice and metadata: <https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data>
* TimesFM official repository and API example: <https://github.com/google-research/timesfm>
* TimesFM official API reference: <https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/api_reference.md>
* TimesFM official LoRA fine-tuning example: <https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning>
* Chronos official repository and inference interface: <https://github.com/amazon-science/chronos-forecasting>
* PatchTST official repository: <https://github.com/yuqinie98/PatchTST>
* TimesFM paper: <https://arxiv.org/abs/2310.10688>
* Chronos paper: <https://arxiv.org/abs/2403.07815>
* PatchTST paper: <https://arxiv.org/abs/2211.14730>
* UCI Gas Turbine dataset 551: <https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set>
* UCI Gas Turbine DOI: <https://doi.org/10.24432/C5WC95>
* Public mirror used when the UCI endpoint was unavailable: <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/turbine_emission.csv>
* Tennessee Eastman official challenge archive: <https://depts.washington.edu/control/LARRY/TE/download.html>
* Tennessee Eastman official format: <https://depts.washington.edu/control/LARRY/TE/IDVs/format.txt>
* Uni2TS official code: <https://github.com/SalesforceAIResearch/uni2ts>
* Moirai-2.0-R-small model card: <https://huggingface.co/Salesforce/moirai-2.0-R-small>

## 2026 extension decision

The candidate architecture and industrial-domain review is recorded in
[extension_roadmap.md](extension_roadmap.md). Chronos-2 is the first accepted
extension because its official package is already present in the environment,
its checkpoint is Apache-2.0, and its native multivariate interface provides a
real architecture/input-contract contrast to the existing channel-wise
TimesFM and original Chronos adapters. It is evaluated with `cross_learning`
disabled by default so unrelated entities do not influence one another.

Moirai-2 and Tennessee Eastman are now implemented as the first architecture
and process-domain extension. The fixed protocol and provenance rules are in
[tennessee_eastman_protocol.md](tennessee_eastman_protocol.md); selection and
trade-off artifact semantics are in
[selection_and_tradeoff_protocol.md](selection_and_tradeoff_protocol.md).
