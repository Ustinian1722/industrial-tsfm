# IndusTSFM：工业时序基础模型迁移与泛化平台（中文版）

英文版：[README.md](README.md)

[Apache-2.0](LICENSE) · [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md) · [第三方说明](THIRD_PARTY_NOTICES.md) · [开源发布清单](docs/open_source_release_checklist.md)

## 1. 项目现在是什么状态

IndusTSFM 是一个真实可运行、配置驱动、可审计的工业时间序列基础模型（TSFM）迁移与泛化平台。它不使用电池数据，当前主线是：

```text
Zero-shot Transfer
→ Target-data Scaling / Few-shot Adaptation
→ PEFT / LoRA
→ Cross-device
→ Cross-regime
→ Cross-domain
→ Failure / Representation Analysis
→ OOD / Reliability
→ Native Multivariate Chronos-2 / Moirai-2
→ Distribution Shift / Generalization Gap
→ Automatic Adaptation / Model Selection
→ IndusTSFM Demo
```

目前已经不是“把几个模型跑在一个数据集上”的简单 benchmark，而是具备以下实验边界：

- 所有 scaler 和 preprocessing 只在 source model-train 数据上拟合；
- 每个窗口都在单一 entity 内生成，不跨设备/发动机拼接；
- source entity、validation entity、target entity 严格分离；
- 最终 target forecast horizon 不参与适配、校准或模型选择；
- 每个实验保存 config、split、scaler、预测数组、指标、运行时间、参数量、显存和版本信息；
- 提供独立 audit 脚本，从原始数据和 NPZ 预测文件重新计算关键结果。

需要明确的是：仓库同时保留 legacy 单 seed 工程 profile 和经过多 seed 加固的研究 profile；例如 Tennessee Eastman 已完成三 seed、validation-only selection、shift/gap 与自动适配审计。任何结果都不应被外推为某个模型在所有工业场景中最好。

本仓库已经按公开协作方式整理：项目原创代码采用 Apache-2.0；数据集和预训练 checkpoint 不随仓库再分发，用户需要从上游来源下载并遵守各自条款；CI 和测试只使用 synthetic fixtures，不会下载外部数据或模型权重。

项目正式名称为 **IndusTSFM**。为保持现有脚本兼容，Python distribution 名称
仍为 `industrial-tsfm`，import package 仍为 `industrial_tsfm`，CLI 名称不变。

## 2. 研究问题

项目研究的是：

1. 预训练 TSFM 在工业多变量传感器序列上的 zero-shot forecasting 能力如何？
2. 少量目标域数据是否能带来 data-efficiency advantage？
3. 设备变化、运行工况变化和时间域变化会让模型怎样失败？
4. 轻量级 target scaling、few-shot fine-tuning、残差校准和 LoRA 是否能以较低计算成本恢复性能？
5. OOD 分数、上下文表征和可靠性区间能否识别高风险预测？

当前任务是传感器/运行状态多变量 forecasting，不是 RUL 回归。C-MAPSS 的 RUL 文件被保留用于未来 predictive-maintenance 实验，但没有混入当前 forecasting 目标。

## 3. 数据集和任务

### C-MAPSS

- NASA C-MAPSS FD001–FD004；
- 3 个 operating settings + 21 个 sensor，共 24 个输入特征；
- 默认 context length 为 16，forecast horizon 为 8；
- Phase 1/2 使用 FD001；跨工况实验使用 FD001 → FD002/FD003/FD004；
- source train entity 与 validation entity 按固定 seed 划分，官方 test entity 完全隔离；
- 原始文件不进入 Git，下载脚本记录 SHA256。

### UCI Gas Turbine CO/NOx

- 非电池工业域，11 个小时级传感器特征；
- 使用 2011–2012 作为 source train，2013 作为 validation，2014–2015 作为 target test；
- target 采用每 168 个时间点一个 rolling origin，并额外加入每个年份最后一个有效 origin，共 64 个 target windows；
- 这是单个燃气轮机的时间/年份域偏移，不是跨工厂实验；
- UCI 官方元数据显示 36,733 条记录，并规定前三年训练/交叉验证、后两年测试。[UCI 官方页面](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- 当前工作区访问 UCI archive 时发生 TLS 握手失败，因此使用公开 skforecast mirror，并在 `data/raw/gas_turbine/source_manifest.json` 中记录 mirror URL、上游 UCI URL、源文件 hash、年度文件 hash 和行数。[mirror 说明](https://skforecast.org/latest/user_guides/datasets#turbine_emission)

### Tennessee Eastman Process

- 使用 University of Washington 官方 Challenge Archive 的 IDV1–IDV15；
- 默认读取 `y.dat` 的 41 个 measured outputs，保留 10 个 derived outputs 作为可选 feature set；
- 固定 IDV1–8 source、IDV9–10 validation、IDV11–15 target，按 disturbance/entity 划分；
- 默认 context=32、horizon=8、10 分钟采样；
- 官方来源、格式、hash 和许可证边界见 [Tennessee Eastman protocol](docs/tennessee_eastman_protocol.md)。

### NASA/IMS Bearings

- 机械振动扩展采用 NASA/IMS 公共目录分发的原始信号文件；
- loader 将每个 raw file 视为一个 condition cycle，将 bearing 作为 entity；
- 固定输出 `rms`、`std`、`kurtosis`、`crest_factor` 四类每文件特征，并允许配置通道分组；
- 默认 `1st_test` 为 source、`2nd_test` 为 validation、`3rd_test` 为 target；
- 完整压缩包较大，下载脚本需要显式 `--accept-source`，测试不会自动下载；精确契约见 [NASA/IMS Bearings protocol](docs/ims_bearings_protocol.md)。

## 4. 已经实现了什么

### Phase 1：冻结模型 zero-shot 基线

流程：

```text
C-MAPSS FD001
→ train-only standardization
→ entity-safe windows
→ Naive / PatchTST / TimesFM / Chronos / Chronos-2
→ forecast
→ MAE / RMSE / MASE
→ CSV / JSON / NPZ artifacts
```

实现的模型：

- Naive persistence；
- compact PatchTST supervised baseline；
- 官方 TimesFM 2.5 PyTorch checkpoint：`google/timesfm-2.5-200m-pytorch`；
- 官方 Chronos original checkpoint：`amazon/chronos-t5-tiny`，当前按 channel 独立进行 zero-shot forecasting。
- 官方 Chronos-2 checkpoint：`amazon/chronos-2`，按窗口联合建模全部特征，使用官方 p50 quantile 作为点预测；默认关闭跨窗口 `cross_learning`。
- 可选官方 Moirai-2 checkpoint：`Salesforce/moirai-2.0-R-small`，通过 Uni2TS wide multivariate predictor 接入，使用 p50 quantile；依赖边界和版本写入 manifest。

### Phase 2：target-data scaling / few-shot adaptation

目标 support 比例为 0%、1%、5%、10%、25%、50%、100%。实验严格限制 support 只能使用最终 forecast origin 之前的 observed prefix，最终 target horizon 不可见。

实现了：

- target support scaler；
- PatchTST few-shot fine-tuning；
- TimesFM/Chronos frozen target scaling；
- feature-wise residual bias calibration；
- all-test 和 heldout-target 两种评估 scope；
- nested support budget 和 support manifest。
- Moirai-2 zero-shot target scaling、residual calibration，以及不使用最终 target labels 的 `strategy_recommendations.json`。

### Phase 3：TimesFM Transformers + LoRA

使用官方 Transformers 接口和独立 checkpoint：

- checkpoint：`google/timesfm-2.5-200m-transformers`；
- `TimesFm2_5ModelForPrediction`；
- LoRA `target_modules="all-linear"`，rank 4，alpha 8，dropout 0.05；
- context length 32；
- 总参数 232,672,192；可训练参数 1,382,912，约 0.59%；
- source adapter 和 target adapter 分开保存并审计。

### Cross-regime / cross-device

source 使用 FD001 model-train entities，target 使用 FD002、FD003、FD004 官方 test entities。只使用 FD001 source scaler，不用 target 数据重新估计统计量，也不使用 RUL。

### Cross-domain

新增了 UCI Gas Turbine 数据加载器、mirror fetcher、年份 entity 划分、source-only scaling、rolling target evaluation 和完整 audit。

所有新版 runner 还会保存 source-standardized normalized MAE/RMSE、
validation-only `selection.json`、source-to-target `shift_summary.json` 和
validation-to-target `generalization_gap.csv`，语义见
[selection_and_tradeoff_protocol.md](docs/selection_and_tradeoff_protocol.md)。

`src/industrial_tsfm/adaptation_engine.py` 提供轻量级自动适配层：它只读取
validation 证据、target support 数量、shift/OOD 分数和 compute budget，自动
选择模型及 zero-shot、target scaling、residual calibration、few-shot fine-tuning
或 PEFT 策略，并在 `engine_decision.json` 中明确记录
`target_labels_used: false`。

可以直接从已审计结果生成静态 Demo：

```powershell
python scripts/run_adaptation_engine.py results/<run_id> --target-fraction 0.10 --support-windows 20
python scripts/build_industrial_demo.py results/<run_id>
python scripts/audit_demo.py results/<run_id>/demo/index.html
```

### Failure / Representation / OOD / Reliability

`scripts/analyze_run.py` 会生成：

- feature-level error；
- horizon-level error；
- entity-level error；
- source-only PCA context representation；
- source-standardized OOD score；
- OOD quartile 与误差关系；
- source-validation naive residual interval；
- aggregate 和 feature-level reliability coverage。

这里的 PCA 是可解释的 standardized-context representation，不是假装成 foundation model hidden state。可靠性区间是 source-validation naive residual 的诊断区间，不是经过 deployment calibration 的概率预测。

## 5. 真实实验结果

### 5.1 Phase 1：FD001 zero-shot baseline

100 个官方 test entity、每个 entity 一个最终有效窗口，context=16、horizon=8。

| 模型 | 模式 | MAE | RMSE | MASE | 总参数 | 可训练参数 |
|---|---|---:|---:|---:|---:|---:|
| Naive | persistence | 0.8639 | 2.3554 | 0.7267 | 0 | 0 |
| PatchTST | source supervised | **0.6722** | **1.8214** | **0.5566** | 69,384 | 69,384 |
| TimesFM | frozen zero-shot | 0.7006 | 1.9361 | 0.5620 | 231,289,280 | 0 |
| Chronos | frozen zero-shot | 0.7703 | 2.2185 | 0.6101 | 8,394,496 | 0 |

在这个单 seed FD001 profile 中，PatchTST 的三个聚合指标最好，TimesFM 很接近，Chronos 较弱。这个结果说明预训练模型具有可用的工业迁移能力，但没有证明 TSFM 一定超过专门训练的小模型。

结果目录：`results/20260817T154441Z_5bb8cb75f0/`

### 5.2 Phase 2：目标数据 scaling / few-shot

主要观察：

- PatchTST 在 all-test 上从 0% support 的 MAE 0.6722 出发，小比例 support 没有稳定改善，较高比例的简单 fine-tuning 反而可能变差；
- TimesFM 的 target scaling 在 all-test 上几乎不改变 zero-shot 结果；
- heldout-target 结果在高 support 时可能改善，但 100% support 的 heldout scope 只剩极少量实体，不能作为稳定结论；
- Chronos 的 adaptation 曲线不稳定；
- 因此当前结果更像是“适配效果高度依赖模型和评估 scope”，而不是“目标数据越多必然越好”。

结果目录：`results/20260817T155914Z_9b226eb061/`

### 5.3 Phase 3：TimesFM LoRA

最终有效 test entity 为 96 个，4 个过短 entity 被跳过；support candidate 为 95 个。以下是 all-valid-test profile：

| 适配方式 | support | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| Base zero-shot | 0% | 0.70163 | 1.93711 | 0.55966 |
| Source LoRA | source train | **0.69314** | **1.90100** | 0.55704 |
| Target LoRA | 1% | 0.70111 | 1.93500 | 0.55958 |
| Target LoRA | 5% | 0.70027 | 1.93132 | **0.55925** |
| Target LoRA | 10% | 0.70129 | 1.93601 | 0.55945 |

当前 bounded profile 中 source LoRA 有小幅改善，target LoRA 基本接近 base。该实验只有一个 seed、一个 epoch 和有限的 series budget，属于 PEFT 工程链路验证，不是最终适配结论。

结果目录：`results/20260817T162540Z_906dfc042b/`

### 5.4 C-MAPSS 跨工况/跨设备

source=FD001，target 为其他 regime 的官方 test entity；context=16、horizon=8；所有 target 都使用 FD001 source scaler。

| Target | 有效实体 | 跳过实体 | Naive MAE | PatchTST MAE | TimesFM MAE | Chronos MAE |
|---|---:|---:|---:|---:|---:|---:|
| FD002 | 256 | 3 | 64.6193 | 82.1675 | **52.4953** | 57.9771 |
| FD003 | 100 | 0 | 0.8535 | 0.7560 | **0.7341** | 0.7902 |
| FD004 | 242 | 6 | 62.8121 | 83.7359 | **51.5366** | 55.7999 |

解释：FD002/FD004 在 source-only scale 下出现非常大的 regime/domain shift，因此 raw-unit MAE 远高于 FD003。预训练 TSFM 在这两个 shift 较大的 subset 上比 source-trained PatchTST 和 Naive 更稳，但不存在对所有 subset、指标都最优的模型。详细 RMSE/MASE 在 [cross_regime_protocol.md](docs/cross_regime_protocol.md)。

结果目录：`results/20260817T163507Z_9a55c0ca99/`

### 5.5 UCI Gas Turbine 跨域/跨年份

target 为 2014–2015，共 64 个 rolling windows、2 个年份 entity。

| 模型 | MAE | RMSE | MASE |
|---|---:|---:|---:|
| Naive | 5.1793 | 10.4331 | 2.9372 |
| PatchTST | **3.7881** | **7.1021** | 2.3375 |
| TimesFM | 4.0434 | 8.4874 | **2.3144** |
| Chronos | 4.3671 | 9.2032 | 2.4346 |

PatchTST 在 raw-unit MAE/RMSE 上最好，TimesFM 在 source-train MASE 上略好。这个实验是单工厂的跨年份 shift，不是跨工厂泛化。

结果目录：`results/20260817T170355Z_a1f73fd9bc/`

### 5.6 Chronos-2 native multivariate smoke

在与 Phase-1 相同的 FD001、context=16、horizon=8、24 个特征和 seed=42 配置下，Chronos-2 的第一次完整 smoke 已真实生成并保存 artifact：

| 模型 | 输入契约 | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| Chronos-2 | native multivariate, p50 | 0.7044 | 1.9656 | 0.5615 |

该结果不能直接替换历史 channel-wise Chronos，也不是多 seed 结论；它的价值是验证原生多变量模型已接入同一配置、指标、预测数组和资源记录链路。结果目录：`results/20260817T175633Z_b5764d39ee/`。

### 5.7 Tennessee Eastman 三 seed profile

在 IDV1–8 → IDV9–10 → IDV11–15、source-only scaling、context=32、
horizon=8 的协议下，Naive、PatchTST、Moirai-2 的 target normalized RMSE
均值如下：

| 模型 | target normalized RMSE | seed std | validation-only selection |
|---|---:|---:|---|
| Naive | 2.0144 | 0 |  |
| PatchTST | 1.6022 | 0.0072 |  |
| Moirai-2 | 1.6399 | 0 | selected on validation in this run |

该 profile 的 target 结果支持 PatchTST，但 validation selector 选择了
Moirai-2；两者差异通过 generalization gap 保留，不能用 target 最优反推
selector。结果目录：`results/20260817T193124Z_1263d3db69/`。

### 5.8 OOD 和 reliability 观察

- C-MAPSS FD002/FD004 的 source-standardized context OOD RMS 中位数约为 3,197/3,244，FD003 约为 1.14；source-calibrated interval 在 FD002/FD004 上严重 under-coverage；
- UCI Gas Turbine 的 target OOD RMS 中位数约为 0.97，范围约 0.71–2.09；90% source-validation naive interval 的实际 coverage 为 Naive 83.9%、PatchTST 90.8%、TimesFM 89.5%、Chronos 88.2%；
- UCI 结果中，最高 OOD quartile 的误差对所有模型都有上升趋势，但 Spearman 相关约为 0.23–0.30，只能作为诊断信号，不能直接当作可靠风险概率。

## 6. 重要的研究结论

当前最可信的结论不是“TimesFM/Chronos 一定最好”，而是：

1. 小型 PatchTST 在同域 FD001 上很强，不能被忽略；
2. TimesFM 在跨 regime 的大 shift subset 上表现出较好的 raw-unit robustness；
3. zero-shot 优势不是在所有域、所有指标上自动成立；
4. 少量目标数据适配并不保证单调收益，必须区分 all-test 和 heldout-target；
5. source-only scaling 下的 OOD score 能显著暴露 FD002/FD004 的分布偏移；
6. 当前 reliability interval 更适合作为 shift diagnostic，而不是 production uncertainty guarantee。

## 7. 如何运行

### 安装

```powershell
python -m pip install -e ".[dev]"
```

只有需要对应模型时才安装可选依赖：

```powershell
python -m pip install -e ".[tsfm]"    # TimesFM 和 Chronos
python -m pip install -e ".[moirai]"  # Moirai-2 / Uni2TS
python -m pip install -e ".[peft]"    # TimesFM Transformers + LoRA
```

如果 C 盘空间紧张，先把 Hugging Face cache 放到 D 盘：

```powershell
$env:HF_HOME = "D:\IndustrialTSFM\hf_cache"
$env:HF_HUB_CACHE = "D:\IndustrialTSFM\hf_cache\hub"
$env:HF_HUB_DISABLE_XET = "1"
```

### C-MAPSS

```powershell
python scripts/fetch_cmapss_fd001.py --accept-mirror --subsets FD001,FD002,FD003,FD004
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast.yaml
python -m industrial_tsfm.adapt_cli --config configs/cmapss_fd001_adaptation.yaml
python -m industrial_tsfm.peft_cli --config configs/cmapss_fd001_timesfm_peft.yaml
python -m industrial_tsfm.cross_regime_cli --config configs/cmapss_cross_regime.yaml
```

### Tennessee Eastman

```powershell
python scripts/fetch_tennessee_eastman.py --accept-source
python -m industrial_tsfm.cross_domain_cli --config configs/tennessee_eastman_cross_domain_multiseed.yaml --models naive,patchtst,moirai2
```

如需生成三 seed 的 Phase-1 均值、样本标准差和 95% CI，可运行：

```powershell
python -m industrial_tsfm.cli --config configs/cmapss_fd001_forecast_multiseed.yaml
```

### UCI Gas Turbine

```powershell
python scripts/fetch_gas_turbine.py --accept-mirror
python -m industrial_tsfm.cross_domain_cli --config configs/gas_turbine_cross_domain.yaml
```

### NASA/IMS Bearings

```powershell
python scripts/fetch_ims_bearings.py --accept-source --metadata-only
python scripts/fetch_ims_bearings.py --accept-source
python -m industrial_tsfm.cross_domain_cli --config configs/ims_bearings_cross_domain.yaml --models naive,patchtst,moirai2
python scripts/audit_cross_domain_run.py results/<ims_run_id>
```

### 审计与分析

```powershell
python scripts/audit_phase2_run.py results/<phase2_run_id>
python scripts/audit_peft_run.py results/<peft_run_id>
python scripts/audit_cross_regime_run.py results/<cross_regime_run_id>
python scripts/audit_cross_domain_run.py results/<cross_domain_run_id>
python scripts/analyze_run.py results/<cross_regime_or_domain_run_id>
python scripts/audit_analysis.py results/<cross_regime_or_domain_run_id>
```

## 8. 目录结构

```text
src/industrial_tsfm/data/          数据解析、entity split、窗口、scaler
src/industrial_tsfm/models/        Naive、PatchTST、TimesFM、Chronos、Chronos-2、PEFT adapter
src/industrial_tsfm/adaptation.py  target support adaptation
src/industrial_tsfm/peft_experiment.py  TimesFM Transformers + LoRA
src/industrial_tsfm/cross_regime.py     C-MAPSS 跨工况 runner
src/industrial_tsfm/cross_domain.py     UCI/TE/IMS cross-domain runner
src/industrial_tsfm/selection.py        validation-only 选模、gap、策略推荐
src/industrial_tsfm/adaptation_engine.py 自动模型/策略/预算决策 API
src/industrial_tsfm/evaluation/          metrics 和 detail artifacts
scripts/                             fetch、run audit、分析脚本
configs/                             所有实验配置
docs/                                固定 protocol 和 research decisions
results/                             本地实验输出，默认不进入 Git
tests/                               synthetic unit/smoke tests
```

## 9. 当前局限和下一步

仍然保留的研究边界：

- legacy profile 中仍有单 seed 结果；TE reference profile 已具备三 seed mean/std，更多 domain/seed 仍可继续扩展；
- UCI Gas Turbine 目前是单工厂跨年份，不是跨工厂；
- Paderborn bearing 尚未接入；Tennessee Eastman 已完成第一版数据源与协议审计，IMS 已完成 loader/fetcher 的 streaming feature contract 和 synthetic audit，但实际原始 benchmark 仍需显式下载后执行；
- MOMENT、Time-MoE 尚未接入主环境，原因是依赖版本/资源隔离需要单独 adapter；这些边界记录在 [docs/extension_roadmap.md](docs/extension_roadmap.md)；
- 当前 representation 是 source-only PCA，不是 TimesFM/Chronos hidden representation；
- reliability interval 是诊断型 source-validation interval，尚未形成 production-grade calibrated uncertainty；
- Chronos 当前保留原始 univariate channel-wise 接口，Chronos-2 已作为独立 native multivariate 模型接入，不能覆盖历史 Chronos 结果；
- accuracy–data–compute trade-off、support budget、PEFT 资源记录和静态 Demo 已接入；更大规模 budget sweep、更多 PEFT epoch 和 calibrated deployment uncertainty 仍可继续扩展。

因此，项目核心已经处于“研究闭环打通、主要 profile 可运行可审计、便于复现与审阅”的状态；后续工作主要是扩大实验覆盖面，而不是补齐基础链路。

## 10. 关键文档和结果

- [Phase-1 protocol](docs/experiment_protocol.md)
- [Phase-2 adaptation protocol](docs/phase2_adaptation_protocol.md)
- [TimesFM PEFT protocol](docs/peft_protocol.md)
- [Cross-regime protocol](docs/cross_regime_protocol.md)
- [Cross-domain protocol](docs/cross_domain_protocol.md)
- [Failure/OOD/reliability protocol](docs/analysis_reliability_protocol.md)
- [Tennessee Eastman protocol](docs/tennessee_eastman_protocol.md)
- [NASA/IMS Bearings protocol](docs/ims_bearings_protocol.md)
- [Selection and trade-off protocol](docs/selection_and_tradeoff_protocol.md)
- [Research decisions](docs/research_decisions.md)
- [Extension roadmap](docs/extension_roadmap.md)

最终验证状态：`22 passed`、compileall、Ruff、Moirai-2 FD001 adaptation、C-MAPSS cross-regime smoke、Tennessee Eastman 三 seed cross-domain、Phase-2/cross-regime/cross-domain/analysis audits、自动适配决策审计、Demo 审计，以及 IMS synthetic loader/runner tests 均通过；真实 IMS 原始 profile 仍为 opt-in。实验结果目录默认不进入公开仓库，需要时按 README 命令重新生成并审计。
