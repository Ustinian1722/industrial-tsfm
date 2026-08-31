# Selection, shift, and trade-off protocol

The platform separates model selection from final target evaluation.

## Model selection

Runners compute the same raw/MASE/normalized metrics on source validation
entities after fitting. `selection.json` ranks complete candidates by the
configured `selection.metric` (default `normalized_rmse`) and can apply explicit
resource constraints such as maximum parameters, inference time, or allowed
training modes. The official target table is not read by the selector.

## Generalization gap

`generalization_gap.csv` joins validation and target metrics by model and seed.
For cross-regime runs with several target subsets, the target value is the
mean across target subsets before the gap is computed. Positive gap means that
target performance is worse than validation under the same source-only scale.
Both absolute and relative gaps are retained.

## Shift quantification

`shift_summary.json` compares source and target raw feature distributions using
source-only reference statistics. It reports absolute mean shift divided by the
source scale, target/source standard-deviation ratios, and log-dispersion
ratios. These are diagnostics, not target-fitted preprocessing parameters.

## Adaptation strategy recommendation

Phase 2 writes `strategy_recommendations.json`. The policy is intentionally
transparent and label-free:

* no support → source-only inference;
* supervised model with support → few-shot fine-tuning;
* zero-shot model with enough observed support → residual calibration;
* otherwise → frozen inference with target-support scaling.

The recommendation is a deployment policy artifact. The runner still reports
all configured support budgets and adaptation variants so the policy can be
challenged in a later observed-prefix cross-validation study.

`src/industrial_tsfm/adaptation_engine.py` upgrades this recommendation to a
runtime decision plan. It reads validation ranking, derives a source-to-target
shift score from the saved artifact, accepts target support and compute/memory
constraints, and returns the selected model, strategy, evidence source, and
budget status. The decision is a plan for the existing adaptation runners; it
does not silently alter or cherry-pick final target results.

## Pareto trade-off

`tradeoff_summary.csv` aggregates complete target rows by model and marks the
lower-is-better Pareto front over normalized RMSE, fit seconds, and total
parameters. It is a cost/performance view, not another model-selection split;
the standalone `scripts/report_tradeoffs.py` command can regenerate it from a
saved result directory.
