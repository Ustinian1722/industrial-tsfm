from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_SECTIONS = (
    "Data Check",
    "Shift Assessment",
    "TSFM Selection",
    "Adaptation Plan",
    "Forecast",
    "OOD / Reliability",
    "Accuracy–Data–Compute Trade-off",
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


def build_demo(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "manifest.json", {})
    dataset_summary = _read_json(run_dir / "dataset_summary.json", {})
    if not dataset_summary:
        dataset_summary = _read_json(run_dir / "source_dataset_summary.json", {})
    shift = _read_json(run_dir / "shift_summary.json", {})
    selection = _read_json(run_dir / "selection.json", {})
    engine = _read_json(run_dir / "engine_decision.json", {})
    result_records = _read_records(run_dir / "result_table.csv")
    tradeoff_records = _read_records(run_dir / "tradeoff_summary.csv")
    reliability_records = _read_records(run_dir / "analysis" / "reliability_summary.csv")
    ood_records = _read_records(run_dir / "analysis" / "ood_window_summary.csv")
    if not result_records:
        raise ValueError(f"No result_table.csv records found in {run_dir}")
    if output_path is None:
        output_path = run_dir / "demo" / "index.html"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "dataset_summary": dataset_summary,
        "shift": shift,
        "selection": selection,
        "engine": engine,
        "results": result_records,
        "tradeoff": tradeoff_records,
        "reliability": reliability_records,
        "ood": ood_records,
    }
    title = str(manifest.get("source_dataset", manifest.get("run_id", "IndusTSFM")))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IndusTSFM Demo — {escape(title)}</title>
<style>
:root {{ --ink:#172033; --muted:#667085; --line:#dbe2ea; --blue:#1769aa; --teal:#0f766e; --amber:#b45309; --bg:#f5f7fb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--bg); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ padding:34px max(24px, calc((100% - 1180px)/2)); background:linear-gradient(120deg,#102a43,#1769aa); color:white; }}
header h1 {{ margin:0 0 6px; font-size:30px; }} header p {{ margin:0; opacity:.86; }}
main {{ max-width:1180px; margin:24px auto 60px; padding:0 20px; }} section {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px; margin:16px 0; box-shadow:0 3px 12px #102a4308; }}
h2 {{ margin:0 0 14px; font-size:20px; }} h3 {{ margin:14px 0 8px; font-size:15px; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }} .card {{ border:1px solid var(--line); border-radius:10px; padding:12px; }} .label {{ color:var(--muted); font-size:12px; }} .value {{ font-size:19px; font-weight:650; margin-top:3px; }}
.pill {{ display:inline-block; padding:3px 9px; border-radius:999px; background:#e7f1fb; color:var(--blue); font-weight:600; }} .pill.good {{ background:#e8f7f2; color:var(--teal); }} .pill.warn {{ background:#fff4e5; color:var(--amber); }}
.table-wrap {{ overflow:auto; }} table {{ border-collapse:collapse; width:100%; min-width:620px; }} th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.bar {{ height:8px; border-radius:5px; background:#e8eef5; overflow:hidden; min-width:110px; }} .bar i {{ display:block; height:100%; background:linear-gradient(90deg,#1769aa,#0f766e); }} .note {{ color:var(--muted); font-size:13px; }} code {{ background:#eef2f6; padding:2px 5px; border-radius:4px; }} footer {{ max-width:1180px; margin:0 auto 40px; padding:0 20px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<header><h1>IndusTSFM: Adaptation &amp; Generalization</h1><p>Data check → shift assessment → TSFM selection → adaptation → forecast → reliability</p></header>
<main>
<section><h2>Data Check</h2><div id="data-cards" class="grid"></div><p id="data-note" class="note"></p></section>
<section><h2>Shift Assessment</h2><div id="shift-cards" class="grid"></div><div class="table-wrap"><table id="shift-table"></table></div></section>
<section><h2>TSFM Selection</h2><div id="selection-cards" class="grid"></div><div class="table-wrap"><table id="selection-table"></table></div></section>
<section><h2>Adaptation Plan</h2><div id="engine-cards" class="grid"></div><p id="engine-note" class="note"></p></section>
<section><h2>Forecast</h2><div class="table-wrap"><table id="result-table"></table></div></section>
<section><h2>OOD / Reliability</h2><div class="grid" id="reliability-cards"></div><div class="table-wrap"><table id="reliability-table"></table></div></section>
<section><h2>Accuracy–Data–Compute Trade-off</h2><div class="table-wrap"><table id="tradeoff-table"></table></div><p class="note">Pareto front uses lower-is-better normalized RMSE, fit time, and parameter count. It is a decision aid, not test-set model selection.</p></section>
</main>
<footer>Generated from a validated run directory. Target labels are not used by the engine decision.</footer>
<script id="demo-data" type="application/json">{_json_script(payload)}</script>
<script>
const D = JSON.parse(document.getElementById('demo-data').textContent);
const fmt = (x) => x === null || x === undefined || Number.isNaN(Number(x)) ? '—' : (typeof x === 'number' ? x.toFixed(4) : String(x));
const card = (label, value, cls='') => `<div class="card"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`;
const table = (id, rows, columns) => {{ const el=document.getElementById(id); if(!rows.length) {{el.innerHTML='<tr><td class="note">No artifact available</td></tr>'; return;}} el.innerHTML='<thead><tr>'+columns.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+columns.map(c=>`<td>${{fmt(r[c])}}</td>`).join('')+'</tr>').join('')+'</tbody>'; }};
const aggregate = D.shift.aggregate || (Object.values(D.shift).find(x=>x && x.aggregate)||{{}}).aggregate || {{}};
document.getElementById('data-cards').innerHTML = [card('Source', D.manifest.source_dataset || D.manifest.source_subset || '—'), card('Target', D.manifest.target_dataset || (D.manifest.target_subsets||[]).join(', ') || '—'), card('Device', D.manifest.device || '—'), card('Protocol', D.manifest.protocol || '—')].join('');
document.getElementById('data-note').textContent = `Run: ${{D.manifest.run_id || '—'}} · Dataset summary rows/entities: ${{D.dataset_summary.train_rows || D.dataset_summary.source_rows || '—'}} / ${{D.dataset_summary.train_entities || D.dataset_summary.entities || '—'}}`;
document.getElementById('shift-cards').innerHTML = [card('Mean standardized shift', fmt(aggregate.mean_abs_standardized_mean_shift)), card('RMS standardized shift', fmt(aggregate.rms_standardized_mean_shift)), card('Max mean shift', fmt(aggregate.max_abs_standardized_mean_shift)), card('Mean |log std ratio|', fmt(aggregate.mean_abs_log_std_ratio))].join('');
const shiftRows = D.shift.features || []; table('shift-table', shiftRows.slice().sort((a,b)=>(b.standardized_mean_shift||0)-(a.standardized_mean_shift||0)).slice(0,10), ['feature','standardized_mean_shift','std_ratio','abs_log_std_ratio']);
const sel = D.selection || {{}}; document.getElementById('selection-cards').innerHTML = [card('Selected model', sel.selected_model || '—','good'), card('Metric', sel.selection_metric || '—'), card('Evidence', sel.selection_split || '—'), card('Target labels used', sel.target_labels_used === false ? 'No' : 'Not applicable')].join(''); table('selection-table', sel.ranking || [], ['rank','model','normalized_rmse_mean','normalized_rmse_std','fit_seconds_mean','total_parameters_mean','training_modes']);
const E = D.engine || {{}}; document.getElementById('engine-cards').innerHTML = [card('Model', E.selected_model || sel.selected_model || '—','good'), card('Strategy', E.selected_strategy || '—','good'), card('Target fraction', E.request ? `${{fmt(E.request.target_data_fraction*100)}}%` : '—'), card('Budget', E.adaptation_budget_status || '—')].join(''); document.getElementById('engine-note').textContent = `${{E.strategy_reason || 'No engine decision artifact.'}} · selection evidence: ${{E.selection_evidence || '—'}} · target labels used: ${{E.target_labels_used === false ? 'false' : 'unknown'}}`;
const resultRows = D.results.filter(r=>String(r.status)==='complete'); table('result-table', resultRows, ['model','seed','mae','rmse','mase','normalized_mae','normalized_rmse','fit_seconds','inference_seconds']);
const reli = D.reliability; if(reli.length) {{ const cov = reli.reduce((s,r)=>s+Number(r.coverage||0),0)/reli.length; document.getElementById('reliability-cards').innerHTML=card('Mean coverage',fmt(cov),'good')+card('Nominal coverage',fmt(reli[0].nominal_coverage))+card('Rows',String(reli.length)); }} else {{ document.getElementById('reliability-cards').innerHTML=card('Reliability','Run analysis to populate'); }} table('reliability-table', reli, ['model','target','coverage','nominal_coverage','mean_interval_width','ood_mae_spearman']);
table('tradeoff-table', D.tradeoff, ['model','normalized_rmse','fit_seconds','total_parameters','pareto_optimal','n_seeds']);
</script>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static IndusTSFM demo report")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(build_demo(args.run_dir, args.output))


if __name__ == "__main__":
    main()
