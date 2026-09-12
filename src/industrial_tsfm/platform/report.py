from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .contracts import ProjectSpec


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


def build_platform_report(
    project: ProjectSpec,
    audit: dict[str, Any],
    output_path: str | Path,
    *,
    route_decision: dict[str, Any] | None = None,
    replay_snapshot: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Render a dependency-free product-style industrial intelligence page."""

    project.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    route = route_decision or {}
    replay = replay_snapshot or {}
    source_provenance = provenance or {}
    summary = audit.get("summary", {})
    variables = audit.get("variables", [])
    risks = audit.get("risks", [])
    usable = audit.get("usable_numeric_features", [])
    time_info = audit.get("time", {})

    cards = [
        ("Rows", summary.get("rows")),
        ("Variables", summary.get("columns")),
        ("Readiness", f"{_fmt(summary.get('readiness_score'))}/100"),
        ("Route", route.get("status", "not run")),
        ("Model", route.get("selected_model")),
        ("Strategy", route.get("selected_strategy")),
    ]
    card_html = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(_fmt(value))}</strong></div>'
        for label, value in cards
    )
    variable_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('column', '')))}</td>"
        f"<td>{escape(str(row.get('dtype', '')))}</td>"
        f"<td>{float(row.get('missing_ratio', 0.0))*100:.1f}%</td>"
        f"<td>{escape('yes' if row.get('is_constant') else 'no')}</td>"
        f"<td>{escape('yes' if row.get('column') in usable else 'no')}</td>"
        "</tr>"
        for row in variables
    )
    risk_html = "".join(f"<li>{escape(str(item))}</li>" for item in risks) or "<li>No blocking risk detected.</li>"
    task_html = "".join(
        f"<li><b>{escape(task.name)}</b> · {escape(task.task_type.value)} · targets: "
        f"{escape(', '.join(task.target_columns) or 'n/a')} · KPI: {escape(task.business_kpi or 'n/a')}</li>"
        for task in project.tasks
    )
    ranking_rows = "".join(
        "<tr>"
        f"<td>{escape(_fmt(row.get('rank')))}</td>"
        f"<td>{escape(_fmt(row.get('model')))}</td>"
        f"<td>{escape(_fmt(row.get('normalized_rmse_mean')))}</td>"
        f"<td>{escape(_fmt(row.get('inference_seconds_mean')))}</td>"
        f"<td>{escape(_fmt(row.get('training_modes')))}</td>"
        "</tr>"
        for row in route.get("model_ranking", [])
    )
    route_reasons = route.get("blocking_reasons", []) + route.get("warnings", [])
    route_reason_html = "".join(
        f"<li>{escape(str(item))}</li>" for item in route_reasons
    ) or "<li>No routing blocker.</li>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IndusTSFM Platform — {escape(project.name)}</title>
<style>
:root {{ --bg:#f4f7fb; --panel:#fff; --ink:#172033; --muted:#667085; --line:#d9e2ec; --accent:#155eef; --good:#067647; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ background:linear-gradient(120deg,#0b1f33,#123f66); color:white; padding:30px max(22px, calc((100% - 1180px)/2)); }}
header h1 {{ margin:0; font-size:28px; }} header p {{ margin:6px 0 0; opacity:.84; }}
main {{ max-width:1180px; margin:22px auto 60px; padding:0 18px; }} nav {{ color:var(--muted); margin-bottom:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }} .card,section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.card {{ padding:14px; }} .card span {{ display:block; color:var(--muted); font-size:12px; }} .card strong {{ display:block; font-size:20px; margin-top:3px; overflow-wrap:anywhere; }}
section {{ margin-top:14px; padding:18px; }} h2 {{ margin:0 0 12px; font-size:18px; }} h3 {{ margin:16px 0 8px; font-size:14px; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-size:12px; }}
.badge {{ display:inline-block; background:#e8f0ff; color:var(--accent); border-radius:999px; padding:3px 8px; margin:2px 4px 2px 0; }} .ok {{ color:var(--good); font-weight:650; }} .note {{ color:var(--muted); }} .wrap {{ overflow:auto; }} code {{ background:#f0f3f7; padding:2px 5px; border-radius:5px; }}
</style>
</head>
<body>
<header><h1>IndusTSFM Industrial Intelligence Platform</h1><p>{escape(project.name)} · Data → audit → model routing → replay/runtime → application</p></header>
<main>
<nav>Overview / Data / Analysis / Models / Applications / Deployment</nav>
<div class="grid">{card_html}</div>
<section><h2>Project &amp; tasks</h2><p>{escape(project.description or 'No description')}</p><ul>{task_html}</ul></section>
<section><h2>Data source &amp; health</h2><p class="note">Source: {escape(_fmt(source_provenance.get('source_name')))} · kind: {escape(_fmt(source_provenance.get('kind')))} · SHA-256: <code>{escape(_fmt(source_provenance.get('sha256')))}</code></p><p class="note">Timestamp: {escape(_fmt(time_info.get('timestamp_column')))} · valid: {escape(_fmt(time_info.get('valid')))} · median step: {escape(_fmt(time_info.get('median_step_seconds')))} s</p><ul>{risk_html}</ul></section>
<section><h2>Variables</h2><div class="wrap"><table><thead><tr><th>Variable</th><th>Dtype</th><th>Missing</th><th>Constant</th><th>Usable</th></tr></thead><tbody>{variable_rows}</tbody></table></div></section>
<section><h2>Recommended model inputs</h2>{''.join(f'<span class="badge">{escape(name)}</span>' for name in usable) or '<span class="note">No usable numeric features.</span>'}</section>
<section><h2>AutoModel / Model Router</h2><p>Status: <span class="ok">{escape(_fmt(route.get('status', 'not run')))}</span> · evidence: {escape(_fmt(route.get('selection_evidence')))} · target labels used: {escape(_fmt(route.get('target_labels_used')))}</p><p><b>{escape(_fmt(route.get('selected_model')))}</b> → {escape(_fmt(route.get('selected_strategy')))}. {escape(_fmt(route.get('strategy_reason')))}</p><ul>{route_reason_html}</ul><div class="wrap"><table><thead><tr><th>Rank</th><th>Model</th><th>Validation NRMSE</th><th>Inference s</th><th>Mode</th></tr></thead><tbody>{ranking_rows or '<tr><td colspan="5" class="note">No model ranking available.</td></tr>'}</tbody></table></div></section>
<section><h2>Application replay runtime</h2><p>{escape(_fmt(replay.get('rows')))} rows · {escape(_fmt(replay.get('total_batches')))} batches · batch size {escape(_fmt(replay.get('batch_size')))} · preserves input order: {escape(_fmt(replay.get('preserves_input_order')))}</p><p class="note">Replay is deterministic and has no wall-clock sleep. It is the offline application/runtime contract before MQTT/OPC-UA streaming is enabled.</p></section>
<section><h2>Next product slice</h2><p>Bind the routed numerical pipeline to an application service, add anomaly-task adapters and model registry metadata, then expose replay/live inference through an API. LLM Agent orchestration remains above this deterministic layer.</p></section>
</main>
<script type="application/json" id="platform-json">{_json_script({'audit': audit, 'route': route, 'replay': replay, 'provenance': source_provenance})}</script>
</body></html>"""
    output.write_text(html, encoding="utf-8")
    return output
