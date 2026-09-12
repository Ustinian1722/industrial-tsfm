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


def build_platform_report(
    project: ProjectSpec,
    audit: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render a dependency-free product-style data audit page."""

    project.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = audit.get("summary", {})
    variables = audit.get("variables", [])
    risks = audit.get("risks", [])
    usable = audit.get("usable_numeric_features", [])
    time_info = audit.get("time", {})

    cards = [
        ("Rows", summary.get("rows")),
        ("Variables", summary.get("columns")),
        ("Numeric", summary.get("numeric_columns")),
        ("Readiness", f"{_fmt(summary.get('readiness_score'))}/100"),
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
        f"{escape(', '.join(task.target_columns) or 'n/a')}</li>"
        for task in project.tasks
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IndusTSFM Platform — {escape(project.name)}</title>
<style>
:root {{ --bg:#f4f7fb; --panel:#fff; --ink:#172033; --muted:#667085; --line:#d9e2ec; --accent:#155eef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ background:#102a43; color:white; padding:28px max(22px, calc((100% - 1180px)/2)); }}
header h1 {{ margin:0; font-size:28px; }} header p {{ margin:6px 0 0; opacity:.82; }}
main {{ max-width:1180px; margin:22px auto 60px; padding:0 18px; }}
nav {{ color:var(--muted); margin-bottom:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.card, section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.card {{ padding:14px; }} .card span {{ display:block; color:var(--muted); font-size:12px; }} .card strong {{ display:block; font-size:22px; margin-top:3px; }}
section {{ margin-top:14px; padding:18px; }} h2 {{ margin:0 0 12px; font-size:18px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-size:12px; }}
.badge {{ display:inline-block; background:#e8f0ff; color:var(--accent); border-radius:999px; padding:3px 8px; margin:2px 4px 2px 0; }}
.note {{ color:var(--muted); }} .wrap {{ overflow:auto; }}
</style>
</head>
<body>
<header><h1>IndusTSFM Industrial Intelligence Platform</h1><p>{escape(project.name)} · Data audit → task definition → model routing → deployment</p></header>
<main>
<nav>Overview / Data / Analysis / Models / Applications / Deployment</nav>
<div class="grid">{card_html}</div>
<section><h2>Project</h2><p>{escape(project.description or 'No description')}</p><ul>{task_html}</ul></section>
<section><h2>Data health</h2><p class="note">Timestamp: {escape(_fmt(time_info.get('timestamp_column')))} · valid: {escape(_fmt(time_info.get('valid')))} · median step: {escape(_fmt(time_info.get('median_step_seconds')))} s</p><ul>{risk_html}</ul></section>
<section><h2>Variables</h2><div class="wrap"><table><thead><tr><th>Variable</th><th>Dtype</th><th>Missing</th><th>Constant</th><th>Usable</th></tr></thead><tbody>{variable_rows}</tbody></table></div></section>
<section><h2>Recommended model inputs</h2>{''.join(f'<span class="badge">{escape(name)}</span>' for name in usable) or '<span class="note">No usable numeric features.</span>'}</section>
<section><h2>Next product step</h2><p>Bind this audited data source to a task, run validation-only model routing, then expose the selected forecast/anomaly pipeline as an application endpoint. No target-test label is used at this stage.</p></section>
</main>
<script type="application/json" id="audit-json">{escape(json.dumps(audit, ensure_ascii=False))}</script>
</body></html>"""
    output.write_text(html, encoding="utf-8")
    return output
