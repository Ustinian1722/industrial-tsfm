from __future__ import annotations


def render_dashboard() -> str:
    """Return the dependency-free V1 operator/developer dashboard."""

    return r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IndusTSFM Studio</title>
<style>
:root{--bg:#f3f6fa;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d8e0e9;--nav:#0b1726;--nav2:#132c46;--accent:#1e6eea;--good:#087a55;--warn:#b75f00}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.shell{display:grid;grid-template-columns:236px 1fr;min-height:100vh}.side{background:linear-gradient(180deg,var(--nav),var(--nav2));color:#dbe7f3;padding:22px 16px;position:sticky;top:0;height:100vh}.brand{font-weight:800;font-size:20px;color:#fff;padding:4px 8px 20px}.brand small{display:block;font-size:10px;letter-spacing:.12em;color:#8fb3d8;margin-top:4px}.nav a{display:block;color:#bcd0e3;text-decoration:none;padding:10px 12px;border-radius:8px;margin:3px 0}.nav a:hover,.nav a.active{background:#ffffff12;color:#fff}.main{padding:22px 28px 60px;min-width:0}.top{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:18px}.top h1{font-size:24px;margin:0}.top p{margin:2px 0 0;color:var(--muted)}.status{padding:6px 10px;border:1px solid #bfe4d6;background:#edf9f4;border-radius:999px;color:var(--good);font-weight:700}
.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #10243c0a}.card{padding:15px}.k{font-size:12px;color:var(--muted)}.v{font-size:23px;font-weight:750;margin-top:4px}.panel{margin-top:14px;padding:17px}.panel h2{font-size:17px;margin:0 0 12px}.two{display:grid;grid-template-columns:1.2fr .8fr;gap:14px}.table{width:100%;border-collapse:collapse}.table th,.table td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}.table th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}.scroll{overflow:auto}.pill{display:inline-block;background:#edf4ff;color:#245da8;padding:3px 8px;border-radius:999px;font-size:11px;margin:2px}.empty{color:var(--muted);padding:18px 4px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.foot{margin-top:18px;color:var(--muted);font-size:12px}.bar{height:8px;background:#e7edf3;border-radius:999px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,#1e6eea,#17a67a)}
@media(max-width:950px){.shell{grid-template-columns:1fr}.side{display:none}.grid{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.main{padding:18px}}
</style>
</head>
<body>
<div class="shell">
<aside class="side"><div class="brand">IndusTSFM<small>INDUSTRIAL INTELLIGENCE STUDIO</small></div><nav class="nav"><a class="active" href="#overview">Overview</a><a href="#projects">Projects</a><a href="#models">Model Hub</a><a href="#applications">Applications</a><a href="#connectors">Data Sources</a><a href="#runtime">Runtime</a></nav></aside>
<main class="main">
<div class="top"><div><h1>Industrial Time-Series Intelligence</h1><p>Data audit → AutoModel → anomaly triage → application runtime</p></div><span id="health" class="status">checking</span></div>
<section id="overview" class="grid"><div class="card"><div class="k">Projects</div><div id="projectCount" class="v">—</div></div><div class="card"><div class="k">Applications</div><div id="appCount" class="v">—</div></div><div class="card"><div class="k">Implemented models</div><div id="modelCount" class="v">—</div></div><div class="card"><div class="k">Product routes</div><div id="routeCount" class="v">—</div></div></section>
<div class="two">
<section id="projects" class="panel"><h2>Projects</h2><div id="projectTable" class="scroll empty">Loading…</div></section>
<section id="connectors" class="panel"><h2>Data connectors</h2><div id="connectorsBody" class="empty">Loading…</div></section>
</div>
<section id="models" class="panel"><h2>Model Hub</h2><div id="modelTable" class="scroll empty">Loading…</div></section>
<section id="applications" class="panel"><h2>Applications</h2><div id="appTable" class="scroll empty">Loading…</div></section>
<section id="runtime" class="panel"><h2>Runtime boundary</h2><p>V1 runs deterministic offline replay and audited local application materialization. Live MQTT/OPC-UA execution, remote authentication, and closed-loop OT control are intentionally not enabled yet.</p><div class="bar"><i style="width:48%"></i></div><div class="foot">Platform V1 productization progress — deterministic numerical layer first; Agent orchestration later.</div></section>
</main></div>
<script>
const esc=x=>String(x??'—').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
async function get(path){const r=await fetch(path);if(!r.ok)throw new Error(path+' '+r.status);return r.json()}
function table(rows,cols){if(!rows.length)return '<div class="empty">No records yet.</div>';return '<table class="table"><thead><tr>'+cols.map(c=>'<th>'+esc(c[0])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(c[1](r))+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
async function boot(){try{const [h,c,p,a]=await Promise.all([get('/health'),get('/v1/capabilities'),get('/v1/projects'),get('/v1/applications')]);document.getElementById('health').textContent=h.status==='ok'?'API healthy':'degraded';document.getElementById('projectCount').textContent=p.count;document.getElementById('appCount').textContent=a.count;document.getElementById('modelCount').textContent=c.models.length;document.getElementById('routeCount').textContent=c.implemented_product_routes.length;
document.getElementById('projectTable').className='scroll';document.getElementById('projectTable').innerHTML=table(p.projects,[['Project',r=>r.spec?.name],['ID',r=>r.project_id],['Tasks',r=>r.spec?.tasks?.length??0],['Updated',r=>r.updated_at]]);
document.getElementById('modelTable').className='scroll';document.getElementById('modelTable').innerHTML=table(c.models,[['Model',r=>r.name],['Family',r=>r.family],['Tasks',r=>r.task_types?.join(', ')],['Mode',r=>r.training_modes?.join(', ')],['Multivariate',r=>r.native_multivariate?'yes':'no']]);
document.getElementById('appTable').className='scroll';document.getElementById('appTable').innerHTML=table(a.applications,[['Application',r=>r.application_id],['Status',r=>r.status],['Task',r=>r.task],['Model',r=>r.selected_model],['Strategy',r=>r.selected_strategy]]);
document.getElementById('connectorsBody').className='';document.getElementById('connectorsBody').innerHTML='<div>'+c.data_source_contracts.map(x=>'<span class="pill">'+esc(x)+'</span>').join('')+'</div><p class="foot">Implemented local: '+esc(c.implemented_local_connectors.join(', '))+'</p>';
}catch(e){document.getElementById('health').textContent='API error';document.getElementById('health').style.color='var(--warn)';console.error(e)}}boot();
</script>
</body></html>'''
