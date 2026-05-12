from __future__ import annotations

import importlib.util

if importlib.util.find_spec("streamlit") is not None:
    import streamlit as st
else:
    st = None


def apply_theme() -> None:
    if st is None:
        return
    st.markdown(
        """
<style>
:root {
  --app-bg:#0F1722;
  --surface:#233142;
  --surface-hover:#28384B;
  --surface-soft:#28384B;
  --header-bg:#243243;
  --divider:rgba(255,255,255,0.08);
  --subtle-border:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.08);
  --text:#F4F7FB;
  --text-main:#F4F7FB;
  --text-secondary:#93A3B8;
  --text-muted:#71829A;
  --accent:#6F70FF;
  --accent-line:#A89CFF;
  --success:#7AD0A9;
  --warning:#E7B768;
  --danger:#E67C7C;
  --radius-card:24px;
  --radius-action:18px;
}

html, body, .stApp, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); }
.stApp { background: var(--app-bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1260px; padding: 24px; }

.surface-card { background: var(--surface); border: 1px solid var(--subtle-border); border-radius: var(--radius-card); padding: 22px; margin-bottom: 18px; }
.section-title { font-size: clamp(1.8rem,3vw,2.4rem); font-weight: 700; margin: 0 0 8px; color: var(--text); }
.section-subtitle { color: var(--text-secondary); margin-bottom: 12px; }
.card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; color: var(--text); }
.muted { color: var(--text-secondary); }
.mini { color: var(--text-muted); font-size: .82rem; }

.top-header {
  display:flex; align-items:center; justify-content:space-between;
  background: var(--header-bg); border-radius: 0 0 24px 24px;
  height: 72px; padding: 0 24px; margin: -24px -24px 20px;
}
.top-header-left, .top-header-right { display:flex; align-items:center; gap: 10px; }
.top-header-title { font-size: clamp(1.75rem,3vw,2rem); font-weight: 700; letter-spacing: .2px; }
.top-header-title .accent { color: var(--accent-line); }
.icon-btn {
  width: 38px; height: 38px; border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  background: var(--surface); border: 1px solid var(--subtle-border); color: var(--text);
}

.object-header { background: var(--surface); border: 1px solid var(--subtle-border); border-radius: var(--radius-card); padding: 22px; margin-bottom: 14px; }
.object-row { display:flex; align-items:center; gap: 14px; }
.obj-badge { width: 64px; height: 64px; border-radius: 999px; background: var(--header-bg); display:flex; align-items:center; justify-content:center; font-size: 1.3rem; }
.obj-title { font-size: clamp(1.8rem,4vw,2.25rem); font-weight:700; line-height:1.15; }
.obj-meta { color: var(--text-secondary); font-size: .95rem; }

.metric-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; }
.metric-item { background: var(--surface-hover); border-radius: 16px; padding: 14px; border:1px solid var(--subtle-border); }
.big-metric { font-size: clamp(2.3rem,5vw,3.2rem); font-weight: 800; line-height:1.05; }

.landing-nav, .hero-grid, .grid-3, .grid-4, .step-grid { display:grid; gap: 16px; }
.landing-nav { grid-template-columns: 1fr auto; align-items:center; background: var(--header-bg); border-radius: 20px; padding: 12px 16px; }
.hero-grid { grid-template-columns: 1.1fr 1fr; align-items: stretch; }
.grid-4 { grid-template-columns: repeat(4,minmax(0,1fr)); }
.grid-3 { grid-template-columns: repeat(3,minmax(0,1fr)); }
.step-grid { grid-template-columns: repeat(4,minmax(0,1fr)); }
.hero-headline { font-size: clamp(2.2rem,6vw,4.2rem); line-height:1.05; margin: 8px 0 12px; font-weight: 800; }
.eyebrow { color: var(--accent-line); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }
.divider-top { border-top: 1px solid var(--divider); padding-top: 8px; margin-top: 8px; }
.chip-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.floating-chip {
  display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
  background: rgba(111,112,255,0.18); border:1px solid rgba(168,156,255,0.35); color:var(--text); font-size: .78rem;
}
.hero-visual { position: relative; overflow: hidden; min-height: 320px; background: linear-gradient(160deg, rgba(111,112,255,0.10), rgba(35,49,66,0.95)); }
.hero-chart-title { color: var(--text-secondary); font-size:.86rem; margin-bottom:6px; }
.hero-chart-svg { width:100%; height:220px; background:rgba(0,0,0,0.16); border:1px solid var(--subtle-border); border-radius:16px; }
.metric-float {
  position:absolute; background:rgba(24,35,49,0.90); border:1px solid rgba(122,208,169,0.30); border-radius:14px; padding:8px 10px;
  display:flex; flex-direction:column; font-size:.76rem; color:var(--text-secondary);
}
.metric-float b { color:var(--success); font-size:.95rem; }
.chip-a { position:absolute; left:24px; top:56px; }
.chip-b { position:absolute; right:28px; top:46px; }
.chip-c { position:absolute; left:154px; top:250px; }
.metric-a { left:24px; bottom:18px; }
.metric-b { left:170px; bottom:18px; }
.metric-c { left:316px; bottom:18px; }
.report-badge {
  display:inline-flex; margin-top:10px; padding:6px 10px; border-radius:999px; background:rgba(122,208,169,0.16);
  border:1px solid rgba(122,208,169,0.45); color:var(--success); font-size:.78rem; font-weight:600;
}
.proof-strip {
  margin: 14px 0 18px; background: var(--surface); border:1px solid var(--subtle-border); border-radius:18px;
  padding: 10px; display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px;
}
.proof-item {
  background: rgba(255,255,255,0.02); border:1px solid var(--subtle-border); border-radius:14px; padding:10px 12px;
  font-size:.84rem; color:var(--text-secondary);
}
.split-showcase { display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin: 6px 0 18px; }
.landing-list { margin: 8px 0 14px 16px; color: var(--text-secondary); line-height: 1.55; }
.landing-list li { margin-bottom: 4px; }
.mini-control-panel, .mini-dashboard {
  background: rgba(20,30,42,0.72); border:1px solid var(--subtle-border); border-radius:16px; padding:12px;
}
.mini-slider { margin:8px 0; }
.mini-slider span { display:block; color:var(--text-muted); font-size:.74rem; margin-bottom:4px; }
.mini-slider div { height:6px; border-radius:999px; background:linear-gradient(90deg, var(--accent), rgba(255,255,255,0.15)); }
.mini-control-panel button {
  margin-top:10px; border-radius:12px; background:var(--accent); border:1px solid var(--accent); color:white; padding:7px 10px;
}
.mini-dashboard-grid { display:grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap:8px; margin-top:10px; }
.mini-dashboard-grid div { background:rgba(255,255,255,0.02); border:1px solid var(--subtle-border); border-radius:12px; padding:8px; }
.mini-dashboard-grid span { display:block; color:var(--text-muted); font-size:.74rem; }
.mini-dashboard-grid b { color:var(--text); font-size:.95rem; }
.pipeline-row {
  margin: 8px 0 18px; background: var(--surface); border:1px solid var(--subtle-border); border-radius:18px; padding:14px;
  display:grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; align-items:center; gap:8px;
}
.pipeline-step {
  background:rgba(255,255,255,0.02); border:1px solid var(--subtle-border); border-radius:14px; padding:10px; text-align:center;
  min-height:74px; display:flex; flex-direction:column; justify-content:center; gap:6px;
}
.pipeline-step span { font-size:1.05rem; }
.pipeline-step b { font-size:.8rem; font-weight:600; color:var(--text-secondary); }
.pipeline-arrow { color:var(--text-muted); font-size:1rem; text-align:center; }
.mini-report-preview { margin-top:10px; }
.cta-float {
  position:absolute; background:rgba(35,49,66,0.94); border:1px solid var(--subtle-border); border-radius:999px;
  padding:6px 10px; font-size:.74rem; color:var(--text-secondary);
}
.cta-a { top:14px; left:22px; }
.cta-b { top:14px; right:22px; color:var(--success); border-color:rgba(122,208,169,0.35); }
.cta-c { bottom:14px; right:22px; color:var(--accent-line); border-color:rgba(168,156,255,0.35); }

/* Tabs: text-only look with underline on active, like reference */
[role="radiogroup"] {
  overflow-x: auto !important;
  white-space: nowrap !important;
  flex-wrap: nowrap !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  gap: 20px !important;
  margin-bottom: 20px;
}
[role="radiogroup"] label {
  background: transparent !important;
  border: none !important;
  color: var(--text-muted) !important;
  padding: 0 0 8px 0 !important;
  border-radius: 0 !important;
}
[role="radiogroup"] label[data-checked="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--text) !important;
}

.stButton button, .stDownloadButton button {
  border-radius: 18px !important;
  border: 1px solid var(--subtle-border) !important;
  background: var(--surface-hover) !important;
  color: var(--text) !important;
}
.stButton button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; }

.status-dot.status-success { color: var(--success); }
.status-dot.status-warning { color: var(--warning); }
.status-dot.status-muted { color: var(--text-muted); }
.status-dot.status-danger { color: var(--danger); }
.metric-card, .next-action-card, .scenario-preview-card, .risk-card {
  background: var(--surface-hover); border: 1px solid var(--border); border-radius: 18px; padding: 16px; margin-bottom: 12px;
}
.metric-card-label { color: var(--text-secondary); font-size: .82rem; }
.metric-card-value { color: var(--text-main); font-weight: 800; font-size: 1.35rem; margin-top: 4px; }
.metric-card-delta { color: var(--text-muted); font-size: .86rem; margin-top: 6px; }
.next-action-card.success, .metric-card.success { border-color: color-mix(in srgb, var(--success) 45%, transparent); }
.next-action-card.warning, .metric-card.warning, .risk-card.warning { border-color: color-mix(in srgb, var(--warning) 45%, transparent); }
.next-action-card.danger, .metric-card.danger, .risk-card.danger { border-color: color-mix(in srgb, var(--danger) 45%, transparent); }
.risk-card ul { margin: 8px 0 0 18px; color: var(--text-secondary); line-height: 1.55; }

button:focus, input:focus, textarea:focus, [role="radio"]:focus, [role="radio"]:focus-visible, [data-baseweb="select"]:focus-within {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}
button { min-height: 42px; }

[data-testid="stMetric"] { background: var(--surface-hover); border: 1px solid var(--subtle-border); border-radius: 16px; padding: 12px; }
.stExpander { border:1px solid var(--subtle-border) !important; background: var(--surface-hover) !important; border-radius: 16px !important; }

.scenario-shell { display:flex; flex-direction:column; gap:16px; }
.scenario-status-card {
  background: var(--surface); border:1px solid var(--subtle-border); border-radius: var(--radius-card);
  padding:18px 20px; display:flex; justify-content:space-between; align-items:center; gap:16px;
}
.scenario-status-left {}
.scenario-status-title { font-size:1.05rem; font-weight:700; color:var(--text); }
.scenario-status-subtitle { color:var(--text-secondary); font-size:.9rem; margin-top:4px; }
.scenario-pill {
  display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
  font-size:.78rem; font-weight:600; border:1px solid var(--subtle-border); background:rgba(255,255,255,0.03);
}
.scenario-pill-success { color:var(--success); border-color:rgba(122,208,169,0.35); background:rgba(122,208,169,0.12); }
.scenario-pill-warning { color:var(--warning); border-color:rgba(231,183,104,0.35); background:rgba(231,183,104,0.12); }
.scenario-pill-muted { color:var(--text-secondary); border-color:var(--subtle-border); background:rgba(255,255,255,0.03); }
.scenario-grid { display:grid; grid-template-columns:minmax(0, 1.15fr) minmax(320px, .85fr); gap:16px; align-items:start; }
.scenario-card {
  background:var(--surface); border:1px solid var(--subtle-border); border-radius:var(--radius-card);
  padding:20px; margin-bottom:16px;
}
.scenario-card-header { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px; }
.scenario-card-title { font-size:1.05rem; font-weight:700; color:var(--text); }
.scenario-card-caption { color:var(--text-secondary); font-size:.88rem; line-height:1.45; margin-top:4px; }
.scenario-step {
  width:28px; height:28px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center;
  background:rgba(111,112,255,0.18); border:1px solid rgba(168,156,255,0.35); color:var(--accent-line); font-weight:700; font-size:.8rem;
}
.scenario-preview {
  background:linear-gradient(160deg, rgba(111,112,255,0.10), rgba(35,49,66,0.92));
  border:1px solid rgba(168,156,255,0.20); border-radius:var(--radius-card); padding:20px;
}
.preview-row { display:flex; justify-content:space-between; gap:12px; padding:10px 0; border-bottom:1px solid var(--divider); }
.preview-row:last-child { border-bottom:none; }
.preview-label { color:var(--text-secondary); font-size:.86rem; }
.preview-value { color:var(--text); font-weight:700; text-align:right; }
.result-kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; }
.result-kpi-card { background:var(--surface-hover); border:1px solid var(--subtle-border); border-radius:18px; padding:14px; }
.result-kpi-label { color:var(--text-secondary); font-size:.82rem; }
.result-kpi-value { font-size:1.55rem; font-weight:800; color:var(--text); margin-top:6px; }
.result-kpi-delta { font-size:.86rem; margin-top:8px; color:var(--text-secondary); }
.delta-positive { color:var(--success); }
.delta-negative { color:var(--danger); }
.delta-neutral { color:var(--text-secondary); }
.effect-list { display:grid; gap:8px; }
.effect-row {
  display:flex; justify-content:space-between; align-items:center; padding:10px 12px;
  background:rgba(255,255,255,0.025); border:1px solid var(--subtle-border); border-radius:14px;
}
.effect-name { color:var(--text-secondary); font-size:.9rem; }
.effect-value { color:var(--text); font-weight:700; }
.scenario-help { color:var(--text-muted); font-size:.82rem; line-height:1.45; margin-top:6px; }
.scenario-divider { border-top:1px solid var(--divider); margin:16px 0; }
.scenario-warning-inline, .scenario-success-inline, .scenario-danger-inline {
  border-radius:14px; padding:10px 12px; font-size:.86rem; line-height:1.45;
}
.scenario-warning-inline { color:var(--warning); background:rgba(231,183,104,0.10); border:1px solid rgba(231,183,104,0.25); }
.scenario-success-inline { color:var(--success); background:rgba(122,208,169,0.10); border:1px solid rgba(122,208,169,0.25); }
.scenario-danger-inline { color:var(--danger); background:rgba(230,124,124,0.10); border:1px solid rgba(230,124,124,0.25); }
.page-header { font-size:1.65rem; font-weight:800; margin:8px 0 4px; }
.page-subtitle { color:var(--text-secondary); margin-bottom:10px; }
.stepper { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:10px 0 14px; }
.stepper-item { background:rgba(255,255,255,0.02); border:1px solid var(--subtle-border); border-radius:12px; padding:10px; display:flex; gap:8px; }
.stepper-item.active { border-color:rgba(168,156,255,0.5); background:rgba(111,112,255,0.12); }
.stepper-item.done { border-color:rgba(122,208,169,0.4); }
.stepper-index { width:22px; height:22px; border-radius:999px; background:var(--header-bg); text-align:center; line-height:22px; font-size:.75rem; }
.stepper-title { font-weight:700; font-size:.8rem; }
.stepper-caption { color:var(--text-muted); font-size:.72rem; }
.decision-card { border-radius:16px; padding:14px; border:1px solid var(--subtle-border); background:var(--surface-hover); margin-bottom:12px; }
.decision-card.success { border-color:rgba(122,208,169,0.4); }
.decision-card.warning { border-color:rgba(231,183,104,0.4); }
.decision-card.danger { border-color:rgba(230,124,124,0.4); }
.decision-title { font-weight:800; margin-bottom:6px; }
.decision-reason { color:var(--text-secondary); margin-bottom:10px; }
.decision-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }
.decision-metric, .kpi-card { background:rgba(255,255,255,0.02); border:1px solid var(--subtle-border); border-radius:12px; padding:10px; }
.decision-metric-label, .kpi-label { color:var(--text-secondary); font-size:.78rem; }
.decision-metric-value, .kpi-value { font-weight:700; margin-top:4px; }
.decision-metric-delta, .kpi-delta, .kpi-base, .technical-muted { color:var(--text-muted); font-size:.75rem; margin-top:3px; }
.help-callout { border-radius:12px; padding:10px 12px; margin:8px 0; }
.help-callout.info { border:1px solid var(--subtle-border); background:rgba(255,255,255,0.03); }
.help-callout.success { border:1px solid rgba(122,208,169,0.35); background:rgba(122,208,169,0.10); color:var(--success); }
.help-callout.warning { border:1px solid rgba(231,183,104,0.35); background:rgba(231,183,104,0.10); color:var(--warning); }
.help-callout.danger { border:1px solid rgba(230,124,124,0.35); background:rgba(230,124,124,0.10); color:var(--danger); }

.decision-hero {
  background: linear-gradient(160deg, rgba(111,112,255,0.16), rgba(35,49,66,0.96));
  border: 1px solid rgba(168,156,255,0.28);
  border-radius: var(--radius-card);
  padding: 22px;
  margin-bottom: 18px;
}
.decision-hero.success { border-color: rgba(122,208,169,0.42); }
.decision-hero.warning { border-color: rgba(231,183,104,0.42); }
.decision-hero.danger { border-color: rgba(230,124,124,0.42); }
.decision-hero-title { font-size: 1.25rem; font-weight: 800; color: var(--text); margin-bottom: 8px; }
.decision-hero-text { color: var(--text-secondary); line-height: 1.5; }
.decision-section-grid, .decision-hero-grid, .decision-plan-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
  gap: 12px;
  margin: 12px 0;
}
.decision-section-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--subtle-border);
  border-radius: 16px;
  padding: 14px;
}
.decision-section-label { color: var(--text-secondary); font-size: .82rem; margin-bottom: 4px; }
.decision-section-value { color: var(--text); font-weight: 700; }
.decision-next-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(180px,1fr));
  gap: 10px;
  margin-top: 12px;
}
.decision-next-item {
  border-radius: 14px;
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--subtle-border);
  padding: 12px;
  color: var(--text-secondary);
}

@media (max-width: 768px) {
  .kpi-row,
  .result-kpi-grid,
  .decision-section-grid,
  .decision-next-grid,
  .scenario-grid {
    grid-template-columns: 1fr !important;
  }
  .surface-card { padding: 16px; border-radius: 18px; }
  .page-header { font-size: 26px; }
}

@media (max-width: 900px) {
  .block-container { padding: 16px; }
  .top-header { margin: -16px -16px 16px; padding: 0 16px; height:80px; }
  .hero-grid, .grid-4, .grid-3, .step-grid, .metric-grid, .split-showcase, .proof-strip { grid-template-columns: 1fr; }
  .pipeline-row { grid-template-columns: 1fr; }
  .pipeline-arrow { display:none; }
  .metric-a, .metric-b, .metric-c { position: static; margin-top:8px; }
  .chip-a, .chip-b, .chip-c { position: static; margin-right:6px; }
  .hero-visual { min-height: auto; }
  .cta-float { position: static; display:inline-flex; margin:8px 4px 0; }
  [role="radiogroup"] { display: flex !important; gap: 16px !important; padding-bottom: 8px !important; }
  [role="radiogroup"] label { min-width: max-content !important; }
  .object-row { align-items: flex-start; }
  .obj-badge { width: 44px; height: 44px; font-size: 1rem; }
  .scenario-grid, .result-kpi-grid, .kpi-row, .decision-grid, .stepper, .decision-hero-grid, .decision-plan-grid, .decision-next-grid { grid-template-columns: 1fr !important; }
  .scenario-card, .decision-card, .surface-card { padding:16px !important; }
  .top-header-right { display:none !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )
