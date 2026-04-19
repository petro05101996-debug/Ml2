from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --app-bg:#0F1722;
  --surface:#233142;
  --surface-hover:#28384B;
  --header-bg:#243243;
  --divider:rgba(255,255,255,0.08);
  --subtle-border:rgba(255,255,255,0.04);
  --text:#F4F7FB;
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

/* Tabs: text-only look with underline on active, like reference */
[role="radiogroup"] {
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
[data-testid="stMetric"] { background: var(--surface-hover); border: 1px solid var(--subtle-border); border-radius: 16px; padding: 12px; }
.stExpander { border:1px solid var(--subtle-border) !important; background: var(--surface-hover) !important; border-radius: 16px !important; }

@media (max-width: 900px) {
  .block-container { padding: 16px; }
  .top-header { margin: -16px -16px 16px; padding: 0 16px; height:80px; }
  .hero-grid, .grid-4, .grid-3, .step-grid, .metric-grid { grid-template-columns: 1fr; }
}
</style>
""",
        unsafe_allow_html=True,
    )
