from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --bg:#0B1420;
  --section:#101C2B;
  --card:#162334;
  --card-elev:#1A2940;
  --border:rgba(255,255,255,.08);
  --text:#F4F7FB;
  --text-2:#A8B4C7;
  --text-3:#7D8AA0;
  --accent:#6C63FF;
  --accent-hover:#7A73FF;
  --success:#62D394;
  --warn:#F5C46B;
  --danger:#FF6B7A;
  --line:#8B80FF;
  --line-alt:#7ED6A2;
}

html, body, [class*="css"], .stApp { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); }
.stApp { background: linear-gradient(180deg, #0b1420 0%, #0e1726 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1280px; padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; padding-bottom: 3rem; }

.tw-header {
  display:flex; align-items:center; justify-content:space-between;
  background: var(--section); border:1px solid var(--border); border-radius: 20px;
  padding: 12px 16px; margin-bottom: 14px;
}
.tw-header-title{font-size:1rem;font-weight:700;}
.tw-header-sub{font-size:.78rem;color:var(--text-2)}
.tw-actions{display:flex; gap:8px;}
.tw-icon-btn{width:38px;height:38px;border-radius:14px;background:var(--card);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;color:var(--text-2);font-size:1rem;}

.cloud-card, .glass-card {
  background: var(--card);
  border:1px solid var(--border);
  border-radius:22px;
  padding:20px;
  box-shadow: 0 10px 28px rgba(0,0,0,.26);
  margin-bottom: 14px;
}
.cloud-card.elevated { background: var(--card-elev); }
.section-title { font-size: clamp(1.35rem, 2.6vw, 1.7rem); font-weight: 700; margin: 2px 0 12px; }
.card-title { font-size: 1.12rem; font-weight: 600; margin-bottom: 8px; }
.muted, .micro-note { color: var(--text-2); font-size: .84rem; }

.kpi-grid { display:grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap:12px; }
.kpi-card { background: var(--card-elev); border:1px solid var(--border); border-radius:20px; padding:18px; }
.kpi-label { font-size:.8rem;color:var(--text-2); }
.kpi-val { font-size: clamp(1.45rem, 3.2vw, 2.2rem); font-weight: 800; line-height:1.1; margin: 6px 0; }
.kpi-delta-pos { color: var(--success); font-size:.82rem; font-weight:600;}
.kpi-delta-neg { color: var(--danger); font-size:.82rem; font-weight:600;}
.kpi-pos { color: var(--success); font-size:.82rem; font-weight:600;}
.kpi-neg { color: var(--danger); font-size:.82rem; font-weight:600;}

.chip-row, .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top: 8px; }
.chip, .pill { border:1px solid var(--border); border-radius:999px; padding:5px 10px; font-size:.77rem; color:var(--text-2); background: rgba(255,255,255,.02); }
.chip.ok { color: var(--success); }
.chip.warn { color: var(--warn); }
.chip.danger { color: var(--danger); }
.chip.active { color: #b8b3ff; border-color: rgba(108,99,255,.45); }

.action-grid { display:grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap:10px; margin-top: 10px; }
.action-item {
  border-radius:16px; border:1px solid var(--border); background: var(--card-elev);
  min-height:84px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; color:var(--text); font-size:.8rem; font-weight:600;
}
.action-item .icon{font-size:1.2rem; margin-bottom:6px; color:#b8b3ff;}

.stTabs [data-baseweb="tab-list"] { background: var(--section); border:1px solid var(--border); border-radius:16px; padding: 6px; gap:8px; }
.stTabs [data-baseweb="tab"] {
  border-radius:14px; min-height:42px; color:var(--text-2); padding: 0 14px;
}
.stTabs [aria-selected="true"] { background: rgba(108,99,255,.18) !important; color: #e7e6ff !important; }
[role="radiogroup"] {
  background: var(--section);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 6px;
  gap: 8px;
}
[role="radiogroup"] label {
  background: transparent !important;
  border-radius: 14px !important;
  padding: 8px 12px !important;
}
[role="radiogroup"] label[data-checked="true"] {
  background: rgba(108,99,255,.18) !important;
  border: 1px solid rgba(108,99,255,.45) !important;
}

.stButton button, .stDownloadButton button {
  border-radius:16px !important; border:1px solid var(--border) !important;
  background: var(--card-elev) !important; color: var(--text) !important;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
  background: var(--accent) !important; border-color: var(--accent) !important;
}
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--accent-hover) !important; color: #fff !important; }

[data-testid="stMetric"] {
  background: var(--card-elev); border:1px solid var(--border); border-radius:18px; padding:14px;
}
[data-testid="stDataFrame"] div[role="table"] { background: var(--card-elev); }
.stExpander { border:1px solid var(--border) !important; border-radius:16px !important; background: var(--card-elev) !important; }

@media (max-width: 1024px){ .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));} .action-grid{grid-template-columns:repeat(3,minmax(0,1fr));} }
@media (max-width: 760px){
  .block-container{padding-left: .75rem; padding-right:.75rem;}
  .tw-header{padding: 10px 12px; border-radius: 16px;}
  .cloud-card,.glass-card{padding:16px; border-radius:20px;}
  .kpi-grid,.action-grid{grid-template-columns:1fr;}
}
</style>
""",
        unsafe_allow_html=True,
    )
