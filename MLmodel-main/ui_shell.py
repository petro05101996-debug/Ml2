from __future__ import annotations

import streamlit as st

PAGES = ["Обзор", "Настройка", "Лаборатория сценариев", "Результаты", "Модель и документация"]


def apply_enterprise_styles() -> None:
    st.markdown(
        """
<style>
:root {--bg:#0c111b;--panel:#121a29;--panel2:#0f1726;--line:#22324d;--text:#e7eefc;--muted:#9fb0cf;--accent:#5da0ff;--good:#4ecb8f;--warn:#f0b35f;--bad:#ef6f7d;}
.stApp {background:#0c121d; color:var(--text);} .block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:2rem;}
#MainMenu, footer, header {visibility:hidden;}
.enterprise-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px;}
.primary-card{background:#132039;border-color:#2d4e80;} .secondary-card{background:var(--panel2);} .flat-card{background:rgba(255,255,255,.01);} 
.page-title{font-size:1.75rem;font-weight:700;margin:0 0 .2rem;} .muted{color:var(--muted);} .card-title{font-weight:650;margin-bottom:.55rem;} .kpi-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;}
.kpi{background:#0f1a2a;border:1px solid #243958;border-radius:10px;padding:12px;} .kpi label{color:var(--muted);font-size:.78rem;display:block;} .kpi strong{font-size:1.1rem;}
.top-nav-wrap{margin-bottom:12px;} .diagram-card{min-height:130px;} .diagram-svg{width:100%;height:130px;}
.factor-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;} .factor-grid span{background:#111f33;border:1px solid #274266;border-radius:8px;padding:8px;text-align:center;font-size:.8rem;}
.flow-row{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;} .flow-row div{background:#111f33;border:1px solid #2a4468;padding:10px;border-radius:8px;text-align:center;font-size:.8rem;}
.compare-bars > div{margin:8px 0;} .compare-bars label{font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px;} .bar{height:14px;border-radius:999px;} .bar.current{background:#6f86ad}.bar.reco{background:#4ecb8f}.bar.cons{background:#f0b35f}
.interp-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;} .interp-grid div{background:#101c2f;border:1px solid #294565;border-radius:8px;padding:8px;} .interp-grid p{margin:2px 0 0;font-size:.75rem;color:var(--muted);}
.step-card{padding:14px;border:1px solid #27415f;background:#101a2b;border-radius:12px;margin-bottom:12px;}
@media (max-width:900px){.kpi-strip,.factor-grid,.flow-row,.interp-grid{grid-template-columns:1fr;}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
    current = st.session_state.get("active_page", "Обзор")
    selected = st.segmented_control("", options=PAGES, default=current, key="top_nav")
    st.markdown("</div>", unsafe_allow_html=True)
    if selected:
        st.session_state.active_page = selected
    return st.session_state.get("active_page", "Обзор")
