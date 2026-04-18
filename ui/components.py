from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st


def render_top_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class=\"tw-header\">
  <div style=\"display:flex;align-items:center;gap:12px;\">
    <div class=\"tw-icon-btn\">☰</div>
    <div>
      <div class=\"tw-header-title\">{title}</div>
      <div class=\"tw-header-sub\">{subtitle}</div>
    </div>
  </div>
  <div class=\"tw-actions\">
    <div class=\"tw-icon-btn\">⌕</div>
    <div class=\"tw-icon-btn\">⇩</div>
    <div class=\"tw-icon-btn\">?</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_action_row(items: Sequence[tuple[str, str]]) -> None:
    blocks = "".join([f'<div class="action-item"><div class="icon">{icon}</div><div>{label}</div></div>' for icon, label in items])
    st.markdown(f'<div class="action-grid">{blocks}</div>', unsafe_allow_html=True)


def render_chip_row(chips: Iterable[tuple[str, str]]) -> None:
    html = "".join([f'<span class="chip {cls}">{text}</span>' for text, cls in chips])
    st.markdown(f'<div class="chip-row">{html}</div>', unsafe_allow_html=True)


def render_action_buttons(items: Sequence[tuple[str, str, str]]) -> str | None:
    cols = st.columns(len(items))
    clicked: str | None = None
    for idx, (action_id, icon, label) in enumerate(items):
        with cols[idx]:
            if st.button(f"{icon}  {label}", key=f"action_btn_{action_id}", use_container_width=True):
                clicked = action_id
    return clicked


def render_kpi_cards(items: Sequence[dict]) -> None:
    cards = []
    for item in items:
        delta_cls = "kpi-delta-pos" if item.get("positive", True) else "kpi-delta-neg"
        cards.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{item.get("label", "")}</div>
                <div class="kpi-val">{item.get("value", "")}</div>
                <div class="{delta_cls}">{item.get("delta", "")}</div>
            </div>
            """
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
