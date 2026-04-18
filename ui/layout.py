from __future__ import annotations

import streamlit as st


def open_card(title: str | None = None, elevated: bool = False) -> None:
    cls = "cloud-card elevated" if elevated else "cloud-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)


def close_card() -> None:
    st.markdown('</div>', unsafe_allow_html=True)
