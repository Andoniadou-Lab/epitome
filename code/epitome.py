import streamlit as st

from modules.analytics import get_session_id
from modules.cached_loaders import load_all_cached_data
from modules.site_shell import (
    SITE_MOUSE,
    SITE_TUMOR,
    build_mouse_pages,
    build_tumor_pages,
    init_session_state,
    inject_site_styles,
    render_footer,
    render_maintenance_banner,
    render_mouse_header,
    render_tumor_header,
    render_tumor_password_gate,
)

st.set_page_config(
    page_title="epitome",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
get_session_id()

site = st.session_state.active_site

if site == SITE_MOUSE and not st.session_state.cached_all:
    with st.spinner("Initialising epitome data caches..."):
        load_all_cached_data(version="v_0.02")
    st.session_state.cached_all = True

inject_site_styles(site)

# Maintenance banner — comment out the next line to hide during normal operation.
render_maintenance_banner()

if site == SITE_MOUSE:
    pages = build_mouse_pages()
    render_mouse_header(pages)
    st.navigation(pages, position="hidden").run()
else:
    unlocked = st.session_state.tumor_authenticated
    pages = build_tumor_pages() if unlocked else None
    render_tumor_header(pages)
    if unlocked:
        st.navigation(pages, position="hidden").run()
    else:
        render_tumor_password_gate()
        render_footer(site)
