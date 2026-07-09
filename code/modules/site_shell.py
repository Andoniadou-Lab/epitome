import base64
import hashlib
import hmac
import os
from pathlib import Path

import streamlit as st

from config import Config
from modules.citations import epitome_citation, print_citation
from modules.page_runner import page_with_footer

BASE_PATH = Config.BASE_PATH
_ASSETS = Path(__file__).parent.parent / "assets"

SITE_MOUSE = "mouse"
SITE_TUMOR = "tumor"
ACCENT_MOUSE = "#0000ff"
ACCENT_TUMOR = "#cc0000"

# Maintenance banner — set False or comment out render_maintenance_banner() in epitome.py to hide.
SHOW_MAINTENANCE_BANNER = True
_MAINTENANCE_BANNER_TEXT = (
    "Maintenance: new data/functionalities are being added. "
    "For any questions / issues, please email epitome@kcl.ac.uk."
)

_MOUSE_TAGLINE = (
    "Explore, analyse, and visualise all mouse pituitary datasets. "
    "Export raw or processed data, and generate publication-ready figures."
)
_TUMOR_TAGLINE = (
    "Human pituitary <strong>tumour</strong> atlas — single-cell and bulk RNA-seq, "
    "pseudobulk expression, and sample curation."
)


@st.cache_data
def _logo_b64():
    with open(BASE_PATH / "data/images/epitome_logo.svg", "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def inject_site_styles(site: str) -> None:
    accent = ACCENT_TUMOR if site == SITE_TUMOR else ACCENT_MOUSE
    css = (_ASSETS / "epitome.css").read_text()
    st.html(
        f"<style>:root {{ --epitome-accent: {accent}; }}</style><style>{css}</style>"
    )


def render_maintenance_banner() -> None:
    """Site-wide maintenance notice. Toggle via SHOW_MAINTENANCE_BANNER or epitome.py call."""
    if not SHOW_MAINTENANCE_BANNER:
        return
    with st.container(key="epitome_maintenance"):
        st.markdown(
            f'<div class="epitome-maintenance-banner">{_MAINTENANCE_BANNER_TEXT}</div>',
            unsafe_allow_html=True,
        )


def init_session_state() -> None:
    for key, value in {
        "active_site": SITE_MOUSE,
        "tumor_authenticated": False,
        "selected_gene": "Sox2",
        "selected_region": "chr3:34650405-34652461",
        "cached_all": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_tumor() -> None:
    st.session_state.active_site = SITE_TUMOR


def go_to_mouse() -> None:
    st.session_state.active_site = SITE_MOUSE


def _tumor_auth_secrets() -> tuple[str, str] | None:
    try:
        cfg = st.secrets["tumor_auth"]
        return str(cfg["salt"]), str(cfg["hash"])
    except (KeyError, FileNotFoundError, TypeError):
        salt = os.environ.get("TUMOR_AUTH_SALT")
        digest = os.environ.get("TUMOR_AUTH_HASH")
        if salt and digest:
            return salt, digest
    return None


def _verify_tumor_password(candidate: str) -> bool:
    creds = _tumor_auth_secrets()
    if not creds:
        return False
    salt, expected = creds
    actual = hmac.new(
        salt.encode("utf-8"),
        candidate.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(actual, expected)


def _render_logo() -> None:
    st.markdown(
        f'<div style="margin: 0; padding: 0; text-align: left; margin-top: -1rem; margin-bottom: 0;">'
        f'<img src="data:image/svg+xml;base64,{_logo_b64()}" width="300" '
        f'style="margin: 0; padding: 0;"></div>',
        unsafe_allow_html=True,
    )


def render_site_switch_button(site: str) -> None:
    if site == SITE_MOUSE:
        st.button(
            "Human Pituitary Tumour Atlas →",
            key="go_tumor_site",
            on_click=go_to_tumor,
        )
    else:
        st.button(
            "← Mouse Pituitary Atlas",
            key="go_mouse_site",
            on_click=go_to_mouse,
        )


def render_navbar(page_map: dict, site: str) -> None:
    with st.container(key="epitome_navbar"):
        for name, section_pages in page_map.items():
            if len(section_pages) == 1:
                st.page_link(section_pages[0], label=name)
            else:
                with st.popover(name):
                    for page in section_pages:
                        st.page_link(page, use_container_width=True)
        render_site_switch_button(site)


def _render_header(page_map: dict | None, site: str, tagline: str) -> None:
    with st.container(key="epitome_header"):
        _render_logo()
        st.markdown(
            f'<p style="margin: 0.6rem 0 0.4rem 0; font-size: 1rem;">{tagline}</p>',
            unsafe_allow_html=True,
        )
        if page_map:
            render_navbar(page_map, site)
        elif site == SITE_TUMOR:
            with st.container(key="epitome_navbar"):
                render_site_switch_button(site)
        st.markdown('<hr style="margin: 0.1rem 0 0.6rem 0;">', unsafe_allow_html=True)


def render_mouse_header(page_map: dict) -> None:
    _render_header(page_map, SITE_MOUSE, _MOUSE_TAGLINE)


def render_tumor_header(page_map: dict | None) -> None:
    _render_header(page_map, SITE_TUMOR, _TUMOR_TAGLINE)


def render_page_footer() -> None:
    """Render the site footer at the end of a page script."""
    render_footer(st.session_state.get("active_site", SITE_MOUSE))


def render_footer(site: str) -> None:
    pit_color = ACCENT_MOUSE if site == SITE_MOUSE else ACCENT_TUMOR
    with st.container(key="epitome_footer"):
        st.markdown("---")
        st.markdown(
            f'<p class="epitome-footer-line">'
            f'The <i>e<span style="color:{pit_color};">pit</span>ome</i> '
            "is maintained by the <strong>Andoniadou Lab</strong> at <strong>King's College "
            'London</strong>. '
            '<a href="https://bsky.app/profile/pituitarylab.bsky.social">Bluesky</a>'
            '<span class="epitome-footer-sep"> | </span>'
            "Lead curator: Bence Kövér "
            '<a href="https://bsky.app/profile/bencekover.bsky.social">Bluesky</a> '
            "(Email: epitome at kcl dot ac dot uk)"
            '<span class="epitome-footer-sep"> | </span>'
            '<a href="https://github.com/Andoniadou-Lab/epitome">GitHub repository</a>'
            "</p>",
            unsafe_allow_html=True,
        )
        st.caption(print_citation)
        st.caption(epitome_citation)
        st.image(f"{BASE_PATH}/data/images/epitome_logo.svg", width=50)


def render_tumor_password_gate() -> None:
    st.markdown("### Password required")
    if _tumor_auth_secrets() is None:
        st.error(
            "Tumour atlas access is not configured on this server. "
            "Please contact the epitome team."
        )
    else:
        st.markdown(
            "The human pituitary tumour atlas is restricted. Enter the password to continue, "
            "or return to the mouse pituitary atlas."
        )
        password = st.text_input(
            "Password",
            type="password",
            key="tumor_password_input",
            placeholder="Enter password",
        )
        if st.button("Unlock tumour atlas", key="tumor_password_submit", type="primary"):
            if _verify_tumor_password(password):
                st.session_state.tumor_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.button(
        "← Back to Mouse Pituitary Atlas",
        key="tumor_password_back",
        on_click=go_to_mouse,
    )


def build_mouse_pages() -> dict:
    overview = st.Page(
        page_with_footer("app_pages/overview/overview.py"),
        title="Overview",
        default=True,
    )
    transcriptome = [
        st.Page(
            page_with_footer("app_pages/transcriptome/expression_box_plots.py"),
            title="Expression Box Plots",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/expression_umap.py"),
            title="Expression UMAP",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/age_correlation.py"),
            title="Age Correlation",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/isoforms.py"),
            title="Isoforms",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/dot_plots.py"),
            title="Dot Plots",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/cell_type_distribution.py"),
            title="Cell Type Distribution",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/gene_gene_relationships.py"),
            title="Gene-Gene Relationships",
        ),
        st.Page(
            page_with_footer("app_pages/transcriptome/ligand_receptor_interactions.py"),
            title="Ligand-Receptor Interactions",
        ),
    ]
    chromatin = [
        st.Page(
            page_with_footer("app_pages/chromatin/accessibility_distribution.py"),
            title="Accessibility Distribution (Motifs/Enhancers)",
        ),
        st.Page(
            page_with_footer("app_pages/chromatin/motif_enrichment_chromvar.py"),
            title="Motif Enrichment (ChromVAR)",
        ),
        st.Page(
            page_with_footer("app_pages/chromatin/cell_type_distribution_atac.py"),
            title="Cell Type Distribution",
        ),
    ]
    downloads = [
        st.Page(
            page_with_footer("app_pages/datasets/rna_datasets.py"),
            title="Dataset Files (h5ad) - RNA",
        ),
        st.Page(
            page_with_footer("app_pages/datasets/atac_datasets.py"),
            title="Dataset Files (h5ad) - ATAC",
        ),
    ]
    return {
        "Overview": [overview],
        "Transcriptome": transcriptome,
        "Chromatin": chromatin,
        "Multimodal": [
            st.Page(
                page_with_footer("app_pages/multimodal/heatmap_tfs.py"),
                title="Multimodal heatmap of TFs",
            )
        ],
        "Automated Cell Typing": [
            st.Page(
                page_with_footer("app_pages/cell_typing/automated_cell_typing.py"),
                title="Automated Cell Typing",
            )
        ],
        "Individual Datasets": [
            st.Page(
                page_with_footer("app_pages/datasets/rna_datasets.py"),
                title="RNA datasets",
            ),
            st.Page(
                page_with_footer("app_pages/datasets/atac_datasets.py"),
                title="ATAC datasets",
            ),
        ],
        "Downloads": downloads,
        "Curation": [
            st.Page(
                page_with_footer("app_pages/curation/curation.py"),
                title="Curation",
            )
        ],
        "Release Notes": [
            st.Page(
                page_with_footer("app_pages/release_notes/release_notes.py"),
                title="Release Notes",
            )
        ],
        "How to Cite": [
            st.Page(
                page_with_footer("app_pages/citation/how_to_cite.py"),
                title="How to Cite",
            )
        ],
        "Contact": [
            st.Page(
                page_with_footer("app_pages/contact/contact.py"),
                title="Contact",
            )
        ],
    }


def build_tumor_pages() -> dict:
    return {
        "Overview": [
            st.Page(
                page_with_footer("app_pages/tumor/overview.py"),
                title="Overview",
                default=True,
            ),
        ],
        "Transcriptome": [
            st.Page(
                page_with_footer("app_pages/tumor/dot_plots.py"),
                title="Dot Plots",
            ),
            st.Page(
                page_with_footer("app_pages/tumor/cell_type_abundance.py"),
                title="Cell Type Abundance",
            ),
            st.Page(
                page_with_footer("app_pages/tumor/pseudobulk_boxplot.py"),
                title="Pseudobulk Boxplot",
            ),
            st.Page(
                page_with_footer("app_pages/tumor/bulk_boxplot.py"),
                title="Bulk Boxplot",
            ),
            st.Page(
                page_with_footer("app_pages/tumor/bulk_heatmap.py"),
                title="Bulk RNA Heatmap",
            ),
            st.Page(
                page_with_footer("app_pages/tumor/volcano_plots.py"),
                title="Volcano Plots",
            ),
        ],
        "Individual Datasets": [
            st.Page(
                page_with_footer("app_pages/tumor/individual_datasets.py"),
                title="RNA datasets",
            ),
        ],
        "Curation": [
            st.Page(
                page_with_footer("app_pages/tumor/curation.py"),
                title="Curation",
            ),
        ],
        "Release Notes": [
            st.Page(
                page_with_footer("app_pages/tumor/release_notes.py"),
                title="Release Notes",
            ),
        ],
    }
