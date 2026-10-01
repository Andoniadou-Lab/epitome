import base64
import hashlib
import hmac
import os
from pathlib import Path

import streamlit as st

from config import Config
from modules.analytics import record_site_access
from modules.citations import epitome_citation, print_citation
from modules.page_runner import page_with_footer

BASE_PATH = Config.BASE_PATH
_ASSETS = Path(__file__).parent.parent / "assets"

SITE_MOUSE = "mouse"
SITE_TUMOR = "tumor"
SITE_OTHER = "other"
ACCENT_MOUSE = "#0000ff"
ACCENT_TUMOR = "#cc0000"
ACCENT_OTHER = "#058005"

# Maintenance banner — set False or comment out render_maintenance_banner() in epitome.py to hide.
SHOW_MAINTENANCE_BANNER = False
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
_OTHER_TAGLINE = "Other."


@st.cache_data
def _logo_b64():
    with open(BASE_PATH / "data/images/epitome_logo.svg", "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _site_accent(site: str) -> str:
    if site == SITE_TUMOR:
        return ACCENT_TUMOR
    if site == SITE_OTHER:
        return ACCENT_OTHER
    return ACCENT_MOUSE


def inject_site_styles(site: str) -> None:
    accent = _site_accent(site)
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
        "other_authenticated": False,
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


def go_to_other() -> None:
    st.session_state.active_site = SITE_OTHER


def _auth_secrets(section: str, salt_env: str, hash_env: str, hashes_env: str) -> tuple[str, list[str]] | None:
    """HMAC salt and accepted hashes for one atlas.

    ``hash`` is the main digest. ``hashes`` is an optional extra list. Either
    source is enough. Tumour and Other must use different salts.
    """
    hashes: list[str] = []
    salt = ""
    try:
        cfg = st.secrets[section]
        salt = str(cfg["salt"])
        single = str(cfg.get("hash", "") or "").strip()
        if single:
            hashes.append(single)
        extra = cfg.get("hashes") or []
        hashes.extend(str(item).strip() for item in extra if str(item).strip())
    except (KeyError, FileNotFoundError, TypeError, AttributeError):
        salt = os.environ.get(salt_env) or ""
        single = (os.environ.get(hash_env) or "").strip()
        if single:
            hashes.append(single)
        extra = os.environ.get(hashes_env) or ""
        hashes.extend(item.strip() for item in extra.split(",") if item.strip())
    hashes = list(dict.fromkeys(hashes))
    if salt and hashes:
        return salt, hashes
    return None


def _tumor_auth_secrets() -> tuple[str, list[str]] | None:
    return _auth_secrets("tumor_auth", "TUMOR_AUTH_SALT", "TUMOR_AUTH_HASH", "TUMOR_AUTH_HASHES")


def _other_auth_secrets() -> tuple[str, list[str]] | None:
    return _auth_secrets("other_auth", "OTHER_AUTH_SALT", "OTHER_AUTH_HASH", "OTHER_AUTH_HASHES")


def _verify_password(candidate: str, creds: tuple[str, list[str]] | None) -> bool:
    if not creds:
        return False
    salt, expected = creds
    actual = hmac.new(
        salt.encode("utf-8"),
        candidate.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return any(hmac.compare_digest(actual, item) for item in expected)


def _verify_tumor_password(candidate: str) -> bool:
    return _verify_password(candidate, _tumor_auth_secrets())


def _verify_other_password(candidate: str) -> bool:
    return _verify_password(candidate, _other_auth_secrets())


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
        st.button(
            "Other Atlas →",
            key="go_other_from_mouse",
            on_click=go_to_other,
        )
    elif site == SITE_TUMOR:
        st.button(
            "← Mouse Pituitary Atlas",
            key="go_mouse_site",
            on_click=go_to_mouse,
        )
        st.button(
            "Other Atlas →",
            key="go_other_from_tumor",
            on_click=go_to_other,
        )
    else:
        st.button(
            "← Mouse Pituitary Atlas",
            key="go_mouse_from_other",
            on_click=go_to_mouse,
        )
        st.button(
            "← Human Pituitary Tumour Atlas",
            key="go_tumor_from_other",
            on_click=go_to_tumor,
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
        elif site in {SITE_TUMOR, SITE_OTHER}:
            with st.container(key="epitome_navbar"):
                render_site_switch_button(site)
        st.markdown('<hr style="margin: 0.1rem 0 0.6rem 0;">', unsafe_allow_html=True)


def render_mouse_header(page_map: dict) -> None:
    _render_header(page_map, SITE_MOUSE, _MOUSE_TAGLINE)


def render_tumor_header(page_map: dict | None) -> None:
    _render_header(page_map, SITE_TUMOR, _TUMOR_TAGLINE)


def render_other_header(page_map: dict | None) -> None:
    _render_header(page_map, SITE_OTHER, _OTHER_TAGLINE)


def render_page_footer() -> None:
    """Render the site footer at the end of a page script."""
    render_footer(st.session_state.get("active_site", SITE_MOUSE))


def render_footer(site: str) -> None:
    pit_color = _site_accent(site)
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


def render_other_password_gate() -> None:
    st.markdown("### Password required")
    if _other_auth_secrets() is None:
        st.error(
            "Other atlas access is not configured on this server. "
            "Please contact the epitome team."
        )
    else:
        st.markdown(
            "The Other atlas is restricted. Enter the password to continue, "
            "or return to the mouse pituitary atlas."
        )
        password = st.text_input(
            "Password",
            type="password",
            key="other_password_input",
            placeholder="Enter password",
        )
        if st.button("Unlock Other atlas", key="other_password_submit", type="primary"):
            if _verify_other_password(password):
                st.session_state.other_authenticated = True
                record_site_access(SITE_OTHER, password)
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.button(
        "← Back to Mouse Pituitary Atlas",
        key="other_password_back",
        on_click=go_to_mouse,
    )


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
                record_site_access(SITE_TUMOR, password)
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
            page_with_footer("app_pages/downloads/h5ad_rna.py"),
            title="Dataset Files (h5ad) - RNA",
        ),
        st.Page(
            page_with_footer("app_pages/downloads/h5ad_atac.py"),
            title="Dataset Files (h5ad) - ATAC",
        ),
        st.Page(
            page_with_footer("app_pages/downloads/analysis_data_files.py"),
            title="Analysis Data Files",
        ),
        st.Page(
            page_with_footer("app_pages/downloads/usage_guide.py"),
            title="Single-Cell Object Usage Guide",
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
                page_with_footer("app_pages/tumor/umap_visualisation.py"),
                title="UMAP visualisation",
            ),
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
        "Automated Cell Typing": [
            st.Page(
                page_with_footer("app_pages/tumor/automated_cell_typing.py"),
                title="Automated Cell Typing",
            ),
        ],
        "Individual Datasets": [
            st.Page(
                page_with_footer("app_pages/tumor/individual_datasets.py"),
                title="RNA datasets",
            ),
        ],
        "Downloads": [
            st.Page(
                page_with_footer("app_pages/tumor/downloads.py"),
                title="Downloads",
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
        "How to Cite": [
            st.Page(
                page_with_footer("app_pages/tumor/how_to_cite.py"),
                title="How to Cite",
            ),
        ],
        "Contact": [
            st.Page(
                page_with_footer("app_pages/tumor/contact.py"),
                title="Contact",
            ),
        ],
    }


def build_other_pages() -> dict:
    return {
        "Overview": [
            st.Page(
                page_with_footer("app_pages/other/overview.py"),
                title="Overview",
                default=True,
            ),
        ],
        "Species Atlases": [
            st.Page(
                page_with_footer("app_pages/other/species_atlases.py"),
                title="Species Atlases",
            ),
        ],
        "Phylogeny": [
            st.Page(
                page_with_footer("app_pages/other/phylogeny.py"),
                title="Phylogeny",
            ),
        ],
        "Curation": [
            st.Page(
                page_with_footer("app_pages/other/curation.py"),
                title="Curation",
            ),
        ],
    }
