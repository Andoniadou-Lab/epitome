from config import Config
from modules.download import TUMOUR_SITE, create_downloads_ui_with_metadata_rna
from modules.pta.page_layout import pta_page_header

selected_version = pta_page_header(
    "Dataset Files (h5ad) - RNA",
    "Download processed single-cell RNA-seq datasets in `.h5ad` format.",
    "tumor_version_select_download_h5ad_rna",
)

create_downloads_ui_with_metadata_rna(Config.BASE_PATH, version=selected_version, site=TUMOUR_SITE)
