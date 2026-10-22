from config import Config
from modules.download import TUMOUR_SITE, create_bulk_data_downloads_ui, tumour_analysis_files
from modules.pta.page_layout import pta_page_header

selected_version = pta_page_header(
    "Analysis Data Files",
    "Download bulk RNA-seq, pseudobulk and curation files of the tumour atlas.",
    "tumor_version_select_download_analysis_files",
)

create_bulk_data_downloads_ui(
    Config.BASE_PATH,
    version=selected_version,
    site=TUMOUR_SITE,
    files_by_category=tumour_analysis_files(Config.BASE_PATH),
    header="Download Analysis Data Files",
)
