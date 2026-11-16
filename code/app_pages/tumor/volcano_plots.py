from modules.pta.volcano_page import render_volcano_page

render_volcano_page(
    title="Volcano Plots",
    description=(
        "Visualise bulk tumour differential-expression results. "
        "Pairwise plots show all genes; one-group marker plots show the significant "
        "marker genes only. Significance thresholds affect colouring only."
    ),
    collection="bulk",
    key_prefix="tumor_volcano",
    version_key="version_select_tumor_volcano",
    analysis_name="Tumor Volcano Plot",
)
