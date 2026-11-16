from modules.pta.volcano_page import render_volcano_page

render_volcano_page(
    title="Volcano Plots",
    description=(
        "Visualise single-cell (pseudobulk) differential-expression results, per cell type. "
        "Each plot shows all tested genes. Significance thresholds affect colouring only."
    ),
    collection="sc",
    key_prefix="tumor_sc_volcano",
    version_key="version_select_tumor_sc_volcano",
    analysis_name="Tumor Single-cell Volcano Plot",
)
