import streamlit as st

from modules.citations import pre_print_citation, print_citation
from modules.test_health import render_test_health_bar

st.header("Release Notes")
st.markdown("Details of features and datasets included in each released version of the epitome.")
render_test_health_bar()
st.info(
"v_0.02: Second release of epitome, associated with the published manuscript, and including all mouse pituitary datasets published before Feb, 2026.\n\n"
"- Added new datasets from Guo et al (2025), Jin et al. (2025), Sochodolsky et al. (2026). Statistical and normalisation procedures have been updated.\n"
"- Metadata has been corrected for some publications.\n"
"- New Cell Type Model and Doublet Model available. These have incremental, very slight improvements over the pre-print version.\n"
"- Entirely new features have not been added, but existing features have been further documented (e.g. Automated annotation workflow) and optimised.\n"
"- We will support accessing data from v_0.01 for the sake of reproducibility, but with this release, we recommend using v_0.02 for the most up-to-date data.\n"
f"\nFor more information, see Methods in our publication in Cell Reports {print_citation} \n\n"
)


st.info(
"v_0.01: First release of the epitome, including all mouse pituitary datasets published before October, 2025.\n\n"
"Transcriptome analysis:\n"
"- Expression Box Plots and UMAPs: Visualize gene expression across cell types with filtering options\n"
"- Age Correlation: Analyze expression-age relationships with statistical metrics\n"
"- Isoforms: Explore transcript-level expression with ensembl annotations\n"
"- Dot Plots: Compare expression patterns showing magnitude and prevalence\n"
"- Cell Type Distribution: Examine proportions with sex and age grouping\n"
"- Gene-Gene Relationships: Analyze correlations with cell type specificity\n"
"- Ligand-Receptor Interactions: Identify communication pathways\n"
"- Sex-biased Genes and Marker Browser: Access comprehensive tables\n\n"
"Chromatin analysis:\n"
"- Accessibility Distribution: Visualize chromatin accessibility patterns\n"
"- Interactive Genome Browser: View genomic regions with gene annotations and motifs\n"
"- Motif Enrichment (ChromVAR): Analyze TF binding with enrichment scores\n"
"- Cell Type Distribution: View ATAC-seq based population proportions\n\n"
"Multimodal analysis:\n"
"- TF Heatmaps: Visualize TF co-binding with RNA and ATAC evidence\n"
"- Lineage-specific Factors: Explore TFs driving cell fate decisions\n\n"
"Individual datasets:\n"
"- Interactive RNA and ATAC UMAPs: Explore single datasets with QC reports\n\n"
"Automated cell type and doublet annotation:\n"
"- Access to our Cell Type Model in the browser\n"
"- Access to our Doublet Model in the browser\n"
"- Downloadable results and visualisation of user uploaded data, without any coding\n\n"
"Data access:\n"
"- Downloadable Cell Type Model: Use our model in your own analysis\n"
"- Downloadable H5AD Files: Access pre-processed data with metadata\n"
"- Analysis Data Files: Download matrices and processed data\n"
"- Comprehensive Curation: Browse detailed metadata\n"
"- Usage Guides: Instructions for Python (Scanpy) and R (Seurat)\n"
f"\nFor more information, see Methods in our pre-print on bioRxiv {pre_print_citation}."
"\nThe codebase for this release is found on [GitHub](https://github.com/Andoniadou-Lab/epitome)"
)
