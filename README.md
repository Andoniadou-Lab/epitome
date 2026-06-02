# epitome [![DOI](https://zenodo.org/badge/930878390.svg)](https://doi.org/10.5281/zenodo.17154160)

The epitome is a comprehensive web platform for exploring and visualizing the Consensus Pituitary Atlas, a centralized repository of mouse pituitary single-cell sequencing data.


## Citation

To cite Epitome, please reference both:

1. The Consensus Pituitary Atlas publication in Cell Reports (analysis results and workflow methods):
  Kövér, B., Willis, T.L., Sherwin, O., Kaufman-Cook, J., Kemkem, Y., Segoviano, M.V., Lodge, E.J., Zamojski, M., Mendelev, N., Zhang, Z., et al. (2026). Consensus Pituitary Atlas, a scalable resource for annotation, novel marker discovery, and analyses in mouse pituitary gland research. Cell Rep. 45. https://doi.org/10.1016/j.celrep.2026.117407. 
  
3. The Epitome platform (data visualization and access):
   Kövér, B., Kaufman-Cook, J., Sherwin, O., Vazquez Segoviano, M., Kemkem, Y., Lu, H.-C., & Andoniadou, C. L. (2025). Electronic Pituitary Omics (epitome) platform. Zenodo. https://doi.org/10.5281/zenodo.17154160

## Website
epitome-atlas.com
   
## Features

- **Interactive Visualizations**: Explore gene expression patterns, chromatin accessibility, and cell type distributions across multiple datasets
- **Multi-modal Analysis**: Examine RNA-seq and ATAC-seq data in an integrated environment
- **Cell Type Reference**: Access standardized cell type definitions and marker genes
- **Dataset Browser**: View individual datasets with rich metadata
- **Data Download**: Export processed data for further analysis

## Technical Architecture

- Built on the **Streamlit** framework for interactive data visualization
- Leverages **scverse** packages for single-cell data processing
- Optimized for performance with **parquet** files and **polars** dataframes
- Modular design with separate components for different analysis types

## Repository Structure

- `epitome.py`: Main application entry point
- `/modules`: Core functionality modules
  - `data_loader.py`: Data loading and caching
  - `expression.py`, `accessibility.py`, etc.: Visualization modules
  - `utils.py`: Shared utility functions
  - `download.py`: Data export functionality

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure data paths in `config.py`
4. Run the application: `streamlit run epitome.py`

## Roadmap
- [x] Start curation, and build data processing workflow - July, 2024
- [x] Arrive at intermediate results, start building epitome - Dec, 2024
- [x] Finalise results using all datasets published to date - Oct, 2025
- [x] Release pre-print - Oct, 2025
- [x] Release final peer-reviewed publication - May, 2026

## Use Cases

This platform serves as a template for atlas creation across different tissues and species. If you're interested in developing a similar resource for your field, please contact us.

## Contact

For questions or collaboration inquiries, please contact the Andoniadou Lab at King's College London.

For epitome specific queries, you can reach out to epitome@kcl.ac.uk


## Acknowledgments

This work was supported by the Wellcome Trust Advanced Therapies for Regenerative Medicine PhD Programme (218461/Z/19/Z). Special thanks to the Andoniadou Lab at King's College London and all contributors to the Consensus Pituitary Atlas.

---

**Developer and Lead Curator**: Bence Kövér  
**Lab**: Andoniadou Lab, King's College London  
**Contact**: bence.kover@kcl.ac.uk

https://bsky.app/profile/bencekover.bsky.social
https://www.linkedin.com/in/bence-kover/
