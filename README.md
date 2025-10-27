# epitome [![DOI](https://zenodo.org/badge/930878390.svg)](https://doi.org/10.5281/zenodo.17154160)

The epitome is a comprehensive web platform for exploring and visualizing the Consensus Pituitary Atlas, a centralized repository of mouse pituitary single-cell sequencing data.


## Citation

To cite Epitome, please reference both:

1. The Consensus Pituitary Atlas pre-print (analysis results and workflow methods):
   
2. The Epitome platform (data visualization and access):
   Kövér, B., Kaufman-Cook, J., Sherwin, O., Vazquez Segoviano, M., Kemkem, Y., Lu, H.-C., & Andoniadou, C. L. (2025). Electronic Pituitary Omics (epitome) platform. Zenodo. https://doi.org/10.5281/zenodo.17154160

## Website
https://epitome.sites.er.kcl.ac.uk/ (Currently under password protection)
   
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

- [x] Build Streamlit website (Dec 2024)
- [x] Create working prototype (Jan 2025)
- [x] Finalise for first release (Sep 2025)
- [ ] Public release featured in pre-print
- [ ] Publish peer-reviewed research article
## Use Cases

This platform serves as a template for atlas creation across different tissues and species. If you're interested in developing a similar resource for your field, please contact us.

## Contact

For questions or collaboration inquiries, please contact the Andoniadou Lab at King's College London.

For epitome specific queries, you can reach out to epitome@kcl.ac.uk

### Developer and lead curator
Bence Kövér

https://bsky.app/profile/bencekover.bsky.social


https://www.linkedin.com/in/ben-kover/

(Email: bence dot kover at kcl dot ac dot uk)")
