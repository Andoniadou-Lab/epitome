# epitome

This repository contains the code for the platform epitome, where researchers can access our recently published Consensus Pituitary Atlas. The analysis code for the Consensus Pituitary Atlas can be accessed in a separate repository.

Framework
The epitome platform has been built using the Streamlit Python framework. Under the hood, we rely on scverse packages to handle single-cell data, and apply various tricks to speed up data access (intermediate .parquet files, polars dataframes etc.).

Accountability
We aim to position epitome in the center of pituitary research, to make a platform where consensus omics results can be displayed. For this, we would like to be completely transparent and accountable, and we hope that open-sourcing our code both for analysis (CPA) and creating the platform (this repo) will help others understand our results.

Generate your own atlas
We believe that this platform is a good starting point for others to build upon, and will allow others to easily deploy their own versions.
Unfortunately, while we aim to share as much of our code as possible, we are aware that there are still hurdles in quickly deploying a project like this. If you are interested in creating a novel atlas for a specific species/tissue, please get in touch.

## Goals tracker
- [x] Start building the Streamlit website - Dec, 2024
- [x] Display working prototype in front of the lab - Jan, 2025
- [ ] Generate the version in the pre-print release - 
- [ ] Publication of peer-reviewed research article
- [ ] Extend platform for future secret atlas project

## Citation
To cite Epitome, please cite both the Consensus Pituitary Atlas pre-print (contains the analysis workflow), as well as the epitome separately (updated source of data display beyond the publication).



### Author
Bence Kövér
https://twitter.com/kover_bence 
https://www.linkedin.com/in/ben-kover/
