import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os
from pathlib import Path
import polars as pl
from config import Config

BASE_PATH = Config.BASE_PATH


def load_and_transform_data(version="v_0.01"):
    """
    Load and transform the data with log10 transformation
    """
    matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/expression/{version}/normalized_data.mtx"
    )
    genes = pd.read_parquet(f"{BASE_PATH}/data/expression/{version}/genes.parquet")
    if len(genes.columns) == 1:
        genes.columns = [0]
    meta_data = pd.read_parquet(
        f"{BASE_PATH}/data/expression/{version}/meta_data.parquet"
    )
    meta_data = meta_data[
        [
            "new_cell_type",
            "sample",
            "Age_numeric",
            "Modality",
            "Comp_sex",
            "Name",
            "Author",
            "Normal",
            "SRA_ID",
            "Sorted",
        ]
    ]
    #if nan in Name, replace with SRA_ID
    meta_data["Name"] = meta_data["Name"].fillna(meta_data["SRA_ID"])
    meta_data["Comp_sex"] = meta_data["Comp_sex"].astype(str)
    meta_data["Comp_sex"] = meta_data["Comp_sex"].replace({"1": "Male", "0": "Female"})

    # Print age range for debugging
    print(
        f"Age range in data: {meta_data['Age_numeric'].min()} to {meta_data['Age_numeric'].max()}"
    )

    #if hasattr(matrix, "todense"):
    #    matrix = matrix.todense()

    #matrix = np.log10(matrix + 1)

    if scipy.sparse.issparse(matrix):
        matrix = matrix.tocsr()
        matrix.data = np.log10(matrix.data + 1)  # Only transform non-zero entries
    else:
        matrix = np.log10(np.asarray(matrix) + 1)

    return matrix, genes, meta_data


def _normalize_curation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shared Name / Comp_sex cleanup for curation tables."""
    if "SRA_ID" not in df.columns:
        raise ValueError("Curation data must include an SRA_ID column")
    if "Name" not in df.columns:
        df = df.copy()
        df["Name"] = df["SRA_ID"]
    else:
        df = df.copy()
        df["Name"] = df["Name"].fillna(df["SRA_ID"])
    if "Comp_sex" in df.columns:
        df["Comp_sex"] = df["Comp_sex"].astype(str)
        df["Comp_sex"] = df["Comp_sex"].replace({"1": "Male", "0": "Female"})
    return df


def _synthesize_curation_from_expression(version: str) -> pd.DataFrame:
    """Build a sample-level curation table when ``cpa.parquet`` is missing.

    Uses expression ``meta_data`` for the requested version (so new studies are
    not dropped), and enriches overlapping samples from the newest lower
    curation file when available.
    """
    meta_path = Path(f"{BASE_PATH}/data/expression/{version}/meta_data.parquet")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No curation at data/curation/{version}/cpa.parquet and no "
            f"expression meta at {meta_path}"
        )

    meta = pd.read_parquet(meta_path)
    if "SRA_ID" not in meta.columns:
        raise FileNotFoundError(
            f"Expression meta for {version} lacks SRA_ID; cannot synthesize curation"
        )

    sample_cols = [
        c
        for c in (
            "GEO",
            "SRA_ID",
            "Name",
            "Author",
            "Age_numeric",
            "Comp_sex",
            "Normal",
            "Sorted",
            "Modality",
            "Sex",
            "Age",
            "DOI",
            "Conditions",
            "Background",
        )
        if c in meta.columns
    ]
    samples = meta.loc[:, sample_cols].drop_duplicates(subset=["SRA_ID"]).copy()
    samples["SRA_ID"] = samples["SRA_ID"].astype(str)
    if "Name" not in samples.columns:
        samples["Name"] = samples["SRA_ID"]
    else:
        samples["Name"] = samples["Name"].fillna(samples["SRA_ID"])

    # Newest lower curated table for extra columns on overlapping samples.
    from modules.versioning import MOUSE_AVAILABLE_VERSIONS, version_candidates

    base = None
    for candidate in version_candidates(version, MOUSE_AVAILABLE_VERSIONS)[1:]:
        lower_path = Path(f"{BASE_PATH}/data/curation/{candidate}/cpa.parquet")
        if lower_path.exists():
            base = pd.read_parquet(lower_path)
            base["SRA_ID"] = base["SRA_ID"].astype(str)
            break

    if base is None:
        return samples

    base_ids = set(base["SRA_ID"])
    new_rows = samples[~samples["SRA_ID"].isin(base_ids)].copy()
    for col in base.columns:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows.reindex(columns=list(base.columns))

    # Prefer expression meta fields for Author/Name/etc. on shared IDs.
    overlap = samples[samples["SRA_ID"].isin(base_ids)]
    base = base.set_index("SRA_ID", drop=False)
    for col in ("Name", "Author", "Age_numeric", "Comp_sex", "Normal", "Sorted", "Modality"):
        if col in overlap.columns and col in base.columns:
            updates = overlap.set_index("SRA_ID")[col]
            base.loc[updates.index, col] = updates
    base = base.reset_index(drop=True)

    combined = pd.concat(
        [base, new_rows.astype(base.dtypes.to_dict(), errors="ignore")],
        ignore_index=True,
    )
    # Keep Age_numeric parquet-safe (mixed str/float breaks pyarrow).
    if "Age_numeric" in combined.columns:
        combined["Age_numeric"] = (
            combined["Age_numeric"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        )
        combined["Age_numeric"] = pd.to_numeric(combined["Age_numeric"], errors="coerce")
    return combined


def load_curation_data(version="v_0.01"):
    """
    Load curation data for ``version``.

    If ``data/curation/{version}/cpa.parquet`` is missing, synthesize a
    sample-level table from expression meta (enriched with a lower curation
    file when present) instead of silently using an older CPA that omits
    new studies.
    """
    path = Path(f"{BASE_PATH}/data/curation/{version}/cpa.parquet")
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = _synthesize_curation_from_expression(version)
    return _normalize_curation_frame(df)


def load_annotation_data(version="v_0.01"):
    """
    Load annotation data
    """
    table = pl.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/annotation.parquet"
    )
    return table


def load_motif_data(version="v_0.01"):
    """
    Load ATAC motif data using Polars
    """
    

    data = pl.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/atac_motif_data.parquet"
    )

    
    return data


def load_enhancer_data(version="v_0.01"):
    """
    Load ATAC motif data using Polars
    """
    
    data = pl.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/scmultimap_peak_gene_final.parquet"
    )
    #split the 'peak' column into 'chr', 'start', 'end'
    data = data.with_columns([
    pl.col("peak").str.extract(r"^(.*?)-", 1).alias("seqnames"),
    pl.col("peak").str.extract(r"-(\d+)-", 1).cast(pl.Int64).alias("start"),
    pl.col("peak").str.extract(r"-(\d+)$", 1).cast(pl.Int64).alias("end"),
])    
    return data



def load_chromvar_data(version="v_0.01"):
    """
    Load ChromVAR data
    """
    chromvar_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/chromvar/{version}/normalized_data.mtx"
    )
    chromvar_matrix = scipy.sparse.csr_matrix(chromvar_matrix)
    
    chromvar_meta = pd.read_parquet(
        f"{BASE_PATH}/data/chromvar/{version}/meta_data.parquet"
    )

    chromvar_meta["GEO"] = chromvar_meta["sample"]
    chromvar_meta["Comp_sex"] = chromvar_meta["Comp_sex"].astype(str)
    chromvar_meta["Comp_sex"] = chromvar_meta["Comp_sex"].replace({"1": "Male", "0": "Female"})

    # Read features and columns from parquet, but process them like text files
    features_df = pd.read_parquet(
        f"{BASE_PATH}/data/chromvar/{version}/chromvar_features.parquet"
    )
    features = features_df[features_df.columns[0]].tolist()

    columns_df = pd.read_parquet(
        f"{BASE_PATH}/data/chromvar/{version}/chromvar_columns.parquet"
    )
    columns = columns_df[columns_df.columns[0]].tolist()

    return chromvar_matrix, chromvar_meta, features, columns


def load_isoform_data(version="v_0.01"):
    """
    Load isoform-level data
    """
    matrix = scipy.sparse.load_npz(
        f"{BASE_PATH}/data/isoforms/{version}/isoforms_matrix.mtx.npz"
    )

    # Read with no header behavior
    features = pd.read_parquet(
        f"{BASE_PATH}/data/isoforms/{version}/isoforms_features.parquet"
    )
    samples = pd.read_parquet(
        f"{BASE_PATH}/data/isoforms/{version}/isoforms_samples.parquet"
    )

    # Ensure single column files have column name '0' to match header=None behavior
    if len(features.columns) == 1:
        features.columns = [0]
    if len(samples.columns) == 1:
        samples.columns = [0]

    features[["transcript_id", "gene_name"]] = features[0].str.split(
        "_", n=1, expand=True
    )

    for i in range(len(samples)):
        if len(samples.iloc[i, 0].split("_")) > 2:
            samples.iloc[i, 0] = (
                samples.iloc[i, 0].split("_")[0]
                + "_"
                + samples.iloc[i, 0].split("_")[2]
            )

    samples[["SRA_ID", "cell_type"]] = samples[0].str.split("_", n=1, expand=True)

    return matrix, features, samples


def load_dotplot_data(version="v_0.01"):
    """
    Load dot plot data
    """
    proportion_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/dotplot/{version}/matrix2.mtx"
    )
    expression_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/dotplot/{version}/matrix1.mtx"
    )
    # mmread returns COO, which is not row-sliceable.
    proportion_matrix = scipy.sparse.csr_matrix(proportion_matrix)
    expression_matrix = scipy.sparse.csr_matrix(expression_matrix)

    # Read all with no header behavior
    genes1 = pd.read_parquet(
        f"{BASE_PATH}/data/dotplot/{version}/matrix1_genes.parquet"
    )
    genes2 = pd.read_parquet(
        f"{BASE_PATH}/data/dotplot/{version}/matrix2_genes.parquet"
    )
    rows1 = pd.read_parquet(f"{BASE_PATH}/data/dotplot/{version}/matrix1_rows.parquet")
    rows2 = pd.read_parquet(f"{BASE_PATH}/data/dotplot/{version}/matrix2_rows.parquet")

    # Ensure single column files have column name '0' to match header=None behavior
    for df in [genes1, genes2, rows1, rows2]:
        if len(df.columns) == 1:
            df.columns = [0]

    if len(rows1) != proportion_matrix.shape[0]:
        min_len = min(len(rows1), proportion_matrix.shape[0])
        rows1 = rows1.iloc[:min_len]
        proportion_matrix = proportion_matrix[:min_len, :]

    if len(rows2) != expression_matrix.shape[0]:
        min_len = min(len(rows2), expression_matrix.shape[0])
        rows2 = rows2.iloc[:min_len]
        expression_matrix = expression_matrix[:min_len, :]

    if len(genes1) != proportion_matrix.shape[1]:
        min_len = min(len(genes1), proportion_matrix.shape[1])
        genes1 = genes1.iloc[:min_len]
        proportion_matrix = proportion_matrix[:, :min_len]

    if len(genes2) != expression_matrix.shape[1]:
        min_len = min(len(genes2), expression_matrix.shape[1])
        genes2 = genes2.iloc[:min_len]
        expression_matrix = expression_matrix[:, :min_len]

    return proportion_matrix, genes1, rows1, expression_matrix, genes2, rows2
def load_accessibility_data(version="v_0.01"):
    """
    Load accessibility data
    """
    accessibility_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/accessibility/{version}/normalized_data.mtx"
    )
    accessibility_matrix = scipy.sparse.csr_matrix(accessibility_matrix)

    accessibility_meta = pd.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/atac_meta_data.parquet"
    )

    accessibility_meta["GEO"] = accessibility_meta["sample"]

    accessibility_meta["Comp_sex"] = accessibility_meta["Comp_sex"].astype(str)
    accessibility_meta["Comp_sex"] = accessibility_meta["Comp_sex"].replace({"1": "Male", "0": "Female"})

    # Read features and columns from parquet but process them like text files
    features_df = pd.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/accessibility_features.parquet"
    )
    features = features_df[features_df.columns[0]].tolist()

    columns_df = pd.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/accessibility_columns.parquet"
    )
    columns = columns_df[columns_df.columns[0]].tolist()

    return accessibility_matrix, accessibility_meta, features, columns


def gene_group_annotation_path(version="v_0.01"):
    """Path to ``cpdb`` gene categories for ``version``, or the newest lower one.

    These categories (TF / ligand / receptor / metabolism) are a gene-level
    reference rather than per-release data, so a version without its own copy
    reuses an older one instead of failing the whole table.
    """
    from modules.versioning import version_candidates

    for candidate in version_candidates(version):
        for suffix in (".csv", ".parquet"):
            path = Path(
                f"{BASE_PATH}/data/gene_group_annotation/{candidate}/cpdb{suffix}"
            )
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"No gene_group_annotation/cpdb file for {version} or any lower version"
    )


def load_gene_group_annotation(version="v_0.01"):
    """Gene category table, one-hot encoded, resolved with version fallback."""
    path = gene_group_annotation_path(version)
    cpdb = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return pd.get_dummies(cpdb, columns=["category"])


def load_gene_curation(version="v_0.01"):
    path = gene_group_annotation_path(version)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_marker_data(version="v_0.01"):
    """
    Load marker data from both cell typing and grouping/lineage files
    """
    cell_typing_markers = pd.read_parquet(
        f"{BASE_PATH}/data/markers/{version}/cell_typing_markers.parquet"
    )
    grouping_lineage_markers = pd.read_parquet(
        f"{BASE_PATH}/data/markers/{version}/grouping_lineage_markers.parquet"
    )

    cpdb = load_gene_group_annotation(version)
    # merge with markers such that genes remain even if they are not in cpdb
    cell_typing_markers = cell_typing_markers.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )
    grouping_lineage_markers = grouping_lineage_markers.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )
    #turn nans into 0
    cell_typing_markers = cell_typing_markers.fillna(0)

    cell_typing_markers["category_TF"] = cell_typing_markers["category_TF"].astype(int).astype(bool)
    cell_typing_markers["category_ligand"] = cell_typing_markers["category_ligand"].astype(int).astype(bool)
    cell_typing_markers["category_receptor"] = cell_typing_markers["category_receptor"].astype(int).astype(bool)
    cell_typing_markers["category_metabolism"] = cell_typing_markers["category_metabolism"].astype(int).astype(bool)
    
    grouping_lineage_markers = grouping_lineage_markers.fillna(0)
    #convert True to 1 and False to 0
    grouping_lineage_markers["category_TF"] = grouping_lineage_markers["category_TF"].astype(int).astype(bool)
    grouping_lineage_markers["category_ligand"] = grouping_lineage_markers["category_ligand"].astype(int).astype(bool)
    grouping_lineage_markers["category_receptor"] = grouping_lineage_markers["category_receptor"].astype(int).astype(bool)
    grouping_lineage_markers["category_metabolism"] = grouping_lineage_markers["category_metabolism"].astype(int).astype(bool)
    
    #print head of grouping_lineage_markers
    print(grouping_lineage_markers.head())

    # from grouping_lineage markers, remove cols log2fc pvalue mean_AvgExpr TF
    grouping_lineage_markers = grouping_lineage_markers.drop(
        columns=["log2fc", "pvalue", "mean_AvgExpr", "TF"]
    )

    # if any gene grouping duplicates exist, remove
    cell_typing_markers = cell_typing_markers.drop_duplicates()
    grouping_lineage_markers = grouping_lineage_markers.drop_duplicates()

    return cell_typing_markers, grouping_lineage_markers



def load_marker_data_atac(version="v_0.01"):
    """
    Load marker data from both cell typing and grouping/lineage files
    """
    cell_typing_markers = pd.read_parquet(
        f"{BASE_PATH}/data/markers/{version}/cell_typing_markers_atac.parquet"
    )
    grouping_lineage_markers = pd.read_parquet(
        f"{BASE_PATH}/data/markers/{version}/atac_grouping_lineage_markers.parquet"
    )
    

    # if any gene grouping duplicates exist, remove
    cell_typing_markers = cell_typing_markers.drop_duplicates()
    grouping_lineage_markers = grouping_lineage_markers.drop_duplicates()

    return cell_typing_markers, grouping_lineage_markers





def load_proportion_data(version="v_0.01"):
    """
    Load cell type proportion data
    """
    abundance_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/cell_proportion/{version}/abundance.mtx"
    )
    abundance_rows = pd.read_csv(
        f"{BASE_PATH}/data/cell_proportion/{version}/abundance_rows.tsv",
        sep="\t",
        header=None,
    )
    abundance_cols = pd.read_csv(
        f"{BASE_PATH}/data/cell_proportion/{version}/abundance_cols.tsv",
        sep="\t",
        header=None,
    )

    return abundance_matrix, abundance_rows, abundance_cols


def load_atac_proportion_data(version="v_0.01"):
    """
    Load ATAC cell type proportion data
    """
    # Errors propagate so callers can fall back to a lower version; swallowing
    # them here made a missing version look like a successful load.
    abundance_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/cell_proportion_atac/{version}/abundance.mtx"
    )
    abundance_rows = pd.read_csv(
        f"{BASE_PATH}/data/cell_proportion_atac/{version}/abundance_rows.tsv",
        sep="\t",
        header=None,
    )
    abundance_cols = pd.read_csv(
        f"{BASE_PATH}/data/cell_proportion_atac/{version}/abundance_cols.tsv",
        sep="\t",
        header=None,
    )

    return abundance_matrix, abundance_rows, abundance_cols


def load_single_cell_dataset(sra_id, version="v_0.01",rna_atac="rna"):
    """
    Load a single-cell H5AD dataset

    Parameters:
    -----------
    filepath : str
        Path to the .h5ad file

    Returns:
    --------
    anndata.AnnData
        Loaded single-cell dataset
    """
    import scanpy as sc
    if rna_atac == "rna":
        # Load RNA data
        print(f"Loading: {BASE_PATH}/sc_data/datasets/{version}/epitome_h5_files/{sra_id}_processed.h5ad")
        return sc.read(
            f"{BASE_PATH}/sc_data/datasets/{version}/epitome_h5_files/{sra_id}_processed.h5ad",
            backed="r",
        )
    elif rna_atac == "atac":
        # Load ATAC data
        print(f"Loading: {BASE_PATH}/sc_atac_data/datasets/{version}/epitome_h5_files/{sra_id}.h5ad")
        return sc.read(
            f"{BASE_PATH}/sc_atac_data/datasets/{version}/epitome_h5_files/{sra_id}.h5ad",
            backed="r",
        )

def load_aging_genes(version="v_0.01"):
    """
    Load aging genes data

    Parameters:
    -----------
    version : str, optional
        Version of the dataset (default is "v_0.01")

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing aging genes information
    """
    aging_genes_path = f"{BASE_PATH}/data/aging/{version}/aging_genes.parquet"

    # Raise (rather than return empty) so callers can fall back to a lower version.
    if not os.path.exists(aging_genes_path):
        raise FileNotFoundError(f"Aging genes file not found at {aging_genes_path}")

    aging_genes_df = pd.read_parquet(aging_genes_path)

    # Clean column names
    aging_genes_df.columns = [
        col.replace("_", " ").title() for col in aging_genes_df.columns
    ]
    print(aging_genes_df.columns)

    aging_genes_df  = aging_genes_df.drop(
            columns=[
                "Unnamed: 0"])
    
    #rename genes to gene
    aging_genes_df = aging_genes_df.rename(columns={"Genes": "gene"})


    
    cpdb = load_gene_group_annotation(version)
    # merge with markers such that genes remain even if they are not in cpdb
    aging_genes_df = aging_genes_df.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )
    #turn nans into 0
    aging_genes_df = aging_genes_df.fillna(0)
    #convert True to 1 and False to 0
    aging_genes_df["category_TF"] = aging_genes_df["category_TF"].astype(int).astype(bool)
    aging_genes_df["category_ligand"] = aging_genes_df["category_ligand"].astype(int).astype(bool)
    aging_genes_df["category_receptor"] = aging_genes_df["category_receptor"].astype(int).astype(bool)
    aging_genes_df["category_metabolism"] = aging_genes_df["category_metabolism"].astype(int).astype(bool)

    #print head of grouping_lineage_markers
    print(aging_genes_df.head())

    aging_genes_df = aging_genes_df.drop_duplicates()

    return aging_genes_df


def load_ligand_receptor_data(version="v_0.01"):
    """
    Load ligand-receptor interaction data
    """
    import pandas as pd

    liana_df = pd.read_parquet(
        f"{BASE_PATH}/data/lig_rec/{version}/liana_consensus.parquet"
    )

    # Process the data
    liana_df["ligand_complex"] = liana_df["gene"].str.split("___").str[0]
    liana_df["receptor_complex"] = liana_df["gene"].str.split("___").str[1]

    liana_df["target"] = liana_df["gene"].str.split("___").str[2]
    liana_df["source"] = liana_df["gene"].str.split("___").str[3]

    # remove those interactions that exist twice, both as source and target and target and source.
    # Create a sorted interaction pair column to identify duplicates
    liana_df["interaction"] = liana_df.apply(
        lambda row: tuple(
            sorted(
                [
                    row["ligand_complex"],
                    row["receptor_complex"],
                    row["source"],
                    row["target"],
                ]
            )
        ),
        axis=1,
    )
    # sort by ligand_complex
    liana_df = liana_df.sort_values(
        ["ligand_complex", "corrected_score_y"], ascending=[True, True]
    )
    # Drop duplicate interactions
    liana_df = liana_df.drop_duplicates(subset=["interaction"], keep="first")

    # Drop the helper column
    liana_df = liana_df.drop(columns=["interaction"])

    # Rename and filter scores
    liana_df = liana_df.rename(
        columns={
            "corrected_score_x": "specificity_rank",
            "corrected_score_y": "magnitude_rank",
        }
    )

    # remove duplicates
    liana_df = liana_df.drop_duplicates()

    return liana_df


def load_enrichment_results(version="v_0.01"):
    """
    Load enrichment results for all groupings and concatenate them into a single DataFrame

    Parameters:
    -----------
    version : str, optional
        Version of the dataset (default is "v_0.01")

    Returns:
    --------
    pandas.DataFrame
        Concatenated DataFrame containing all enrichment results
    """
    import pandas as pd
    import os

    # Initialize empty list to store all dataframes
    all_dfs = []

    # Loop through groupings 1-8
    for grouping in range(1, 9):
        for direction in ["up", "down"]:
            file_path = os.path.join(
                BASE_PATH,
                "data",
                "accessibility",
                version,
                f"enrichment_results_grouping_{grouping}_{direction}.csv",
            )

            try:
                # Read CSV file
                df = pd.read_csv(file_path)

                # Add columns to identify the source
                df["Grouping"] = f"Grouping {grouping}"
                df["Direction"] = direction.upper()
                # make these the first two cols
                cols = df.columns.tolist()
                cols = cols[-2:] + cols[:-2]
                df = df[cols]
                # Append to list
                all_dfs.append(df)

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    # Concatenate all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Clean and format the DataFrame
        # Round numeric columns to 3 decimal places
        numeric_columns = combined_df.select_dtypes(include=["float64"]).columns
        # combined_df[numeric_columns] = combined_df[numeric_columns].round(3)

        # Sort by adjusted p-value and grouping
        combined_df = combined_df.sort_values(["Grouping", "Direction", "p.adjust"])

        return combined_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no files were loaded


# load motif_genes


def load_motif_genes(version="v_0.01"):
    import pandas as pd

    df = pd.read_csv(f"{BASE_PATH}/data/accessibility/{version}/annotation.csv")
    # return entries from col gene_name
    return df["gene_name"].unique()


# load heatmap_data
def load_heatmap_data(version="v_0.01"):
    motif_analysis_summary = pd.read_csv(
        f"{BASE_PATH}/data/heatmap/{version}/all_motif_results.csv"
    )

    # turn each motif.name into capitalized (first letter)
    motif_analysis_summary["motif.name"] = motif_analysis_summary[
        "motif.name"
    ].str.capitalize()
    # if it has . or :, remove it
    motif_analysis_summary["motif.name"] = motif_analysis_summary[
        "motif.name"
    ].str.replace(".", "")
    motif_analysis_summary["motif.name"] = motif_analysis_summary[
        "motif.name"
    ].str.replace(":", "")

    # for each motif, analysis pair, keep the one with the highest fold.enrichment
    motif_analysis_summary = motif_analysis_summary.sort_values(
        "fold.enrichment", ascending=False
    ).drop_duplicates(["motif.name", "analysis"])

    coefs = pd.read_csv(f"{BASE_PATH}/data/heatmap/{version}/coef.csv", index_col=0)
    rna_res = pd.read_csv(
        f"{BASE_PATH}/data/heatmap/{version}/rna_grouping_lineage_markers.csv"
    )
    atac_res = pd.read_csv(
        f"{BASE_PATH}/data/heatmap/{version}/atac_grouping_lineage_markers.csv"
    )
    mat = scipy.io.mmread(f"{BASE_PATH}/data/heatmap/{version}/motif_matrix.mtx")
    features = pd.read_table(
        f"{BASE_PATH}/data/heatmap/{version}/motif_matrix_rows.txt", header=None
    )
    columns = pd.read_table(
        f"{BASE_PATH}/data/heatmap/{version}/motif_matrix_cols.txt", header=None
    )

    # keep those rows where at least one column has 1
    coefs = coefs[(coefs > 1).any(axis=1)]
    genes_to_keep = coefs.index

    # only keep those motifs where motif.name is in tfs, but first print those motif.name that are not in tfs
    motif_analysis_summary = motif_analysis_summary[
        motif_analysis_summary["motif.name"].isin(genes_to_keep)
    ]
    # rename motif.name to gene
    motif_analysis_summary = motif_analysis_summary.rename(
        columns={"motif.name": "gene"}
    )

    return motif_analysis_summary, coefs, rna_res, atac_res, mat, features, columns



def load_sex_dim_data(version):
    sex_dim_data = pd.read_parquet(f'{BASE_PATH}/data/sex_dimorphism/{version}/sexually_dimorphic_genes.parquet')

    for col in ("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B", "z.std", "occurs"):
        if col in sex_dim_data.columns:
            sex_dim_data[col] = pd.to_numeric(sex_dim_data[col], errors="coerce")

    #add col -log10_pval from adj.P.Val
    sex_dim_data['-log10_pval'] = -1 * np.log10(sex_dim_data['adj.P.Val'].clip(lower=1e-300))
    #remove col P.Value
    sex_dim_data = sex_dim_data.drop(columns=['P.Value', 'adj.P.Val'])

    cpdb = load_gene_group_annotation(version)
    # merge with markers such that genes remain even if they are not in cpdb
    sex_dim_data = sex_dim_data.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )
    #turn nans into 0
    sex_dim_data = sex_dim_data.fillna(0)
    #convert True to 1 and False to 0
    sex_dim_data["category_TF"] = sex_dim_data["category_TF"].astype(int).astype(bool)
    sex_dim_data["category_ligand"] = sex_dim_data["category_ligand"].astype(int).astype(bool)
    sex_dim_data["category_receptor"] = sex_dim_data["category_receptor"].astype(int).astype(bool)
    sex_dim_data["category_metabolism"] = sex_dim_data["category_metabolism"].astype(int).astype(bool)

    #print head of grouping_lineage_markers
    print(sex_dim_data.head())

    sex_dim_data = sex_dim_data.drop_duplicates()

    return sex_dim_data




def check_file_exists(filepath):
    """Check if a file exists and print detailed info if it doesn't"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            print(f"Directory does not exist: {dirpath}")
        else:
            print(f"Directory exists but file is missing")
            print("Files in directory:")
            for f in os.listdir(dirpath):
                print(f"  - {f}")
        return False
    return True


def verify_all_paths(version="v_0.01"):
    """Verify all data paths exist and print their status"""
    paths = {
        "Expression Matrix": f"{BASE_PATH}/data/expression/{version}/normalized_data.mtx",
        "Genes": f"{BASE_PATH}/data/expression/{version}/genes.parquet",
        "Metadata": f"{BASE_PATH}/data/expression/{version}/meta_data.parquet",
        "Curation": f"{BASE_PATH}/data/curation/{version}/cpa.parquet",
        "Annotation": f"{BASE_PATH}/data/accessibility/{version}/annotation.parquet",
        "ATAC Motif": f"{BASE_PATH}/data/accessibility/{version}/atac_motif_data.parquet",
    }

    print("\nVerifying data paths:")
    all_exist = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False

    return all_exist


if __name__ == "__main__":
    print("Testing data loader...")
    print(f"Base path: {BASE_PATH}")
    verify_all_paths()