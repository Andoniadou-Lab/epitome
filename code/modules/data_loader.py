import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os
from pathlib import Path

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
            "sc_sn_atac",
            "Comp_sex",
            "Name",
            "Author",
            "Normal",
            "SRA_ID",
            "Sorted",
        ]
    ]

    # Print age range for debugging
    print(
        f"Age range in data: {meta_data['Age_numeric'].min()} to {meta_data['Age_numeric'].max()}"
    )

    if hasattr(matrix, "todense"):
        matrix = matrix.todense()

    matrix = np.log10(matrix + 1)

    return matrix, genes, meta_data


def load_curation_data(version="v_0.01"):
    """
    Load curation data
    """
    return pd.read_parquet(f"{BASE_PATH}/data/curation/{version}/cpa.parquet")


def load_annotation_data(version="v_0.01"):
    """
    Load annotation data
    """
    return pd.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/annotation.parquet"
    )


def load_motif_data(version="v_0.01", just_motif_names=False):
    """
    Load ATAC motif data using Polars
    """
    import polars as pl

    data = pl.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/atac_motif_data.parquet"
    )

    if just_motif_names:
        # Get unique motif names and sort them
        return sorted(data.select("motif").unique().to_series().to_list())

    return data


def load_chromvar_data(version="v_0.01"):
    """
    Load ChromVAR data
    """
    chromvar_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/chromvar/{version}/normalized_data.mtx"
    )
    chromvar_meta = pd.read_parquet(
        f"{BASE_PATH}/data/chromvar/{version}/meta_data.parquet"
    )

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

    return proportion_matrix, genes1, rows1, expression_matrix, genes2, rows2


def load_accessibility_data(version="v_0.01"):
    """
    Load accessibility data
    """
    accessibility_matrix = scipy.io.mmread(
        f"{BASE_PATH}/data/accessibility/{version}/normalized_data.mtx"
    )
    accessibility_meta = pd.read_parquet(
        f"{BASE_PATH}/data/accessibility/{version}/atac_meta_data.parquet"
    )

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

    cpdb = pd.read_csv(f"{BASE_PATH}/data/gene_group_annotation/{version}/cpdb.csv")
    # this has two columns gene and category. add one hot encoding
    cpdb = pd.get_dummies(cpdb, columns=["category"])
    # merge with markers such that genes remain even if they are not in cpdb
    cell_typing_markers = cell_typing_markers.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )
    grouping_lineage_markers = grouping_lineage_markers.merge(
        cpdb, how="left", left_on="gene", right_on="gene"
    )

    # from grouping_lineage markers, remove cols log2fc pvalue mean_AvgExpr TF
    grouping_lineage_markers = grouping_lineage_markers.drop(
        columns=["log2fc", "pvalue", "mean_AvgExpr", "TF"]
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
    try:
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
    except Exception as e:
        print(f"Error loading ATAC proportion data: {str(e)}")
        return None, None, None


def load_single_cell_dataset(sra_id, version="v_0.01"):
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
    import anndata

    return anndata.read_h5ad(
        f"{BASE_PATH}/sc_data/datasets/{version}/epitome_h5_files/{sra_id}_processed.h5ad",
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

    # Check if the file exists
    if not os.path.exists(aging_genes_path):
        print(f"Warning: Aging genes file not found at {aging_genes_path}")
        return pd.DataFrame()

    aging_genes_df = pd.read_parquet(aging_genes_path)

    # Clean column names
    aging_genes_df.columns = [
        col.replace("_", " ").title() for col in aging_genes_df.columns
    ]
    print(aging_genes_df.columns)

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
