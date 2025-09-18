import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import sparse
import scipy.stats as stats
from tqdm.notebook import tqdm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def process_heatmap_data(
    motif_analysis_summary,
    coefs,
    rna_res,
    atac_res,
    mat,
    features,
    columns,
    chosen_names=None,
    top_x_rna=10,
    top_x_atac=10,
    grouping="grouping_1_up",
    multimodal=False,
    AveExpr_threshold=0,
    mean_log2fc_threshold=0,
    fold_enrichment_threshold=0,
):

    # Lists of lists with string representations including "assignments" prefix
    grouping_1 = [
        ["assignmentsStem_cells"],
        [
            "assignmentsMelanotrophs",
            "assignmentsCorticotrophs",
            "assignmentsSomatotrophs",
            "assignmentsLactotrophs",
            "assignmentsThyrotrophs",
            "assignmentsGonadotrophs",
        ],
    ]

    grouping_2 = [
        ["assignmentsGonadotrophs"],
        [
            "assignmentsMelanotrophs",
            "assignmentsCorticotrophs",
            "assignmentsSomatotrophs",
            "assignmentsLactotrophs",
            "assignmentsThyrotrophs",
            "assignmentsStem_cells",
        ],
    ]

    grouping_3 = [
        ["assignmentsMelanotrophs", "assignmentsCorticotrophs"],
        [
            "assignmentsSomatotrophs",
            "assignmentsLactotrophs",
            "assignmentsThyrotrophs",
            "assignmentsGonadotrophs",
            "assignmentsStem_cells",
        ],
    ]

    grouping_4 = [["assignmentsMelanotrophs"], ["assignmentsCorticotrophs"]]

    grouping_5 = [
        ["assignmentsSomatotrophs", "assignmentsLactotrophs", "assignmentsThyrotrophs"],
        [
            "assignmentsGonadotrophs",
            "assignmentsStem_cells",
            "assignmentsMelanotrophs",
            "assignmentsCorticotrophs",
        ],
    ]

    grouping_6 = [
        ["assignmentsLactotrophs"],
        ["assignmentsSomatotrophs", "assignmentsThyrotrophs"],
    ]

    grouping_7 = [
        ["assignmentsSomatotrophs"],
        ["assignmentsLactotrophs", "assignmentsThyrotrophs"],
    ]

    grouping_8 = [
        ["assignmentsThyrotrophs"],
        ["assignmentsLactotrophs", "assignmentsSomatotrophs"],
    ]

    # create column analysis which is grouping + direction
    rna_res["analysis"] = rna_res["grouping"] + "_" + rna_res["direction"]
    # only keep those genes that are either in tfs or in the motif_analysis_summary
    rna_res = rna_res[rna_res["gene"].isin(motif_analysis_summary["gene"])]

    # merge motif_analysis_summary with rna_res on gene, and analysis col
    all_results = motif_analysis_summary.merge(
        rna_res, on=["gene", "analysis"], how="outer"
    )

    #
    all_results = all_results[all_results["analysis"] == grouping]

    all_results["hit_type"] = "unimodal"
    for analysis in all_results["analysis"].unique():
        sub_df = all_results[all_results["analysis"] == analysis].copy()

        for gene in sub_df["gene"].unique():
            if (
                sub_df[sub_df["gene"] == gene]["p.adjust"].values[0] < 0.05
                and sub_df[sub_df["gene"] == gene]["geom_mean_adj_pval"].values[0]
                < 0.05
            ):
                sub_df.loc[sub_df["gene"] == gene, "hit_type"] = "multimodal"
            elif sub_df[sub_df["gene"] == gene]["p.adjust"].values[0] < 0.05:
                sub_df.loc[sub_df["gene"] == gene, "hit_type"] = "atac"
            elif sub_df[sub_df["gene"] == gene]["geom_mean_adj_pval"].values[0] < 0.05:
                sub_df.loc[sub_df["gene"] == gene, "hit_type"] = "rna"
            else:
                sub_df.loc[sub_df["gene"] == gene, "hit_type"] = "not_hit"

        # now modify all_results to include this hit_type
        all_results.loc[all_results["analysis"] == analysis, "hit_type"] = sub_df[
            "hit_type"
        ]

    all_results = all_results[
        (all_results["mean_log2fc"] > mean_log2fc_threshold)
        | (all_results["fold.enrichment"] > fold_enrichment_threshold)
    ]

    # remove those that are not_hit
    all_results = all_results[all_results["hit_type"] != "not_hit"]

    all_results["grouping"] = [
        x.split("_")[0] + "_" + x.split("_")[1] for x in all_results["analysis"]
    ]

    for idx, row in all_results.iterrows():
        gene = row["gene"]
        grouping_name = row["grouping"]

        # Get the actual grouping lists
        current_grouping = eval(grouping_name)

        # Calculate mean for each sublist in the grouping
        if gene in coefs.index:
            for i, sublist in enumerate(current_grouping):
                # Calculate mean across the columns in this sublist for the current gene
                mean_value = coefs.loc[gene, sublist].mean()
                # Store in the appropriate column
                column_name = f"means_group{i+1}"
                all_results.at[idx, column_name] = mean_value

    direction = grouping.split("_")[2]
    if direction == "up":

        all_results["AveExpr"] = all_results["means_group1"]
    elif direction == "down":
        all_results["AveExpr"] = all_results["means_group2"]

    all_results = all_results[all_results["AveExpr"] > AveExpr_threshold]

    atac_res["analysis"] = atac_res["grouping"] + "_" + atac_res["direction"]
    # rename gene to peak, and in entries replace : to -
    atac_res = atac_res.rename(columns={"gene": "peak"})
    atac_res["peak"] = atac_res["peak"].str.replace(":", "-")

    import scipy.io
    import pandas as pd
    import numpy as np

    # Convert to CSR format once (if not already)
    if not scipy.sparse.isspmatrix_csr(mat):
        mat_csr = mat.tocsr()
    else:
        mat_csr = mat

    # Create dictionaries for fast lookups
    feature_to_idx = {features.iloc[i, 0]: i for i in range(len(features))}
    column_to_idx = {columns.iloc[i, 0]: i for i in range(len(columns))}

    plot_results = all_results[all_results["analysis"] == grouping]

    if chosen_names:
        motifs = plot_results[plot_results["gene"].isin(chosen_names)][
            "motif"
        ].values.tolist()
    elif multimodal:
        motifs = plot_results[plot_results["hit_type"] == "multimodal"][
            "motif"
        ].values.tolist()
    else:
        motifs_top_rna = (
            plot_results.sort_values("geom_mean_adj_pval")
            .head(top_x_rna)["motif"]
            .values.tolist()
        )
        motifs_top_atac = (
            plot_results.sort_values("p.adjust")
            .head(top_x_atac)["motif"]
            .values.tolist()
        )
        motifs = list(set(list(motifs_top_rna) + list(motifs_top_atac)))
        # motifs = motifs.dropna()
    sig_peaks = atac_res[atac_res["analysis"] == grouping]

    # Get indices using dictionary lookups (much faster)
    rows_indices = [
        feature_to_idx.get(peak) for peak in sig_peaks["peak"] if peak in feature_to_idx
    ]
    col_indices = [
        column_to_idx.get(motif) for motif in motifs if motif in column_to_idx
    ]

    # Filter out None values (in case some peaks or motifs weren't found)
    rows_indices = [idx for idx in rows_indices if idx is not None]
    col_indices = [idx for idx in col_indices if idx is not None]

    # Extract submatrix with efficient indexing
    sub_matrix = mat_csr[:, col_indices][rows_indices, :]
    return sub_matrix, motifs, plot_results


def analyze_tf_cobinding(sub_matrix, motifs, return_matrix=False):
    """
    Analyze TF co-binding patterns using Fisher's exact test

    Parameters:
    -----------
    sub_matrix : scipy.sparse matrix
        Binary matrix where rows are chromatin positions and columns are TFs
    motifs : list
        List of TF motif names corresponding to columns in sub_matrix
    return_matrix : bool
        If True, also return the fold change matrix for easy visualization

    Returns:
    --------
    pandas.DataFrame
        Results with p-values, odds ratios, and adjusted p-values
    """
    # Get the number of rows (chromatin positions)
    n_positions = sub_matrix.shape[0]

    # Convert to numpy array if not too large, otherwise keep as sparse
    if (
        n_positions * len(motifs) < 10**7
    ):  # Arbitrary threshold, adjust based on your memory
        matrix = sub_matrix.toarray()
    else:
        matrix = sub_matrix

    # Create a DataFrame to store results
    results = []

    # Create a matrix to hold fold change values for easier visualization/clustering
    fold_change_matrix = np.ones((len(motifs), len(motifs)))
    np.fill_diagonal(fold_change_matrix, 5)  # Set diagonal to max value

    # For each pair of TFs (avoiding redundant calculations)
    for i in range(len(motifs)):
        for j in range(len(motifs)):
            # Skip self-comparisons for statistical tests (but keep the matrix values)
            if i == j and not return_matrix:
                continue

            # Get binding vectors for both TFs
            if isinstance(matrix, np.ndarray):
                tf1_binding = matrix[:, i]
                tf2_binding = matrix[:, j]
            else:
                tf1_binding = matrix[:, i].toarray().flatten()
                tf2_binding = matrix[:, j].toarray().flatten()

            # Create the contingency table for Fisher's exact test
            # a: both TFs bind
            # b: TF1 binds, TF2 doesn't
            # c: TF1 doesn't bind, TF2 binds
            # d: Neither binds
            a = np.sum((tf1_binding == 1) & (tf2_binding == 1))
            b = np.sum((tf1_binding == 1) & (tf2_binding == 0))
            c = np.sum((tf1_binding == 0) & (tf2_binding == 1))
            d = np.sum((tf1_binding == 0) & (tf2_binding == 0))

            # Perform Fisher's exact test
            contingency_table = np.array([[a, b], [c, d]])
            odds_ratio, p_value = stats.fisher_exact(
                contingency_table, alternative="two-sided"
            )

            # Calculate observed co-binding rate and expected rate
            observed_rate = a / n_positions
            expected_rate = (a + b) * (a + c) / n_positions**2

            # Calculate fold change (fold enrichment)
            if expected_rate > 0:
                fold_change = observed_rate / expected_rate
            else:
                if i != j:  # Only print warning for non-diagonal elements
                    print(
                        f"Warning: Expected rate is 0 for {motifs[i]} and {motifs[j]}"
                    )
                fold_change = 5  # Cap infinite values

            # Store fold change in matrix (for visualization/clustering)
            fold_change_matrix[i, j] = min(fold_change, 5)  # Cap at 5 for visualization

            # Skip self-comparisons in the results DataFrame (to avoid self-correlations)
            if i == j and not return_matrix:
                continue

            # Store results
            results.append(
                {
                    "TF1": motifs[i],
                    "TF2": motifs[j],
                    "co_binding_count": a,
                    "TF1_only_count": b,
                    "TF2_only_count": c,
                    "neither_count": d,
                    "p_value": p_value,
                    "odds_ratio": odds_ratio,
                    "fold_change": fold_change,
                    "observed_rate": observed_rate,
                    "expected_rate": expected_rate,
                }
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Apply multiple testing correction
    results_df["adjusted_p_value"] = multipletests(
        results_df["p_value"], method="fdr_bh"
    )[1]

    if return_matrix:
        return results_df, fold_change_matrix
    else:
        return results_df



def plot_heatmap(
    results_df,
    motifs,
    all_results=None,
    sig_threshold=0.05,
    cluster=True,
    fold_change_matrix=None,
    font_scale_factor=3,
    use_motif_name=False,
    log10_pval_cap=10,
    fc_cap_log2=3,
):
    """
    Plot a split diagonal heatmap of TF co-binding relationships using seaborn's clustermap
    Lower triangle: log2 fold change (blue-white-red colormap)
    Upper triangle: -log10(adjusted p-value) (white to darkblue colormap)
    Diagonal: black

    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from analyze_tf_cobinding
    motifs : list
        List of TF motif names
    all_results : pandas.DataFrame, optional
        DataFrame containing hit_type annotations for each motif
    sig_threshold : float
        Significance threshold for adjusted p-value
    cluster : bool
        Whether to cluster the TFs based on similarity
    fold_change_matrix : numpy.ndarray, optional
        Pre-computed fold change matrix for improved clustering
    font_scale_factor : float, optional
        Scale factor for font sizes (default: 2.5)
    use_motif_name : bool, optional
        If True, use motif names instead of gene names for labels
    log10_pval_cap : float, optional
        Cap for -log10(p-value) values (default: 10)
    fc_cap_log2 : float, optional
        Cap for log2 fold change values (default: 3)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # Create fold change matrix (convert to log2)
    if fold_change_matrix is not None:
        fc_data = np.log2(fold_change_matrix.copy())
    else:
        # Create a matrix to hold the log2 fold change values
        n_tfs = len(motifs)
        fc_data = np.zeros((n_tfs, n_tfs))  # Start with zeros for log2
        
        # Fill the matrix with log2 fold change values
        for _, row in results_df.iterrows():
            if row["adjusted_p_value"] <= sig_threshold:
                try:
                    i = motifs.index(row["TF1"])
                    j = motifs.index(row["TF2"])
                    fc_data[i, j] = np.log2(max(row["fold_change"], 1e-10))  # Avoid log2(0)
                except ValueError:
                    continue

    # Create p-value matrix
    n_tfs = len(motifs)
    pval_data = np.zeros((n_tfs, n_tfs))
    
    # Fill with p-values
    for _, row in results_df.iterrows():
        if row["adjusted_p_value"] <= sig_threshold:
            try:
                i = motifs.index(row["TF1"])
                j = motifs.index(row["TF2"])
                # Set to -log10(p-value)
                value = -np.log10(max(row["adjusted_p_value"], 1e-100))
                pval_data[i, j] = value
            except ValueError:
                continue

    # Apply capping to log2 fold change
    fc_data = np.clip(fc_data, -fc_cap_log2, fc_cap_log2)
    pval_data = np.clip(pval_data, 0, log10_pval_cap)

    # Create gene name mapping
    gene_names = {}
    if all_results is not None and "gene" in all_results.columns:
        for motif in motifs:
            gene_rows = all_results[all_results["motif"] == motif]
            if len(gene_rows) > 0 and not pd.isna(gene_rows["gene"].values[0]):
                gene_names[motif] = gene_rows["gene"].values[0]
            else:
                gene_names[motif] = motif
    else:
        gene_names = {motif: motif for motif in motifs}

    # Create the combined data matrix for display - use fc_data for clustering
    display_data = fc_data.copy()

    # Convert to DataFrame with proper labels
    if not use_motif_name:
        labels = [gene_names[motif] for motif in motifs]
    else:
        labels = motifs
        
    display_df = pd.DataFrame(display_data, index=labels, columns=labels)

    # Create row colors for annotations if all_results is provided
    row_colors = None
    col_colors = None
    if all_results is not None:
        # Create a Series mapping motifs to hit_types (categorical)
        hit_types = pd.Series(index=display_df.index)
        expr_data = pd.Series(index=display_df.index)

        # Determine which column to use for expression
        expr_col = None
        for col in ["AveExpr"]:
            if col in all_results.columns:
                expr_col = col
                break

        for i, motif in enumerate(motifs):
            gene_name = display_df.index[i]
            hit_type_rows = all_results[all_results["motif"] == motif]
            if len(hit_type_rows) > 0:
                hit_types[gene_name] = (
                    hit_type_rows["hit_type"].values[0]
                    if "hit_type" in hit_type_rows.columns
                    else "unknown"
                )
                if expr_col is not None:
                    expr_data[gene_name] = hit_type_rows[expr_col].values[0]
                else:
                    expr_data[gene_name] = np.nan
            else:
                hit_types[gene_name] = "unknown"
                expr_data[gene_name] = np.nan

        # Create categorical color mapping
        if "hit_type" in all_results.columns:
            lut = {
                "rna": "#1f77b4",
                "atac": "#ff7f0e", 
                "multimodal": "#2ca02c",
            }
            unique_hit_types = hit_types.unique()
            for ht in unique_hit_types:
                if ht not in lut:
                    lut[ht] = "#7f7f7f"
            row_colors = hit_types.map(lut)

        # Create expression colors
        if expr_col is not None and not expr_data.isna().all():
            cmap = plt.cm.YlOrRd
            def map_to_color(x):
                if np.isnan(x):
                    return (0.8, 0.8, 0.8, 1)
                else:
                    max_expr = 10
                    return cmap(min(x, max_expr) / max_expr)
            col_colors = expr_data.map(map_to_color)

    try:
        # Create initial clustermap to determine ordering
        np.fill_diagonal(display_df.values, np.maximum(np.diag(display_df.values), 1e-9))

        clustermap_kwargs = {
            "data": display_df,
            "cmap": "Blues",  # Temporary colormap for clustering
            "figsize": (16, 14),
            "linewidths": 0.5,
            "cbar_pos": None,
        }

        if row_colors is not None:
            clustermap_kwargs["row_colors"] = row_colors
        if col_colors is not None:
            clustermap_kwargs["col_colors"] = col_colors

        if cluster:
            clustermap_kwargs["method"] = "average"
            clustermap_kwargs["metric"] = "correlation"
            clustermap_kwargs["row_cluster"] = True
            clustermap_kwargs["col_cluster"] = True
            clustermap_kwargs["dendrogram_ratio"] = 0
        else:
            clustermap_kwargs["row_cluster"] = False
            clustermap_kwargs["col_cluster"] = False
            clustermap_kwargs["dendrogram_ratio"] = 0

        # Get ordering from initial clustermap
        g_initial = sns.clustermap(**clustermap_kwargs)
        plt.close()

        if cluster:
            row_order = g_initial.dendrogram_row.reordered_ind
            col_order = g_initial.dendrogram_col.reordered_ind
        else:
            row_order = list(range(len(display_df.index)))
            col_order = list(range(len(display_df.columns)))

        # CRITICAL FIX: Ensure row and column orders are the same for symmetric matrix
        if cluster:
            # Use the same ordering for both rows and columns to maintain symmetry
            combined_order = row_order  # or col_order, they should be the same for clustering
            row_order = combined_order
            col_order = combined_order

        ordered_index = [display_df.index[i] for i in row_order]
        ordered_columns = [display_df.columns[i] for i in col_order]

        # Reorder data according to clustering - use the same order for both dimensions
        display_df_ordered = display_df.loc[ordered_index, ordered_columns]
        fc_data_ordered = fc_data[np.ix_(row_order, col_order)]
        pval_data_ordered = pval_data[np.ix_(row_order, col_order)]

        # Reorder row and column colors to match the clustering order
        if row_colors is not None:
            row_colors_ordered = row_colors.loc[ordered_index]
        else:
            row_colors_ordered = None
            
        if col_colors is not None:
            col_colors_ordered = col_colors.loc[ordered_columns]
        else:
            col_colors_ordered = None

        # Calculate dynamic figure size based on number of rows
        n_rows = len(motifs)
        if n_rows <= 15:
            fig_size = max(1500, n_rows * 50) / 100  # Convert pixels to inches (approx)
        else:
            base_size = 1500 / 100  # Base size for first 15 rows
            extra_size = (n_rows - 15) * 50 / 100  # Additional size for rows > 15
            fig_size = base_size + extra_size
        
        fig_size = max(8, fig_size)  # Minimum 8 inches
        
        # Create custom figure with manual heatmap plotting using clustermap structure
        # Use the same approach as the original - create a clustermap first to get the structure
        temp_clustermap_kwargs = {
            "data": display_df_ordered,
            "cmap": "Blues",
            "figsize": (fig_size, fig_size),
            "linewidths": 0.5,
            "cbar_pos": None,
            "row_cluster": False,
            "col_cluster": False,
            "dendrogram_ratio": 0
        }

        # Add row/column colors if available (reordered)
        if row_colors_ordered is not None:
            temp_clustermap_kwargs["row_colors"] = row_colors_ordered
        if col_colors_ordered is not None:
            temp_clustermap_kwargs["col_colors"] = col_colors_ordered

        # Create temporary clustermap to get the axes structure
        g_temp = sns.clustermap(**temp_clustermap_kwargs)
        fig = g_temp.figure
        ax = g_temp.ax_heatmap
        
        # Clear the temporary heatmap
        ax.clear()

        # Create custom colormaps
        # Blue-white-red for log2 fold change (lower triangle)
        bwr_cmap = plt.cm.RdBu_r  # Red-blue colormap (reversed so red=positive, blue=negative)
        # Custom darkblue colormap for p-values (upper triangle)
        darkblue_colors = ['white', '#0000ff']
        darkblue_cmap = LinearSegmentedColormap.from_list('custom_darkblue', darkblue_colors)

        # Plot lower triangle (log2 fold change) with blue-white-red
        masked_fc = np.ma.masked_where(~np.tril(np.ones_like(fc_data_ordered), k=-1).astype(bool), 
                                       fc_data_ordered)
        im1 = ax.imshow(masked_fc, cmap=bwr_cmap, aspect='equal', 
                       vmin=-fc_cap_log2, vmax=fc_cap_log2, interpolation='nearest')

        # Plot upper triangle (p-values) with custom darkblue
        masked_pval = np.ma.masked_where(~np.triu(np.ones_like(pval_data_ordered), k=1).astype(bool), 
                                        pval_data_ordered)
        im2 = ax.imshow(masked_pval, cmap=darkblue_cmap, aspect='equal',
                       vmin=0, vmax=log10_pval_cap, interpolation='nearest')

        # Plot diagonal in black
        diag_mask = np.eye(len(motifs), dtype=bool)
        diag_data = np.where(diag_mask, 1, np.nan)
        im3 = ax.imshow(diag_data, cmap='Greys', aspect='equal', 
                       vmin=0, vmax=1, interpolation='nearest')

        # Set ticks and labels - CENTER THE TICKS ON EACH CELL
        n_items = len(ordered_columns)
        ax.set_xticks(np.arange(n_items))
        ax.set_yticks(np.arange(n_items))
        ax.set_xticklabels(ordered_columns, rotation=90, ha='center')
        ax.set_yticklabels(ordered_index, rotation=0, ha='left', va='center')

        # Calculate font sizes
        base_font_size = 14
        num_genes = len(motifs)
        font_size = base_font_size * font_scale_factor

        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # Add gridlines between cells (offset by 0.5 to create borders)
        ax.set_xticks(np.arange(-0.5, n_items, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_items, 1), minor=True)
        ax.tick_params(which='minor', size=0)  # Hide minor tick marks
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Set title
        title = f"TF Co-binding: log2(FC) (lower) / -log10(p-value) (upper)"
        cap_info = f" (log2FC capped at ±{fc_cap_log2:.1f}, p-val capped at {log10_pval_cap})"
        ax.set_title(title + cap_info, fontsize=font_size * 1.5, pad=100)

        # Move y-axis labels to the right side
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        # Add colorbars using the figure structure
        # log2 Fold change colorbar (blue-white-red)
        cax1 = fig.add_axes([1.1, 0.75, 0.03, 0.2])
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar1.set_label("log2(Fold Change)", fontsize=font_size)
        cbar1.ax.tick_params(labelsize=font_size * 0.8)

        # P-value colorbar (darkblue)
        cax2 = fig.add_axes([1.1, 0.35, 0.03, 0.2])
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar2.set_label("-log10(p-value)", fontsize=font_size)
        cbar2.ax.tick_params(labelsize=font_size * 0.8)

        # Add legends for annotations - same as original
        legend_font_size = max(14, font_size)
        
        # Hit Types legend (categorical row colors)
        if all_results is not None and "hit_type" in all_results.columns and row_colors is not None:
            unique_hit_types = hit_types.unique()
            handles = [plt.Rectangle((0, 0), 1, 1, color=lut[ht]) 
                      for ht in unique_hit_types if ht in lut]
            labels = [ht for ht in unique_hit_types if ht in lut]
            leg1 = fig.legend(handles, labels,
                             title="Hit Types",
                             loc="right",
                             fontsize=legend_font_size,
                             title_fontsize=legend_font_size)

        # Expression colorbar (continuous column colors)
        if col_colors is not None and expr_col is not None:
            cax3 = fig.add_axes([1.1, 0.05, 0.03, 0.2])
            vmin = 0
            vmax = 10  # Adjust based on your data
            norm3 = plt.Normalize(vmin=vmin, vmax=vmax)
            sm3 = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm3)
            sm3.set_array([])
            cbar3 = fig.colorbar(sm3, cax=cax3)
            cbar3.set_label("Avg Expr", fontsize=legend_font_size)
            cbar3.ax.tick_params(labelsize=legend_font_size * 0.8)

        return fig

    except Exception as e:
        print(f"Error creating split heatmap: {e}")
        # Fallback to simple heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(display_df, cmap='viridis', ax=ax, linewidths=0.5)
        ax.set_title("TF Co-binding Heatmap (Fallback)", fontsize=font_size * 1.5)
        return fig