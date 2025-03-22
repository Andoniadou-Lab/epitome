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


def process_heatmap_data(motif_analysis_summary, coefs, rna_res, atac_res, mat, features, columns,
                         chosen_names=None, top_x_rna=10, top_x_atac=10,
                           grouping='grouping_1_up',multimodal=False, AveExpr_threshold=0,
                           mean_log2fc_threshold=0, fold_enrichment_threshold=0
                           ):


    # Lists of lists with string representations including "assignments" prefix
    grouping_1 = [["assignmentsStem_cells"], ["assignmentsMelanotrophs", "assignmentsCorticotrophs", 
                "assignmentsSomatotrophs", "assignmentsLactotrophs", "assignmentsThyrotrophs",
                "assignmentsGonadotrophs"]]

    grouping_2 = [["assignmentsGonadotrophs"], ["assignmentsMelanotrophs", "assignmentsCorticotrophs",
                "assignmentsSomatotrophs", "assignmentsLactotrophs", "assignmentsThyrotrophs",
                "assignmentsStem_cells"]]

    grouping_3 = [["assignmentsMelanotrophs", "assignmentsCorticotrophs"], [
                "assignmentsSomatotrophs", "assignmentsLactotrophs", "assignmentsThyrotrophs",
                "assignmentsGonadotrophs", "assignmentsStem_cells"]]

    grouping_4 = [["assignmentsMelanotrophs"], ["assignmentsCorticotrophs"]]

    grouping_5 = [["assignmentsSomatotrophs", "assignmentsLactotrophs", "assignmentsThyrotrophs"], [
                "assignmentsGonadotrophs", "assignmentsStem_cells", "assignmentsMelanotrophs",
                "assignmentsCorticotrophs"]]

    grouping_6 = [["assignmentsLactotrophs"], ["assignmentsSomatotrophs", "assignmentsThyrotrophs"]]

    grouping_7 = [["assignmentsSomatotrophs"], ["assignmentsLactotrophs", "assignmentsThyrotrophs"]]

    grouping_8 = [["assignmentsThyrotrophs"], ["assignmentsLactotrophs", "assignmentsSomatotrophs"]]


    #create column analysis which is grouping + direction
    rna_res['analysis'] = rna_res['grouping'] + '_' + rna_res['direction']
    #only keep those genes that are either in tfs or in the motif_analysis_summary
    rna_res = rna_res[rna_res['gene'].isin(motif_analysis_summary['gene'])]

    #merge motif_analysis_summary with rna_res on gene, and analysis col
    all_results = motif_analysis_summary.merge(rna_res, on=['gene', 'analysis'], how='outer')
    
    #
    all_results = all_results[all_results['analysis']==grouping]
                              
    all_results['hit_type'] = 'unimodal'
    for analysis in all_results['analysis'].unique():
        sub_df = all_results[all_results['analysis'] == analysis].copy()

        for gene in sub_df['gene'].unique():
            if sub_df[sub_df['gene'] == gene]['p.adjust'].values[0] < 0.05 and sub_df[sub_df['gene'] == gene]['geom_mean_adj_pval'].values[0] < 0.05:
                sub_df.loc[sub_df['gene'] == gene, 'hit_type'] = 'multimodal'
            elif sub_df[sub_df['gene'] == gene]['p.adjust'].values[0] < 0.05:
                sub_df.loc[sub_df['gene'] == gene, 'hit_type'] = 'atac'
            elif sub_df[sub_df['gene'] == gene]['geom_mean_adj_pval'].values[0] < 0.05:
                sub_df.loc[sub_df['gene'] == gene, 'hit_type'] = 'rna'
            else:
                sub_df.loc[sub_df['gene'] == gene, 'hit_type'] = 'not_hit'
        

        #now modify all_results to include this hit_type
        all_results.loc[all_results['analysis'] == analysis, 'hit_type'] = sub_df['hit_type']
    
    
    all_results = all_results[(all_results['mean_log2fc'] > mean_log2fc_threshold) | (all_results['fold.enrichment'] > fold_enrichment_threshold)]

    #remove those that are not_hit
    all_results = all_results[all_results['hit_type'] != 'not_hit']
    
    
    all_results['grouping'] = [x.split('_')[0] + '_' + x.split('_')[1] for x in all_results['analysis']]

    for idx, row in all_results.iterrows():
        gene = row['gene']
        grouping_name = row['grouping']
        
        # Get the actual grouping lists
        current_grouping = eval(grouping_name)
        
        # Calculate mean for each sublist in the grouping
        if gene in coefs.index:
            for i, sublist in enumerate(current_grouping):
                # Calculate mean across the columns in this sublist for the current gene
                mean_value = coefs.loc[gene, sublist].mean()
                # Store in the appropriate column
                column_name = f'means_group{i+1}'
                all_results.at[idx, column_name] = mean_value
    

    direction = grouping.split('_')[2]
    if direction == 'up':
        
        all_results["AveExpr"] = all_results["means_group1"]
    elif direction == 'down':
        all_results["AveExpr"] = all_results["means_group2"]

    all_results = all_results[all_results['AveExpr'] > AveExpr_threshold]


    atac_res["analysis"] = atac_res["grouping"] + "_" + atac_res["direction"]
    #rename gene to peak, and in entries replace : to -
    atac_res = atac_res.rename(columns={'gene':'peak'})
    atac_res['peak'] = atac_res['peak'].str.replace(':', '-')

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

    plot_results = all_results[all_results['analysis'] == grouping]

    
    if chosen_names:
        motifs = plot_results[plot_results['gene'].isin(chosen_names)]['motif'].values.tolist()
    elif multimodal:
        motifs = plot_results[plot_results['hit_type'] == 'multimodal']['motif'].values.tolist()
    else:
        motifs_top_rna = plot_results.sort_values('geom_mean_adj_pval').head(top_x_rna)['motif'].values.tolist()
        motifs_top_atac = plot_results.sort_values("p.adjust").head(top_x_atac)['motif'].values.tolist()
        motifs = list(set(list(motifs_top_rna) + list(motifs_top_atac)))
        #motifs = motifs.dropna()
    sig_peaks = atac_res[atac_res['analysis'] == grouping]

    # Get indices using dictionary lookups (much faster)
    rows_indices = [feature_to_idx.get(peak) for peak in sig_peaks['peak'] if peak in feature_to_idx]
    col_indices = [column_to_idx.get(motif) for motif in motifs if motif in column_to_idx]

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
    if n_positions * len(motifs) < 10**7:  # Arbitrary threshold, adjust based on your memory
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
            odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
            
            # Calculate observed co-binding rate and expected rate
            observed_rate = a / n_positions
            expected_rate = (a + b) * (a + c) / n_positions**2
            
            # Calculate fold change (fold enrichment)
            if expected_rate > 0:
                fold_change = observed_rate / expected_rate
            else:
                if i != j:  # Only print warning for non-diagonal elements
                    print(f"Warning: Expected rate is 0 for {motifs[i]} and {motifs[j]}")
                fold_change = 5  # Cap infinite values
                
            # Store fold change in matrix (for visualization/clustering)
            fold_change_matrix[i, j] = min(fold_change, 5)  # Cap at 5 for visualization
                
            # Skip self-comparisons in the results DataFrame (to avoid self-correlations)
            if i == j and not return_matrix:
                continue
                
            # Store results
            results.append({
                'TF1': motifs[i],
                'TF2': motifs[j],
                'co_binding_count': a,
                'TF1_only_count': b,
                'TF2_only_count': c,
                'neither_count': d,
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'fold_change': fold_change,
                'observed_rate': observed_rate,
                'expected_rate': expected_rate
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    if return_matrix:
        return results_df, fold_change_matrix
    else:
        return results_df

def plot_heatmap(results_df, motifs, all_results=None, metric='fold_change', sig_threshold=0.05, 
                color_by_pval=False, cluster=True, fold_change_matrix=None, font_scale_factor=2.5, use_motif_name=False,
                log10_pval_cap=10, fc_cap=3):
    """
    Plot a heatmap of TF co-binding relationships using seaborn's clustermap
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from analyze_tf_cobinding
    motifs : list
        List of TF motif names
    all_results : pandas.DataFrame, optional
        DataFrame containing hit_type annotations for each motif
    metric : str
        Metric to plot ('fold_change', 'odds_ratio')
    sig_threshold : float
        Significance threshold for adjusted p-value
    color_by_pval : bool
        If True, color the heatmap by -log10(adjusted p-value) instead of the metric
    cluster : bool
        Whether to cluster the TFs based on similarity
    fold_change_matrix : numpy.ndarray, optional
        Pre-computed fold change matrix for improved clustering
    font_scale_factor : float, optional
        Scale factor for font sizes (default: 1.0)
    use_motif_name : bool, optional
        If True, use motif names instead of gene names for labels
    log10_pval_cap : float, optional
        Cap for -log10(p-value) values (default: 10)
    fc_cap : float, optional
        Cap for fold change values (default: 3)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # If we have a pre-computed fold change matrix, use it
    if fold_change_matrix is not None:
        # Convert to DataFrame (but still apply significance filtering)
        heatmap_df = pd.DataFrame(fold_change_matrix, index=motifs, columns=motifs)
        
        # If color_by_pval is True, we'll still need to create a new matrix
        if color_by_pval:
            # Create p-value matrix
            n_tfs = len(motifs)
            heatmap_data = np.zeros((n_tfs, n_tfs))
            
            # Fill with p-values
            for _, row in results_df.iterrows():
                if row['adjusted_p_value'] <= sig_threshold:
                    try:
                        i = motifs.index(row['TF1'])
                        j = motifs.index(row['TF2'])
                        
                        # Set to -log10(p-value)
                        value = -np.log10(max(row['adjusted_p_value'], 1e-100))
                        
                        heatmap_data[i, j] = value
                    except ValueError:
                        # Skip if TF not in motifs list
                        continue
                        
            # Convert to DataFrame
            heatmap_df = pd.DataFrame(heatmap_data, index=motifs, columns=motifs)
    else:
        # Create a matrix to hold the values
        n_tfs = len(motifs)
        heatmap_data = np.zeros((n_tfs, n_tfs))
        
        # Ensure we have a minimum value for empty matrices
        heatmap_data[0, 0] = 1e-9  # Add a tiny value to prevent visualization errors
        
        # Fill the matrix with the specified values
        for _, row in results_df.iterrows():
            if row['adjusted_p_value'] <= sig_threshold:
                try:
                    i = motifs.index(row['TF1'])
                    j = motifs.index(row['TF2'])
                    
                    if color_by_pval:
                        # Use -log10(adjusted p-value) for coloring
                        value = -np.log10(max(row['adjusted_p_value'], 1e-100))  # Avoid log(0)
                    else:
                        value = row[metric]
                    
                    heatmap_data[i, j] = value
                except ValueError:
                    # Skip if TF not in motifs list
                    continue
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data, index=motifs, columns=motifs)
    
    # Create a mapping from motifs to gene names if available
    gene_names = {}
    if all_results is not None and 'gene' in all_results.columns:
        for motif in motifs:
            gene_rows = all_results[all_results['motif'] == motif]
            if len(gene_rows) > 0 and not pd.isna(gene_rows['gene'].values[0]):
                gene_names[motif] = gene_rows['gene'].values[0]
            else:
                gene_names[motif] = motif
    else:
        # If no gene names available, use motifs
        gene_names = {motif: motif for motif in motifs}
    
    # Update row and column names with gene names
    if not use_motif_name:
        heatmap_df.index = [gene_names[motif] for motif in motifs]
        heatmap_df.columns = [gene_names[motif] for motif in motifs]
    
    # Create row colors for annotations if all_results is provided
    row_colors = None
    col_colors = None
    if all_results is not None:
        # Create a Series mapping motifs to hit_types (categorical)
        hit_types = pd.Series(index=heatmap_df.index)
        
        # Create a Series for expression (continuous)
        expr_data = pd.Series(index=heatmap_df.index)
        
        # Determine which column to use for expression
        expr_col = None
        for col in ['AveExpr']:
            if col in all_results.columns:
                expr_col = col
                break
        
        for i, motif in enumerate(motifs):
            # Get the corresponding gene name
            gene_name = heatmap_df.index[i]
            
            # Get hit_type (categorical)
            hit_type_rows = all_results[all_results['motif'] == motif]
            if len(hit_type_rows) > 0:
                hit_types[gene_name] = hit_type_rows['hit_type'].values[0] if 'hit_type' in hit_type_rows.columns else 'unknown'
                
                # Get expression if it exists (continuous)
                if expr_col is not None:
                    expr_data[gene_name] = hit_type_rows[expr_col].values[0]
                else:
                    expr_data[gene_name] = np.nan
            else:
                hit_types[gene_name] = 'unknown'
                expr_data[gene_name] = np.nan
        
        # Create categorical color mapping (lut = lookup table)
        if 'hit_type' in all_results.columns:
            # Define a fixed color scheme for the three hit types
            lut = {
                'rna': '#1f77b4',      # Blue for RNA
                'atac': '#ff7f0e',     # Orange for ATAC
                'multimodal': '#2ca02c' # Green for multimodal
            }
            
            # For any additional hit types that might appear
            unique_hit_types = hit_types.unique()
            for ht in unique_hit_types:
                if ht not in lut:
                    # Use gray for any unknown hit types
                    lut[ht] = '#7f7f7f'
            
            # Map hit_types to colors for rows
            row_colors = hit_types.map(lut)
        
        # Create a continuous color mapping for expression
        if expr_col is not None and not expr_data.isna().all():
            # Create a colormap without normalization - use the values directly
            cmap = plt.cm.YlOrRd  # Yellow-Orange-Red colormap
            
            # Use a custom function to map values to colors
            # For missing values, use gray
            def map_to_color(x):
                if np.isnan(x):
                    return (0.8, 0.8, 0.8, 1)
                else:
                    # Use the colormap directly without normalizing
                    # Scale based on the range of your expression data
                    max_expr = 10  # Adjust based on your data
                    return cmap(min(x, max_expr) / max_expr)
                    
            col_colors = expr_data.map(map_to_color)
    
    # Choose the appropriate colormap
    if color_by_pval:
        cmap = 'Reds'
        title_metric = "-log10(adjusted p-value)"
    else:
        cmap = 'viridis'
        title_metric = metric.replace("_", " ").title()
    
    # Set figure title
    title = f'TF Co-binding {title_metric} (adj. p-value < {sig_threshold})'
    
    # Calculate font size based on number of genes
    # Inverse relationship: more genes = smaller font
    base_font_size = 14
    num_genes = len(motifs)
    if num_genes > 20:
        font_size = max(9, base_font_size * (20 / num_genes) * font_scale_factor)
    else:
        font_size = base_font_size * font_scale_factor
    
    legend_font_size = max(16, font_size * 1.1)  # Make legends slightly larger
    title_font_size = max(16, font_size * 1.5)  # Make title larger
    
    # Create the clustermap without capping to determine the order
    try:
        # Add a small non-zero value to diagonal for clustering stability
        np.fill_diagonal(heatmap_df.values, np.maximum(np.diag(heatmap_df.values), 1e-9))
        
        # Set up keyword arguments for the initial clustermap (for determining row/col order)
        clustermap_kwargs = {
            'data': heatmap_df,
            'cmap': cmap,
            'figsize': (16, 14),
            'linewidths': 0.5,
            'cbar_pos': None
        }
        
        # Add row/column colors if available
        if row_colors is not None:
            clustermap_kwargs['row_colors'] = row_colors
        
        if col_colors is not None:
            clustermap_kwargs['col_colors'] = col_colors
            
        # Configure clustering
        if cluster:
            # Use the correlation method for clustering similar patterns
            clustermap_kwargs['method'] = 'average'  # Use average linkage
            clustermap_kwargs['metric'] = 'correlation'  # Use correlation distance
            clustermap_kwargs['row_cluster'] = True
            clustermap_kwargs['col_cluster'] = True
            clustermap_kwargs['dendrogram_ratio'] = 0
        else:
            clustermap_kwargs['row_cluster'] = False
            clustermap_kwargs['col_cluster'] = False
            clustermap_kwargs['dendrogram_ratio'] = 0
        
        # Create initial clustermap to get the ordering
        g_initial = sns.clustermap(**clustermap_kwargs)
        plt.close()  # Close the initial figure to avoid displaying it
        
        # Get the row and column order from the initial clustermap
        if cluster:
            row_order = g_initial.dendrogram_row.reordered_ind
            col_order = g_initial.dendrogram_col.reordered_ind
        else:
            row_order = list(range(len(heatmap_df.index)))
            col_order = list(range(len(heatmap_df.columns)))
        
        # Get the ordered index and columns
        ordered_index = [heatmap_df.index[i] for i in row_order]
        ordered_columns = [heatmap_df.columns[i] for i in col_order]
        
        # Now apply capping to the data before the final visualization
        if color_by_pval:
            capped_data = heatmap_df.copy()
            # Apply log10_pval_cap to all values
            capped_data = capped_data.clip(upper=log10_pval_cap)
        else:
            capped_data = heatmap_df.copy()
            # Apply fc_cap to all values
            capped_data = capped_data.clip(upper=fc_cap)
        
        # Re-create clustermap with capped data but preserving the order
        capped_data = capped_data.loc[ordered_index, ordered_columns]
        
        # Set up keyword arguments for the final clustermap with capped data
        final_clustermap_kwargs = {
            'data': capped_data,
            'cmap': cmap,
            'figsize': (16, 14),
            'linewidths': 0.5,
            'cbar_pos': None,
            'row_cluster': False,  # Don't re-cluster, use the predetermined order
            'col_cluster': False   # Don't re-cluster, use the predetermined order
        }
        
        # Add row/column colors if available (need to reorder them too)
        if row_colors is not None:
            row_colors = row_colors.loc[ordered_index]
            final_clustermap_kwargs['row_colors'] = row_colors
        
        if col_colors is not None:
            col_colors = col_colors.loc[ordered_columns]
            final_clustermap_kwargs['col_colors'] = col_colors
        
        # Create the final clustermap with capped data
        g = sns.clustermap(**final_clustermap_kwargs)
        
        # Set tick font sizes according to calculated size
        plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=font_size)
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=font_size)
        
        # Create a separate title above the plot instead of using suptitle
        # First, adjust the figure to make room for title
        g.figure.subplots_adjust(top=0.9)  # Make room for title at top
        
        # Add title as a separate text element with proper positioning
        cap_info = ""
        if color_by_pval:
            cap_info = f" (capped at {log10_pval_cap})"
        else:
            cap_info = f" (capped at {fc_cap})"
        
        g.figure.text(0.5, 0.96, title + cap_info, fontsize=title_font_size, ha='center', va='center')
        
        # Rotate x-axis labels for better readability
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='right')
        
        # Move y-axis labels to the right to avoid overlap with heatmap
        g.ax_heatmap.yaxis.set_label_position('right')
        g.ax_heatmap.yaxis.tick_right()
        
        # Add padding between y-tick labels and heatmap
        g.ax_heatmap.tick_params(axis='y', pad=10)  # Increase padding (adjust value as needed)
        
        # Adjust alignment of y-tick labels
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, ha='left')
        
        # Add legends on the right side
        # Get the figure
        fig = g.figure
        
        # Hit Types legend (categorical row colors)
        if all_results is not None and 'hit_type' in all_results.columns:
            handles = [plt.Rectangle((0, 0), 1, 1, color=lut[ht]) for ht in unique_hit_types]
            leg1 = fig.legend(handles, unique_hit_types, title="Hit Types", 
                        loc='center right', bbox_to_anchor=(1.35, 0.85), 
                        fontsize=legend_font_size, title_fontsize=legend_font_size)
        
        # Expression colorbar (continuous column colors)
        if col_colors is not None and expr_col is not None:
            cax1 = fig.add_axes([1.08, 0.5, 0.03, 0.2])  # Slightly larger colorbar
            # Use unnormalized values for colorbar
            vmin = 0
            vmax = 10  # Adjust based on your data
            norm1 = plt.Normalize(vmin=vmin, vmax=vmax)
            sm1 = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm1)
            sm1.set_array([])
            cbar1 = fig.colorbar(sm1, cax=cax1)
            cbar1.set_label('Avg Expr', fontsize=legend_font_size)
            cbar1.ax.tick_params(labelsize=legend_font_size * 0.8)
            
        # Main heatmap colorbar
        cax2 = fig.add_axes([1.08, 0.2, 0.03, 0.2])  # Slightly larger colorbar
        
        # Use actual data min/max with capping for colorbar
        vmin = capped_data.values.min()
        if color_by_pval:
            vmax = min(capped_data.values.max(), log10_pval_cap)
        else:
            vmax = min(capped_data.values.max(), fc_cap)
            
        norm2 = plt.Normalize(vmin=vmin, vmax=vmax)
        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.set_label(title_metric, fontsize=legend_font_size)
        cbar2.ax.tick_params(labelsize=legend_font_size * 0.8)
            
        return g
        
    except Exception as e:
        # If clustermap fails, fall back to regular heatmap
        print(f"ClusterMap error: {e}")
        print("Falling back to regular heatmap...")
        
        # Apply capping to data
        if color_by_pval:
            heatmap_df = heatmap_df.clip(upper=log10_pval_cap)
        else:
            heatmap_df = heatmap_df.clip(upper=fc_cap)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_df, cmap=cmap, ax=ax, linewidths=0.5)
        
        cap_info = ""
        if color_by_pval:
            cap_info = f" (capped at {log10_pval_cap})"
        else:
            cap_info = f" (capped at {fc_cap})"
        
        plt.title(title + cap_info, fontsize=title_font_size)
        plt.xticks(rotation=45, ha='right', fontsize=font_size)
        plt.yticks(fontsize=font_size)
        
        return fig