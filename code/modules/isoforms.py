import plotly.express as px
import pandas as pd
from .utils import create_color_mapping
import numpy as np
import scipy
import plotly.graph_objects as go


def filter_isoform_data(isoform_matrix, isoform_features, isoform_samples, meta_data, 
                     selected_samples=None, selected_authors=None, age_range=None, only_normal=False):
    """
    Filter isoform data based on selected criteria
    """
    import numpy as np
    import scipy.sparse as sparse
    import pandas as pd
    
    # Convert COO matrix to CSR format if needed
    if sparse.issparse(isoform_matrix):
        isoform_matrix = isoform_matrix.tocsr()
    
    # Create copies to avoid modifying originals
    meta_data = meta_data.copy()
    isoform_samples = isoform_samples.copy()
    
    # Ensure SRA_IDs are strings
    meta_data['SRA_ID'] = meta_data['SRA_ID'].astype(str)
    isoform_samples['SRA_ID'] = isoform_samples['SRA_ID'].astype(str)
    
    # Get initial set of valid SRA_IDs
    valid_sra_ids = set(meta_data['SRA_ID'].unique())
    
    # Apply filters to meta_data first
    if selected_samples:
        valid_sra_ids &= set(meta_data[meta_data['Name'].isin(selected_samples)]['SRA_ID'].unique())
    
    if selected_authors:
        valid_sra_ids &= set(meta_data[meta_data['Author'].isin(selected_authors)]['SRA_ID'].unique())
    
    if age_range and 'Age_numeric' in meta_data.columns:
        # Convert age values to float, handling both string and numeric inputs
        try:
            age_values = pd.to_numeric(meta_data['Age_numeric'].replace(',', '.', regex=True), errors='coerce')
            # Create age mask, handling NaN values
            age_mask = (age_values.notna() & 
                       (age_values >= float(age_range[0])) & 
                       (age_values <= float(age_range[1])))
            valid_sra_ids &= set(meta_data[age_mask]['SRA_ID'].unique())
        except Exception as e:
            print(f"Error in age filtering: {str(e)}")
            # If age filtering fails, continue without it
            pass
    
    if only_normal:
        valid_sra_ids &= set(meta_data[meta_data['Normal'] == 1]['SRA_ID'].unique())
    
    # Filter samples based on valid SRA_IDs
    mask = isoform_samples['SRA_ID'].isin(valid_sra_ids)
    
    # Filter matrix based on mask
    if sparse.issparse(isoform_matrix):
        col_indices = np.where(mask)[0]
        filtered_matrix = isoform_matrix[:, col_indices]
    else:
        filtered_matrix = isoform_matrix[:, mask]
    
    filtered_samples = isoform_samples[mask]
    
    return filtered_matrix, filtered_samples



def create_isoform_plot(isoform_matrix, isoform_features, isoform_samples, meta_data, selected_gene, selected_cell_types=None):
    """
    Create a box plot for transcript-level expression data with enhanced hover information,
    log10 transformation, jittered points, and wider boxes
    """
    import numpy as np
    import scipy.sparse
    import plotly.graph_objects as go
    
    # Filter features for selected gene
    gene_transcripts = isoform_features[isoform_features['gene_name'] == selected_gene]
    
    if len(gene_transcripts) == 0:
        return None, None, f"No transcripts found for gene {selected_gene}"
    
    # Extract expression values and prepare plot data
    plot_data = []
    for idx, transcript in gene_transcripts.iterrows():
        # Get expression values and ensure they're properly converted
        expression = isoform_matrix[idx].toarray().flatten() if scipy.sparse.issparse(isoform_matrix) else isoform_matrix[idx]
        
        # Apply log10 transformation (add 1 to handle zeros)
        expression = np.log10(expression + 1)
        
        for sample_idx, expr_val in enumerate(expression):
            sra_id = isoform_samples.iloc[sample_idx]['SRA_ID']
            cell_type = isoform_samples.iloc[sample_idx]['cell_type']
            
            if cell_type != 'Erythrocytes':
                if selected_cell_types is None or cell_type in selected_cell_types:
                    # Get metadata for this SRA_ID if available
                    matching_meta = meta_data[meta_data['SRA_ID'] == sra_id]
                    if not matching_meta.empty:
                        sample_meta = matching_meta.iloc[0]
                        
                        sample_data = {
                            'Transcript': transcript['transcript_id'],
                            'Expression': expr_val,  # Now log10-transformed
                            'Cell_Type': cell_type,
                            'SRA_ID': sra_id,
                            'Combined': f"{transcript['transcript_id']}_{cell_type}",
                            'Author': sample_meta['Author'],
                            'Age': sample_meta['Age_numeric'],
                            'Sex': sample_meta['Comp_sex'],
                            'Data_Type': sample_meta['sc_sn_atac']
                        }
                        plot_data.append(sample_data)
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) == 0:
        return None, None, "No data available for the selected combination"
    
    # Get cell types and create color mapping
    cell_types = sorted(plot_df['Cell_Type'].unique())
    color_map = create_color_mapping(cell_types)
    transcripts = sorted(plot_df['Transcript'].unique())
    
    # Create ordered categories
    combined_cats = []
    for cell_type in cell_types:
        for transcript in transcripts:
            combined_cats.append(f"{transcript}_{cell_type}")
    
    plot_df = plot_df.sort_values(by=['Cell_Type', 'Transcript'])
    plot_df['Combined'] = pd.Categorical(plot_df['Combined'], categories=combined_cats, ordered=True)
    
    # Create base figure
    fig = go.Figure()
    
    # Add box plots first
    for cell_type in cell_types:
        cell_data = plot_df[plot_df['Cell_Type'] == cell_type]
        for transcript in transcripts:
            subset = cell_data[cell_data['Transcript'] == transcript]
            if not subset.empty:
                combined_key = f"{transcript}_{cell_type}"
                cat_index = combined_cats.index(combined_key)
                
                # Add box plot
                fig.add_trace(go.Box(
                    x=[cat_index] * len(subset),
                    y=subset['Expression'],
                    name=cell_type,
                    marker_color=color_map[cell_type],
                    width=0.8,
                    boxpoints=False,
                    showlegend=False
                ))
                
                # Add jittered points
                jitter = np.random.normal(0, 0.1, len(subset))
                fig.add_trace(go.Scatter(
                    x=np.array([cat_index] * len(subset)) + jitter,
                    y=subset['Expression'],
                    mode='markers',
                    marker=dict(
                        color=color_map[cell_type],
                        opacity=0.4,
                        size=6
                    ),
                    showlegend=False if cat_index > 0 else True,
                    name=cell_type,
                    hovertemplate="<br>".join([
                        f"<b>{combined_key}</b>",
                        "Expression: %{y:.2f}",
                        f"SRA ID: {subset['SRA_ID'].iloc[0]}",
                        f"Author: {subset['Author'].iloc[0]}",
                        f"Age: {subset['Age'].iloc[0]}",
                        f"Sex: {subset['Sex'].iloc[0]}",
                        f"Data Type: {subset['Data_Type'].iloc[0]}",
                        "<extra></extra>"
                    ])
                ))
    
    # Update layout
    fig.update_layout(
        title=f"Transcript Expression for {selected_gene}",
        xaxis=dict(
            title='',
            ticktext=combined_cats,
            tickvals=list(range(len(combined_cats))),
            tickangle=45,
            tickfont=dict(size=8)
        ),
        yaxis=dict(
            title='log10(eCPM)',
            tickfont=dict(size=12)
        ),
        showlegend=True,
        height=600,
        width=2000,
        margin=dict(b=100),
        boxmode='group'
    )
    
    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': f'{selected_gene}_transcript_expression',
            'height': 800,
            'width': 2400,
            'scale': 2
        }
    }
    
    return fig, config, None