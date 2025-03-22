import pandas as pd
def parse_row_info(rows_df):
    """
    Parse the combined SRA_ID_celltype format into separate columns
    """
    # Split the row identifiers into SRA_ID and cell_type
    split_info = rows_df.iloc[:, 0].str.split('_', n=1)
    return pd.DataFrame({
        'SRA_ID': [x[0] for x in split_info],
        'cell_type': [x[1] if len(x) > 1 else '' for x in split_info]
    })


def create_color_mapping(cell_types):
    """
    Create a consistent color mapping for cell types
    """
    # Define a color palette (you can adjust these colors as needed)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    # Create mapping of cell types to colors
    return dict(zip(sorted(cell_types), colors[:len(cell_types)]))


def filter_data(meta_data, 
                age_range, selected_samples, selected_authors, matrix, only_normal=False):
    """
    Filter the data based on selected samples, authors, and normal status
    """
    # Create base mask for samples, authors, and remove Erythrocytes
    mask = (
        (meta_data['Name'].isin(selected_samples)) & 
        (meta_data['Author'].isin(selected_authors)) &
        (meta_data['new_cell_type'] != 'Erythrocytes') &
        (meta_data['Age_numeric'] >= age_range[0]) &
        (meta_data['Age_numeric'] <= age_range[1])
    )
    
    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data['Normal'] == 1)
    
    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask]
    
    return filtered_meta, filtered_matrix


def filter_chromvar_data(meta_data, selected_samples, selected_authors, matrix, only_normal=False):
    """
    Filter ChromVAR data based on selected criteria
    """
    # Convert matrix to CSR format if it isn't already
    if not hasattr(matrix, 'tocsr'):
        matrix = scipy.sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    
    # Create base mask
    mask = (
        (meta_data['Name'].isin(selected_samples)) & 
        (meta_data['Author'].isin(selected_authors))
    )
    
    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data['Normal'] == 1)
    
    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask.values]  # Use .values to get numpy array
    
    return filtered_meta, filtered_matrix


def filter_accessibility_data(meta_data, selected_samples, selected_authors, matrix, only_normal=False):
    """
    Filter accessibility data based on selected criteria
    """
    # Convert matrix to CSR format if it isn't already
    if not hasattr(matrix, 'tocsr'):
        matrix = scipy.sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    
    # Create base mask
    mask = (
        (meta_data['Name'].isin(selected_samples)) & 
        (meta_data['Author'].isin(selected_authors))
    )
    
    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data['Normal'] == 1)
    
    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask.values]
    
    return filtered_meta, filtered_matrix

def create_filter_ui(meta_data, key_suffix=""):
    """
    Create a consistent filtering UI interface with proper age handling and data validation.
    
    Parameters:
    -----------
    meta_data : pandas.DataFrame
        DataFrame containing metadata with columns:
        - Name: sample names
        - Author: author names
        - Age_numeric: numeric age values
        - Normal: boolean/int indicating if sample is wild-type
    key_suffix : str, optional
        Suffix to make Streamlit keys unique, by default ""
    
    Returns:
    --------
    tuple
        (filter_type, selected_samples, selected_authors, age_range, only_normal)
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    #reset index of metadata
    meta_data = meta_data.reset_index(drop=True)
    # Filter type selection
    filter_type = st.radio(
        "Filter data by:",
        ["No filter", "Sample", "Author", "Age"],
        key=f'filter_type_{key_suffix}'
    )
    
    # Initialize filter variables
    try:
        
        all_samples = [f"{meta_data['SRA_ID'][i]} - {meta_data['Author'][i]} - {meta_data['Name'][i]}" for i in range(len(meta_data))]
        #keep unique
        all_samples = list(set(all_samples))
        all_authors = sorted(meta_data['Author'].unique())
    except KeyError as e:
        st.error(f"Required column missing from metadata: {e}")
        return "No filter", [], [], None, False
    
    selected_samples = [s.split(' - ')[-1] for s in all_samples]
    selected_authors = all_authors
    #meta_data age numeric turn , to .
    meta_data['Age_numeric'] = meta_data['Age_numeric'].replace(',', '.', regex=True)
    age_range = (float(min(meta_data['Age_numeric'])), float(max(meta_data['Age_numeric'])))
    
    # Show relevant filter based on selection
    if filter_type == "Sample":
        if len(all_samples) > 0:
            selected_samples = st.multiselect(
                'Select Samples',
                all_samples,
                default=[all_samples[0]],
                help="Choose which samples to include in the analysis",
                key=f'samples_multiselect_{key_suffix}'
            )
            # Extract Name
            selected_samples = [s.split(' - ')[-1] for s in selected_samples]

        else:
            st.warning("No samples available for selection")
            selected_samples = []
            
    elif filter_type == "Author":
        if len(all_authors) > 0:
            selected_authors = st.multiselect(
                'Select Authors',
                all_authors,
                default=[all_authors[0]],
                help="Choose which authors' data to include",
                key=f'authors_multiselect_{key_suffix}'
            )
        else:
            st.warning("No authors available for selection")
            selected_authors = []
            
    elif filter_type == "Age":
        if 'Age_numeric' not in meta_data.columns:
            st.error("Age_numeric column not found in metadata")
            age_range = None
        else:
            # Convert age values to float, handling both string and numeric inputs
            try:
                
                age_values = pd.to_numeric(meta_data['Age_numeric'].replace(',', '.', regex=True), errors='coerce')
                meta_data['Age_numeric'] = age_values
                valid_ages = age_values.dropna()

                
                
                if len(valid_ages) > 0:
                    min_age = float(valid_ages.min())
                    max_age = float(valid_ages.max())
                    
                    # Create slider with consistent float values
                    age_range = st.slider(
                        'Select Age Range',
                        min_value=float(min_age),
                        max_value=float(max_age),
                        value=(float(min_age), float(max_age)),
                        step=1.0,
                        help="Filter samples by age range",
                        key=f'age_slider_{key_suffix}'
                    )
                    
                    # Display age distribution information
                    st.info(f"""
                        Age range: {age_range[0]} to {age_range[1]}
                        Number of samples with valid age data: {len(meta_data[(meta_data['Age_numeric'] >= age_range[0]) & (meta_data['Age_numeric'] <= age_range[1])])}
                        
                    """)
                else:
                    st.warning("No valid age data available for filtering")
                    age_range = None
            except Exception as e:
                st.error(f"Error processing age data: {str(e)}")
                age_range = None
    
    # Wild-type filter toggle
    if 'Normal' in meta_data.columns:
        only_normal = st.checkbox(
            'Show control samples only', 
            value=False,
            help="Samples that are wild-type, untreated etc. (In curation, Normal == 1)",
            key=f'only_normal_{key_suffix}'
        )
        
        if only_normal:
            # Get total unique SRA_IDs
            total_sra_ids = len(set(meta_data['SRA_ID'].unique()))
            # Get normal samples SRA_IDs
            normal_sra_ids = len(set(meta_data[meta_data['Normal'] == 1]['SRA_ID'].unique()))
    else:
        st.warning("Normal/Wild-type information not available")
        only_normal = False
    
    return filter_type, selected_samples, selected_authors, age_range, only_normal




def create_cell_type_stats_display(version, sra_ids="all", cell_types="all", display_title=None, 
                                 column_count=4, BASE_PATH=None, size="large", atac_rna="rna"):
    """
    Create a standardized cell type statistics display.
    
    Parameters:
    -----------
    version : str
        Version of the data to use (e.g., 'v_0.01')
    sra_ids : str or list
        Either "all" to use all SRA_IDs, or a list of specific SRA_IDs
    cell_types : str or list
        Either "all" to show all cell types, or a list of specific cell types to display
    display_title : str, optional
        Title to display above the statistics. If None, no title is shown.
    column_count : int, optional
        Number of columns to display stats in (default: 4)
    BASE_PATH : str, optional
        Base path for data files. If None, uses default BASE_PATH
    size : str, optional
        Size of the display boxes. Either "large" (default) or "small"
    atac_rna : str, optional
        Type of data to display. Either "rna" (default), "atac", or "atac+rna"
    
    Returns:
    --------
    dict
        Dictionary containing the cell type totals
    """
    import streamlit as st
    import pandas as pd
    
    # Define styles based on size
    styles = {
        "large": {
            "padding": "20px",
            "margin": "10px",
            "title_font": "16px",
            "number_font": "24px",
            "shadow": "0 2px 4px rgba(0,0,0,0.1)"
        },
        "small": {
            "padding": "10px",
            "margin": "5px",
            "title_font": "14px",
            "number_font": "18px",
            "shadow": "0 1px 2px rgba(0,0,0,0.1)"
        }
    }
    
    # Use default style if invalid size provided
    style = styles.get(size, styles["large"])
    
    from config import Config
    BASE_PATH = Config.BASE_PATH
    
    # Load cell type statistics based on data type
    try:
        rna_stats_df = None
        atac_stats_df = None
        
        if atac_rna in ["rna", "atac+rna"]:
            rna_stats_df = pd.read_csv(f'{BASE_PATH}/data/overview/{version}/rna_cell_type_counts.csv')
            # Filter RNA data if specific SRA_IDs provided
            if isinstance(sra_ids, list):
                rna_stats_df = rna_stats_df[rna_stats_df['dataset'].isin(sra_ids)]
                
        if atac_rna in ["atac", "atac+rna"]:
            atac_stats_df = pd.read_csv(f'{BASE_PATH}/data/overview/{version}/atac_cell_type_counts.csv')
            # Filter ATAC data if specific SRA_IDs provided
            if isinstance(sra_ids, list):
                atac_stats_df = atac_stats_df[atac_stats_df['dataset'].isin(sra_ids)]

    except Exception as e:
        st.error(f"Error loading cell type statistics: {str(e)}")
        return {}
    
    # Determine which data to use for display
    if atac_rna == "rna":
        cell_stats_df = rna_stats_df
    elif atac_rna == "atac":
        cell_stats_df = atac_stats_df
    else:  # atac+rna
        # We'll use RNA for determining cell types, but display both
        cell_stats_df = rna_stats_df
    
    # Get all available cell types (excluding 'dataset' column)
    available_cell_types = [col for col in cell_stats_df.columns if col != 'dataset']
    
    # Filter cell types if specific ones are requested
    if isinstance(cell_types, list) and cell_types != "all":
        # Verify requested cell types exist in the data
        valid_cell_types = [ct for ct in cell_types if ct in available_cell_types]
        if len(valid_cell_types) < len(cell_types):
            invalid_types = set(cell_types) - set(valid_cell_types)
            st.warning(f"Some requested cell types were not found in the data: {invalid_types}")
        display_cell_types = valid_cell_types
    else:
        display_cell_types = available_cell_types
    
    # Calculate totals for selected cell types
    rna_totals = {}
    atac_totals = {}
    
    if rna_stats_df is not None:
        rna_totals = {ct: int(rna_stats_df[ct].sum()) if ct in rna_stats_df.columns else 0 
                      for ct in display_cell_types}
    
    if atac_stats_df is not None:
        atac_totals = {ct: int(atac_stats_df[ct].sum()) if ct in atac_stats_df.columns else 0 
                       for ct in display_cell_types}
    
    # Display title if provided
    if display_title:
        if size == "small":
            st.markdown(f"##### {display_title}")
        else:
            st.markdown(f"#### {display_title}")
    
    # Create rows with specified number of columns
    cell_type_list = list(display_cell_types)
    for i in range(0, len(cell_type_list), column_count):
        cols = st.columns(column_count)
        for j in range(column_count):
            if i + j < len(cell_type_list):
                cell_type = cell_type_list[i + j]
                with cols[j]:
                    if atac_rna == "rna":
                        # RNA-only display (blue)
                        st.markdown(f"""
                            <div style="text-align: center; 
                                 padding: {style['padding']}; 
                                 background-color: #f8f9fa; 
                                 border-radius: 8px; 
                                 margin: {style['margin']}; 
                                 box-shadow: {style['shadow']};">
                                <h3 style="color: #666; 
                                         margin-bottom: 3px; 
                                         font-size: {style['title_font']};">
                                    {cell_type.replace('_', ' ')}
                                </h3>
                                <div style="font-size: {style['number_font']}; 
                                          font-weight: bold; 
                                          color: #0000ff;">
                                    {rna_totals[cell_type]:,}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    elif atac_rna == "atac":
                        # ATAC-only display (purple)
                        st.markdown(f"""
                            <div style="text-align: center; 
                                 padding: {style['padding']}; 
                                 background-color: #f8f9fa; 
                                 border-radius: 8px; 
                                 margin: {style['margin']}; 
                                 box-shadow: {style['shadow']};">
                                <h3 style="color: #666; 
                                         margin-bottom: 3px; 
                                         font-size: {style['title_font']};">
                                    {cell_type.replace('_', ' ')}
                                </h3>
                                <div style="font-size: {style['number_font']}; 
                                          font-weight: bold; 
                                          color: #ff2eff;">
                                    {atac_totals[cell_type]:,}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Combined display (RNA in blue, ATAC in purple)
                        st.markdown(f"""
                            <div style="text-align: center; 
                                 padding: {style['padding']}; 
                                 background-color: #f8f9fa; 
                                 border-radius: 8px; 
                                 margin: {style['margin']}; 
                                 box-shadow: {style['shadow']};">
                                <h3 style="color: #666; 
                                         margin-bottom: 3px; 
                                         font-size: {style['title_font']};">
                                    {cell_type.replace('_', ' ')}
                                </h3>
                                <div style="font-size: {style['number_font']}; 
                                          font-weight: bold; 
                                          color: #0000ff;">
                                    {rna_totals[cell_type]:,}
                                </div>
                                <div style="font-size: {style['number_font']}; 
                                          font-weight: bold; 
                                          color: #ff2eff;">
                                    {atac_totals[cell_type]:,}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Return appropriate totals based on mode
    if atac_rna == "rna":
        return rna_totals
    elif atac_rna == "atac":
        return atac_totals
    else:
        return {"rna": rna_totals, "atac": atac_totals}