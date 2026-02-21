import streamlit as st
import pandas as pd
import numpy as np

def parse_row_info(rows_df):
    """
    Parse the combined SRA_ID_celltype format into separate columns
    """
    # Split the row identifiers into SRA_ID and cell_type
    split_info = rows_df.iloc[:, 0].str.split("_", n=1)
    return pd.DataFrame(
        {
            "SRA_ID": [x[0] for x in split_info],
            "cell_type": [x[1] if len(x) > 1 else "" for x in split_info],
        }
    )


def create_color_mapping(cell_types=None):
    """
    Create a consistent color mapping for cell types
    """
    # Define a color palette (you can adjust these colors as needed)
    color_mapping = {
        'Corticotrophs':"#1f77b4",
        'Endothelial_cells':"#ff7f0e",
        'Endothelial cells':"#ff7f0e",

        'Erythrocytes':"#2ca02c",
        'Gonadotrophs':"#d62728",
        'Immune_cells':"#9467bd",
        'Immune cells':"#9467bd",
        'Lactotrophs': "#8c564b",
        'Melanotrophs':"#e377c2",
        'Mesenchymal_cells':"#7f7f7f",
        'Mesenchymal cells':"#7f7f7f",
       'Pituicytes': "#bcbd22",
       'Somatotrophs':  "#17becf",
       'Stem_cells': "#aec7e8",
       "Stem cells": "#aec7e8",
        'Thyrotrophs':"#ffbb78"
    }

    # Create mapping of cell types to colors
    #return dict(zip(sorted(cell_types), colors[: len(cell_types)]))

    #colors = {'Corticotrophs': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    #'Endothelial_cells': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    #'Erythrocytes': (1.0, 0.4980392156862745, 0.054901960784313725),
    #'Gonadotrophs': (1.0, 0.7333333333333333, 0.47058823529411764),
    #'Immune_cells': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    #'Lactotrophs': (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    #'Melanotrophs': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    #'Mesenchymal_cells': (1.0, 0.596078431372549, 0.5882352941176471),
    #'Pituicytes': (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    #'Somatotrophs': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    #'Stem_cells': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    #'Thyrotrophs': (0.7686274509803922, 0.611764705882353, 0.5803921568627451)}

    # convert to rgb
    #color_mapping = {ct: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for ct, (r, g, b) in colors.items() if ct in cell_types}
    return color_mapping


def filter_data(
    meta_data, age_range, selected_samples, selected_authors, matrix, only_normal=False,sex_analysis_settings=False,age_analysis_settings=False, modality=["sc","sn","multi_rna","atac","multi_atac"]
):
    """
    Filter the data based on selected samples, authors, and normal status
    """
    # Create base mask for samples, authors, and remove Erythrocytes

    mask = (
        (meta_data["Name"].isin(selected_samples))
        & (meta_data["Author"].isin(selected_authors))
        & (meta_data["new_cell_type"] != "Erythrocytes")
        & (meta_data["Age_numeric"] >= age_range[0])
        & (meta_data["Age_numeric"] <= age_range[1])
        & (meta_data["Modality"].isin(modality))
    )

    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data["Normal"] == 1)

    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask]

    return filtered_meta, filtered_matrix


def filter_chromvar_data(
    meta_data, selected_samples, selected_authors, matrix, only_normal=False
):
    """
    Filter ChromVAR data based on selected criteria
    """
    # Convert matrix to CSR format if it isn't already
    if not hasattr(matrix, "tocsr"):
        matrix = scipy.sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    # Create base mask
    mask = (meta_data["Name"].isin(selected_samples)) & (
        meta_data["Author"].isin(selected_authors)
    )

    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data["Normal"] == 1)

    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask.values]  # Use .values to get numpy array

    return filtered_meta, filtered_matrix


def filter_accessibility_data(
    meta_data, selected_samples, selected_authors, matrix, only_normal=False
):
    """
    Filter accessibility data based on selected criteria
    """
    # Convert matrix to CSR format if it isn't already
    if not hasattr(matrix, "tocsr"):
        matrix = scipy.sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    # Create base mask
    mask = (meta_data["Name"].isin(selected_samples)) & (
        meta_data["Author"].isin(selected_authors)
    )

    # Add normal filter if requested
    if only_normal:
        mask = mask & (meta_data["Normal"] == 1)

    # Filter metadata and matrix
    filtered_meta = meta_data[mask]
    filtered_matrix = matrix[:, mask.values]

    return filtered_meta, filtered_matrix


def create_filter_ui(meta_data,sex_analysis=False,age_analysis=False,  key_suffix=""):
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
    
    #print a hint that Please refer to Curation tab for more details
    
    # reset index of metadata
    meta_data = meta_data.reset_index(drop=True)
    # Filter type selection
    if sex_analysis:
        filter_type = st.radio(
            "Filter data by:",
            ["No filter", "Sample", "Author", "Age","Modality", "Reproduce sex-specific analysis"],
            key=f"filter_type_{key_suffix}",
        )
    elif age_analysis:
        filter_type = st.radio(
            "Filter data by:",
            ["No filter", "Sample", "Author", "Age","Modality", "Reproduce age-dependent analysis"],
            key=f"filter_type_{key_suffix}",
        )
    else:
        filter_type = st.radio(
            "Filter data by:",
            ["No filter", "Sample", "Author", "Age","Modality"],
            key=f"filter_type_{key_suffix}",
        )
    
    # Initialize filter variables
    try:

        all_samples = [
            f"{meta_data['SRA_ID'][i]} - {meta_data['Author'][i]} - {meta_data['Name'][i]}"
            for i in range(len(meta_data))
        ]
        # keep unique
        all_samples = list(set(all_samples))
        all_authors = sorted(meta_data["Author"].unique())
    except KeyError as e:
        st.error(f"Required column missing from metadata: {e}")
        return "No filter", [], [], None, False

    selected_samples = [s.split(" - ")[-1] for s in all_samples]
    selected_authors = all_authors
    modality = meta_data["Modality"].unique().tolist()
    # meta_data age numeric turn , to .
    meta_data["Age_numeric"] = meta_data["Age_numeric"].replace(",", ".", regex=True)
    age_range = (
        float(min(meta_data["Age_numeric"])),
        float(max(meta_data["Age_numeric"])),
    )

    # Show relevant filter based on selection
    if filter_type == "Reproduce sex-specific analysis":
        st.info(""" The settings are now fixed to match those used for the sex-specific analysis in the publication.""", icon="ℹ️")
    elif filter_type == "Reproduce age-dependent analysis":
        st.info(""" The settings are now fixed to match those used for the age-dependent analysis in the publication.""", icon="ℹ️")
                
    elif filter_type != "No filter":
        st.info("""
    Please refer to the Curation tab for more details on sample metadata, to identify your samples of interest.
    """, icon="ℹ️")
        
    if filter_type == "Sample":
        if len(all_samples) > 0:
            selected_samples = st.multiselect(
                "Select Samples",
                all_samples,
                default=[all_samples[0]],
                help="Choose which samples to include in the analysis",
                key=f"samples_multiselect_{key_suffix}",
            )
            # Extract Name
            selected_samples = [s.split(" - ")[-1] for s in selected_samples]

        else:
            st.warning("No samples available for selection")
            selected_samples = []

    elif filter_type == "Author":
        if len(all_authors) > 0:
            selected_authors = st.multiselect(
                "Select Authors",
                all_authors,
                default=[all_authors[0]],
                help="Choose which authors' data to include",
                key=f"authors_multiselect_{key_suffix}",
            )
        else:
            st.warning("No authors available for selection")
            selected_authors = []

    elif filter_type == "Age":
        if "Age_numeric" not in meta_data.columns:
            st.error("Age_numeric column not found in metadata")
            age_range = None
        else:
            # Convert age values to float, handling both string and numeric inputs
            try:

                age_values = pd.to_numeric(
                    meta_data["Age_numeric"].replace(",", ".", regex=True),
                    errors="coerce",
                )
                meta_data["Age_numeric"] = age_values
                valid_ages = age_values.dropna()

                if len(valid_ages) > 0:
                    min_age = float(valid_ages.min())
                    max_age = float(valid_ages.max())

                    # Create slider with consistent float values
                    age_range = st.slider(
                        "Select Age Range (days)",
                        min_value=float(min_age),
                        max_value=float(max_age),
                        value=(float(min_age), float(max_age)),
                        step=1.0,
                        help="Filter samples by age range",
                        key=f"age_slider_{key_suffix}",
                    )

                    # Display age distribution information
                    st.info(
                        f"""
                        Age range: {age_range[0]} to {age_range[1]} - 
                        Number of pseudobulk samples with valid age data: {len(meta_data[(meta_data['Age_numeric'] >= age_range[0]) & (meta_data['Age_numeric'] <= age_range[1])])}
                        
                    """
                    )
                else:
                    st.warning("No valid age data available for filtering")
                    age_range = None
            except Exception as e:
                st.error(f"Error processing age data: {str(e)}")
                age_range = None
    elif filter_type == "Modality":

        modality_options = meta_data["Modality"].unique().tolist()
        
        selected_modality = st.multiselect(
            "Select Modality",
            modality_options,
            default=modality_options[0],
            help="Choose which data modality to include (e.g., sc, sn, multiome)",
            key=f"modality_multiselect_{key_suffix}",
        )
        modality = selected_modality

    elif filter_type == "Reproduce sex-specific analysis":
        age_range = (10,150)
        #remove where starts with Rebboah
        selected_authors = [s for s in all_authors if not s.startswith("Rebboah")]
        only_normal = False
        #ensure modality is sc or sn
        selected_samples = meta_data[
            (meta_data["Modality"].isin(["sc", "sn"])) & (~meta_data["Author"].str.startswith("Rebboah"))
        ]["Name"].tolist()
        modality = ["sc", "sn"]
        
        return filter_type, selected_samples, selected_authors, age_range, only_normal, modality
    
    elif filter_type == "Reproduce age-dependent analysis":
        
        only_normal = False
        return filter_type, selected_samples, selected_authors, (0,2000), only_normal, modality


    # Wild-type filter toggle
    if "Normal" in meta_data.columns:
        only_normal = st.checkbox(
            "Show control samples only",
            value=False,
            help="Samples that are wild-type, untreated etc. (In curation, Normal == 1)",
            key=f"only_normal_{key_suffix}",
        )

        if only_normal:
            # Get total unique SRA_IDs
            total_sra_ids = len(set(meta_data["SRA_ID"].unique()))
            # Get normal samples SRA_IDs
            normal_sra_ids = len(
                set(meta_data[meta_data["Normal"] == 1]["SRA_ID"].unique())
            )
    else:
        st.warning("Normal/Wild-type information not available")
        only_normal = False

    return filter_type, selected_samples, selected_authors, age_range, only_normal,modality


def create_cell_type_stats_display(
    version,
    sra_ids="all",
    cell_types="all",
    display_title=None,
    column_count=4,
    BASE_PATH=None,
    size="large",
    atac_rna="rna",
):
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


    # Define styles based on size
    styles = {
        "large": {
            "padding": "20px",
            "margin": "10px",
            "title_font": "16px",
            "number_font": "24px",
            "shadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
        "small": {
            "padding": "10px",
            "margin": "5px",
            "title_font": "14px",
            "number_font": "18px",
            "shadow": "0 1px 2px rgba(0,0,0,0.1)",
        },
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
            rna_stats_df = pd.read_parquet(
                f"{BASE_PATH}/data/overview/{version}/rna_cell_type_counts.parquet"
            )
            # Filter RNA data if specific SRA_IDs provided
            if isinstance(sra_ids, list):
                rna_stats_df = rna_stats_df[rna_stats_df["dataset"].isin(sra_ids)]
            
            #in colnames split at _
            rna_stats_df.columns = [col.split("_")[0] if "_" in col else col for col in rna_stats_df.columns]

        if atac_rna in ["atac", "atac+rna"]:
            atac_stats_df = pd.read_parquet(
                f"{BASE_PATH}/data/overview/{version}/atac_cell_type_counts.parquet"
            )
            # Filter ATAC data if specific SRA_IDs provided
            if isinstance(sra_ids, list):
                atac_stats_df = atac_stats_df[atac_stats_df["dataset"].isin(sra_ids)]

            #in colnames split at _
            atac_stats_df.columns = [col.split("_")[0] if "_" in col else col for col in atac_stats_df.columns]

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
    available_cell_types = [col for col in cell_stats_df.columns if col != "dataset"]

    # Filter cell types if specific ones are requested
    if isinstance(cell_types, list) and cell_types != "all":
        # Verify requested cell types exist in the data
        valid_cell_types = [ct for ct in cell_types if ct in available_cell_types]
        if len(valid_cell_types) < len(cell_types):
            invalid_types = set(cell_types) - set(valid_cell_types)
            st.warning(
                f"Some requested cell types were not found in the data: {invalid_types}"
            )
        display_cell_types = valid_cell_types
    else:
        display_cell_types = available_cell_types

    # Calculate totals for selected cell types
    rna_totals = {}
    atac_totals = {}

    if rna_stats_df is not None:
        rna_totals = {
            ct: int(rna_stats_df[ct].sum()) if ct in rna_stats_df.columns else 0
            for ct in display_cell_types
        }

    if atac_stats_df is not None:
        atac_totals = {
            ct: int(atac_stats_df[ct].sum()) if ct in atac_stats_df.columns else 0
            for ct in display_cell_types
        }

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
                        st.markdown(
                            f"""
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
                        """,
                            unsafe_allow_html=True,
                        )
                    elif atac_rna == "atac":
                        # ATAC-only display (purple)
                        st.markdown(
                            f"""
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
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Combined display (RNA in blue, ATAC in purple)
                        st.markdown(
                            f"""
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
                        """,
                            unsafe_allow_html=True,
                        )

    # Return appropriate totals based on mode
    if atac_rna == "rna":
        return rna_totals
    elif atac_rna == "atac":
        return atac_totals
    else:
        return {"rna": rna_totals, "atac": atac_totals}


def create_gene_selector(
    gene_list,
    key_suffix,
    label=None,
    help_text=None,
    on_change_callback=None,
    additional_callback_args=None
):
    """
    Create a gene selector with proper state management.
    
    Parameters:
    -----------
    gene_list : list
        List of genes to choose from
    key_suffix : str
        Suffix for the selectbox key (e.g., "tab1", "browser", "correlation")
        The actual key will be f"gene_select_{key_suffix}"
    label : str, optional
        Custom label for the selectbox. If None, defaults to "Select Gene (X genes)"
    help_text : str, optional
        Help text to display with the selectbox
    on_change_callback : callable, optional
        Additional callback function to run when gene changes
    additional_callback_args : dict, optional
        Additional arguments to pass to the callback function
    
    Returns:
    --------
    str
        The selected gene name
    
    Example usage:
    --------------
    # Simple usage
    selected_gene = create_gene_selector(
        gene_list=sorted(genes[0].unique()),
        key_suffix="tab1"
    )
    
    # With custom callback
    def my_callback(gene_name, version):
        print(f"Gene {gene_name} selected for version {version}")
    
    selected_gene = create_gene_selector(
        gene_list=sorted(genes[0].unique()),
        key_suffix="browser",
        label="Choose target gene",
        on_change_callback=my_callback,
        additional_callback_args={"version": selected_version}
    )
    """

    if hasattr(gene_list, 'tolist'):  # Handle numpy arrays or pandas Series
        gene_list = gene_list.tolist()
    else:
        gene_list = list(gene_list)
    
    # Filter out None, NaN, and non-string values
    gene_list = [g for g in gene_list if g is not None and isinstance(g, str) and g.strip()]
    
    # Now sort the cleaned list
    gene_list = sorted(gene_list)

    
    # Initialize session state if not exists
    if "selected_gene" not in st.session_state:
        st.session_state["selected_gene"] = gene_list[0] if gene_list else None
    
    # Check if current selected gene is in the gene list
    current_gene = st.session_state["selected_gene"]
    if current_gene not in gene_list:
        # If not, use the first gene in the list
        default_gene = gene_list[0] if gene_list else None
    else:
        default_gene = current_gene
    
    # Create the selectbox key
    selectbox_key = f"gene_select_{key_suffix}"
    
    # Create default label if none provided
    if label is None:
        label = f"Select Gene ({len(gene_list)} genes)"
    
    # Define the on_change callback
    def _on_gene_change():
        # Get the new gene from the selectbox
        new_gene = st.session_state[selectbox_key]
        
        # Only update if it's actually different
        if new_gene != st.session_state.get("selected_gene"):
            st.session_state["selected_gene"] = new_gene
            
            # Call additional callback if provided
            if on_change_callback is not None:
                if additional_callback_args:
                    on_change_callback(new_gene, **additional_callback_args)
                else:
                    on_change_callback(new_gene)
    
    # Create the selectbox with callback
    selected_gene = st.selectbox(
        label,
        options=gene_list,
        index=gene_list.index(default_gene) if default_gene and default_gene in gene_list else 0,
        key=selectbox_key,
        help=help_text,
        on_change=_on_gene_change,
        width=250
    )
    
    return selected_gene


# Optional: Add this helper function for genome browser specific needs
def create_gene_selector_with_coordinates(
    gene_list,
    key_suffix,
    annotation_df,
    selected_version,
    flank_fraction=0.2,
    label=None
):
    """
    Create a gene selector that also updates genomic coordinates.
    Useful for genome browser views.
    
    Parameters:
    -----------
    gene_list : list
        List of genes to choose from
    key_suffix : str
        Suffix for the selectbox key
    annotation_df : DataFrame
        Annotation data containing gene coordinates
    selected_version : str
        Version of the data being used
    flank_size : int
        Size of flanking regions to add (default: 10000)
    label : str, optional
        Custom label for the selectbox
    
    Returns:
    --------
    tuple: (selected_gene, selected_region)
    """

    import polars as pl
    
    def update_gene_coordinates(gene_name, annotation_df=annotation_df, flank_fraction=flank_fraction):
        """Callback to update coordinates when gene changes"""
        # Filter for the selected gene
        if isinstance(annotation_df, pl.DataFrame):
            gene_data_pl = annotation_df.filter(pl.col("gene_name") == gene_name)
            gene_data = gene_data_pl.to_pandas()
        else:
            gene_data = annotation_df[annotation_df["gene_name"] == gene_name]
        
        if not gene_data.empty:
            gene_chr = gene_data["seqnames"].iloc[0]
            gene_start = gene_data["start"].min()
            gene_end = gene_data["end"].max()
            flank_size = int((gene_end - gene_start) * flank_fraction)
            selected_region = f"{gene_chr}:{max(0, gene_start - flank_size)}-{gene_end + flank_size}"
            st.session_state["selected_region"] = selected_region
        else:
            # Default fallback if gene not found
            st.session_state["selected_region"] = "chr1:1000000-1100000"
    
    # Create gene selector with coordinate update callback
    selected_gene = create_gene_selector(
        gene_list=gene_list,
        key_suffix=key_suffix,
        label=label,
        on_change_callback=update_gene_coordinates,
        additional_callback_args={"annotation_df": annotation_df, "flank_fraction": flank_fraction}
    )
    
    # Get current region
    selected_region = st.session_state.get("selected_region", "chr1:1000000-1100000")
    
    return selected_gene, selected_region

def create_region_selector(
    key_suffix,
    label=None,
    help_text=None,
    on_change_callback=None,
    additional_callback_args=None
):
    """
    Create a gene selector with proper state management.
    
    Parameters:
    -----------
    gene_list : list
        List of genes to choose from
    key_suffix : str
        Suffix for the selectbox key (e.g., "tab1", "browser", "correlation")
        The actual key will be f"gene_select_{key_suffix}"
    label : str, optional
        Custom label for the selectbox. If None, defaults to "Select Gene (X genes)"
    help_text : str, optional
        Help text to display with the selectbox
    on_change_callback : callable, optional
        Additional callback function to run when gene changes
    additional_callback_args : dict, optional
        Additional arguments to pass to the callback function
    
    Returns:
    --------
    str
        The selected gene name
    
    Example usage:
    --------------
    # Simple usage
    selected_gene = create_gene_selector(
        gene_list=sorted(genes[0].unique()),
        key_suffix="tab1"
    )
    
    # With custom callback
    def my_callback(gene_name, version):
        print(f"Gene {gene_name} selected for version {version}")
    
    selected_gene = create_gene_selector(
        gene_list=sorted(genes[0].unique()),
        key_suffix="browser",
        label="Choose target gene",
        on_change_callback=my_callback,
        additional_callback_args={"version": selected_version}
    )
    """

    # Initialize session state if not exists
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = "chr3:34650405-34652461"
    

    current_region = st.session_state["selected_region"]

    # Create the selectbox key
    selectbox_key = f"region_select_{key_suffix}"
    
    
    # Define the on_change callback
    def _on_region_change():
        # Get the new gene from the selectbox
        new_region= st.session_state[selectbox_key]
        
        # Only update if it's actually different
        if new_region != st.session_state.get("selected_region"):
            st.session_state["selected_region"] = new_region
            
            # Call additional callback if provided
            if on_change_callback is not None:
                if additional_callback_args:
                    on_change_callback(new_region, **additional_callback_args)
                else:
                    on_change_callback(new_region)
    
    # Create the selectbox with callback
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        selected_chr = st.selectbox(
            "Chromosome",
            options=[f"chr{i}" for i in range(1, 23)]
            + ["chrX", "chrY"],
            index=2,
            key="chr_select_browser",
            width=300
        )
    with col2:
        start_pos = st.number_input(
            "Start Position",
            value=34650405,
            min_value=0,
            format="%d",
            key="start_pos_input",
        )
    with col3:
        end_pos = st.number_input(
            "End Position",
            value=34652461,
            min_value=0,
            format="%d",
            key="end_pos_input",
        )

    # Create the new region string
    new_region = f"{selected_chr}:{start_pos}-{end_pos}"
    
    # Check if Submit button is clicked
    if st.button("Submit", key=f"submit_region_{key_suffix}"):
        # Only update if it's actually different
        if new_region != st.session_state.get("selected_region"):
            st.session_state["selected_region"] = new_region
            
            # Call additional callback if provided
            if on_change_callback is not None:
                if additional_callback_args:
                    on_change_callback(new_region, **additional_callback_args)
                else:
                    on_change_callback(new_region)
            
            # Rerun to update the display
            st.rerun()
    
    # Return the current region from session state
    return st.session_state["selected_region"]

def tab_start_button(
    tab_name,
    key_suffix,
):

    # Initialize session state if not exists
    if "current_analysis_tab" not in st.session_state:
        st.session_state["current_analysis_tab"] = tab_name
    
    # Create the selectbox key
    button_key = f"button_{key_suffix}"
    
    
    # Define the on_change callback
    def _on_change():
        st.session_state["current_analysis_tab"] = tab_name


    # Create the selectbox with callback
    click = st.button(
        "Click the button to begin",
        key=button_key,
        on_click=_on_change,

    )

    return click