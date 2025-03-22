import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode



def add_searchbar_to_aggrid(df, filter_columns=None, key_prefix="search"):
    """
    Add a search bar that filters the dataframe in real-time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to be filtered
    filter_columns : list, optional
        List of columns to include in the search. If None, all columns are used.
    key_prefix : str
        Prefix for unique key generation
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame based on search term
    """
    search_term = st.text_input(
        "🔍 Search across all columns:",
        key=f"{key_prefix}_searchbar"
    )
    
    # If search term is provided, filter the dataframe
    if search_term:
        columns_to_search = filter_columns if filter_columns else df.columns
        
        # Create a filter mask that matches the search term in any selected column
        mask = pd.Series(False, index=df.index)
        for col in columns_to_search:
            # Convert column to string and check if it contains the search term (case-insensitive)
            mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
        
        filtered_df = df[mask]
        
        # Show how many results were found
        st.caption(f"Found {len(filtered_df)} matching results out of {len(df)} total entries")
        
        return filtered_df
    
    return df




def configure_grid_options(df, key_suffix="", default_col_width=150):
    """
    Configure grid options for AgGrid with left-aligned values and headers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to display
    key_suffix : str
        Suffix for unique key generation
    default_col_width : int
        Default width for all columns in pixels (default: 120)
        
    Returns:
    --------
    dict
        Grid options configuration
    """
    # First, inject custom CSS to left-align headers
    custom_css = """
    .ag-header-cell-label {
        justify-content: flex-start !important;
    }
    .ag-header-cell-text {
        text-align: right !important;
        margin-right: 4px !important;
    }
    """
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    
    # Configure grid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=50)
    
    # Configure default column settings with left-aligned text
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sorteable=True,
        width=default_col_width,
        minWidth=default_col_width,
        cellStyle={'text-align': 'right'}  # Left-align cell contents
    )
    
    # Configure column definitions manually to ensure header alignment
    for col in df.columns:
        gb.configure_column(
            col,
            headerClass="right-aligned-header"  # Apply custom class to headers
        )
    
    # Enable floating filters and range selection
    gb.configure_grid_options(
        enableRangeSelection=True,
        defaultColDef={
            'width': default_col_width,
            'minWidth': default_col_width,
            'maxWidth': default_col_width * 2,  # Allow columns to expand up to double width
            'resizable': True,
            'sortable': True,
            'filter': True,
            'cellStyle': {'text-align': 'right'}  # Set left alignment for cell contents
        }
    )
    
    return gb.build()




def display_marker_table(version, load_marker_data_func, key_prefix=""):
    """
    Display and handle marker table functionality using AgGrid.
    """
    try:
        marker_type = st.radio(
            "Select marker type:",
            ["Cell Type Markers", "Grouping/Lineage Markers"],
            horizontal=True,
            key=f"{key_prefix}_marker_type"
        )

        cell_typing_markers, grouping_lineage_markers = load_marker_data_func(version=version)
        
        cell_typing_markers = cell_typing_markers.loc[:, ~cell_typing_markers.columns.str.contains('^Unnamed')]
        grouping_lineage_markers = grouping_lineage_markers.loc[:, ~grouping_lineage_markers.columns.str.contains('^Unnamed')]
        
        marker_data = cell_typing_markers if marker_type == "Cell Type Markers" else grouping_lineage_markers
        
        # Format numeric columns
        if marker_type == "Cell Type Markers":
            marker_data['log2fc'] = marker_data['log2fc'].round(2)
            #-log10 pval
            marker_data["pval"] = -np.log10(marker_data["pval"])
            #rename pval to -log10 pval
            marker_data.rename(columns={"pval": "-log10 pval"}, inplace=True)
        else:
            marker_data['mean_log2fc'] = marker_data['mean_log2fc'].round(2)
            #-log10 geom_mean_adj_pval
            marker_data['geom_mean_adj_pval'] = -np.log10(marker_data['geom_mean_adj_pval'])
            #rename geom_mean_adj_pval to -log10 geom_mean_adj_pval
            marker_data.rename(columns={"geom_mean_adj_pval": "-log10 geom_mean_adj_pval"}, inplace=True)
            if 'AveExpr' in marker_data.columns:
                marker_data['AveExpr'] = marker_data['AveExpr'].round(2)
            if 'TF' in marker_data.columns:
                marker_data['TF'] = marker_data['TF'].map({1: 'Yes', 0: 'No'})

        # Apply search filtering to the data
        filtered_marker_data = add_searchbar_to_aggrid(
            marker_data, 
            key_prefix=f"{key_prefix}_marker"
        )
        
        # Configure and display AgGrid with the filtered data
        grid_options = configure_grid_options(filtered_marker_data, key_prefix)
        grid_response = AgGrid(
            filtered_marker_data,
            gridOptions=grid_options,
            height=600,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid"
        )
        
        filtered_data = grid_response['data']
        st.info(f"Showing {len(filtered_data)} of {len(marker_data)} markers")
        
        st.download_button(
            label="Download Marker Data",
            data=filtered_data.to_csv(index=False),
            file_name=f"marker_data_{marker_type.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            help="Download the current filtered marker dataset",
            key=f"{key_prefix}_download"
        )
        
        if marker_type == "Grouping/Lineage Markers":
                st.markdown("### Grouping Definitions")
                st.markdown("""
                    Each grouping represents a specific comparison between cell populations:
                    
                    **Grouping 1: Stem Cells vs Hormone-Producing Cells**
                    - Group A: Stem cells
                    - Group B: All hormone-producing cells (Melanotrophs, Corticotrophs, Somatotrophs, Lactotrophs, Thyrotrophs, Gonadotrophs)
                    
                    **Grouping 2: Gonadotrophs vs Others**
                    - Group A: Gonadotrophs
                    - Group B: All other pituitary cell types
                    
                    **Grouping 3: TPIT (TBX19)-lineage vs Others**
                    - Group A: TPIT (TBX19)-lineage cells (Melanotrophs, Corticotrophs)
                    - Group B: All other pituitary cell types
                    
                    **Grouping 4: Melanotrophs vs Corticotrophs**
                    - Group A: Melanotrophs
                    - Group B: Corticotrophs
                    
                    **Grouping 5: PIT1 (POU1F1)-lineage vs Others**
                    - Group A: PIT1 (POU1F1)-lineage cells (Somatotrophs, Lactotrophs, Thyrotrophs)
                    - Group B: All other pituitary cell types
                    
                    **Grouping 6: Lactotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Lactotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Somatotrophs, Thyrotrophs)
                    
                    **Grouping 7: Somatotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Somatotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Lactotrophs, Thyrotrophs)
                    
                    **Grouping 8: Thyrotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Thyrotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Lactotrophs, Somatotrophs)
                    
                    *Note: For each grouping, positive log2FC values indicate higher expression in Group A.*
                    In each grouping 1 cell type vs 1 cell type tests were performed and aggregated.
                    We display the geometric mean of P-values and mean of log2FC values.
                """)
            
        return filtered_data
        
    except Exception as e:
        st.error(f"Error loading marker data: {str(e)}")
        return None

def display_aging_genes_table(aging_genes_df, key_prefix=""):
    """
    Display an interactive aging genes table using AgGrid
    """
    if aging_genes_df.empty:
        st.warning("No aging genes data available.")
        return None
    
    # Configure and display AgGrid

    # Apply search filtering to the data
    aging_genes_df = add_searchbar_to_aggrid(
        aging_genes_df, 
        key_prefix=f"{key_prefix}_marker"
    )

    grid_options = configure_grid_options(aging_genes_df, key_prefix)
    grid_response = AgGrid(
        aging_genes_df,
        gridOptions=grid_options,
        height=600,
        width='100%',
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        key=f"{key_prefix}_grid"
    )
    
    filtered_data = grid_response['data']
    st.info(f"Showing {len(filtered_data)} of {len(aging_genes_df)} genes")
    
    st.download_button(
        label="Download Aging Genes Data",
        data=filtered_data.to_csv(index=False),
        file_name="aging_genes_data.csv",
        mime="text/csv",
        key="download_aging_genes",
        help="Download the current filtered aging genes dataset"
    )
    
    return filtered_data

def display_curation_table(curation_data, key_prefix=""):
    """
    Display an interactive curation table using AgGrid
    """
    try:
        st.info("We are humans too. Did we get something wrong? Did we miss your dataset?\n"
                "Please let us know by supplying the relevant information through email.\n")
        
        # Configure and display AgGrid

        curation_data = add_searchbar_to_aggrid(
        curation_data,
        key_prefix=f"{key_prefix}_marker"
    )
        
        grid_options = configure_grid_options(curation_data, key_prefix)
        grid_response = AgGrid(
            curation_data,
            gridOptions=grid_options,
            height=800,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid"
        )
        
        filtered_data = grid_response['data']
        st.info(f"Showing {len(filtered_data)} of {len(curation_data)} entries")
        
        st.download_button(
            label="Download Curated Metadata",
            data=filtered_data.to_csv(index=False),
            file_name="curated_metadata.csv",
            mime="text/csv",
            help="Download the current filtered dataset",
            key=f"{key_prefix}_download_curation"
        )
        
        # Display metadata column explanations
        st.markdown("""
        # Metadata Columns Explained

        ## Sample Information
        - **GEO**: Gene Expression Omnibus accession number (in some cases these are ENA IDs)
        - **SRA_ID**: Sequence Read Archive identifier (in some cases these are ENA IDs)
        - **Name**: Short descriptive name of the sample (almost always taken directly from GEO)
        - **Author**: First author of the associated publication
        - **published**: Whether the dataset has been published (1) or not (0)
        - **species**: Species of origin (mouse)

        ## Experimental Conditions
        - **Conditions**: Experimental conditions or treatments applied
        - **Normal**: Whether the sample is from wild-type/untreated animals (1), treated/mutant animals (0), or organoids (2)
        - **Sorted**: Whether the sample was FACS-sorted and enriched for a specific cell type (1) or represents whole pituitary (0)
        - **Split sample**: Whether the sample was split for multi-modal analysis (1) or not (0)
        - **Single_pool**: Whether multiple samples were pooled for sequencing
        - **10X version**: Version of 10X Genomics chemistry (barcoding kit) used

        ## Sample Demographics
        - **Age**: Age of the mouse in text format
        - **Age_numeric**: Numerical age in days
        - **Sex**: Sex (Male/Female) in text format
        - **Sex_numeric**: Numerical encoding of sex
        - **Comp_sex**: Computational prediction of sex (1 = Male, 0 = Female), adjusting for errors in publications

        ## Data Type and Quality Metrics
        - **sc_sn_atac**: Type of data
            - 'sc': Single-cell RNA-seq
            - 'sn': Single-nucleus RNA-seq
            - 'atac': Single-nucleus ATAC-seq
            - 'multi_rna': RNA component of Multiome data
            - 'multi_atac': ATAC component of Multiome data
                    
        - **n_cells**: Number of cells/nuclei that passed quality control and are in the final object

        ## Workflow columns     
        - **most_recent_workflow**: Version of the processing pipeline used
        - **processed_atac**: Whether ATAC-seq data was processed (1) or not (0)
        - **tcc_pseudobulked**: Whether RNA-seq data was processed for pseudobulk transcript compatibility counts analysis (isoforms)

        ## Additional Information
        - **Notes**: Additional relevant information about the sample
                    
        """)
        
        return filtered_data
        
    except Exception as e:
        st.error(f"Error displaying curation table: {str(e)}")
        return None
    


def display_ligand_receptor_table(liana_df, key_prefix=""):
    """
    Display an interactive ligand-receptor interaction table using AgGrid.
    
    Parameters:
    -----------
    liana_df : pandas.DataFrame
        DataFrame containing ligand-receptor interaction data
    key_prefix : str
        Prefix for unique key generation
    """
    try:
        # Format the data for display
        display_df = liana_df.copy()
        
        # Create interaction columns
        display_df['Interaction'] = display_df['ligand_complex'] + ' → ' + display_df['receptor_complex']
        display_df['Cell_Pair'] = display_df['source'] + ' → ' + display_df['target']
        
        # -log10
        display_df['specificity_rank'] = -np.log10(display_df['specificity_rank'])
        display_df['magnitude_rank'] = -np.log10(display_df['magnitude_rank'])
        
        
        # Select and reorder columns
        display_cols = [
            'Interaction', 'Cell_Pair', 'source', 'target',
            'ligand_complex', 'receptor_complex',
            'specificity_rank', 'magnitude_rank'
        ]
        display_df = display_df[display_cols]
        
        # Rename columns for display
        column_names = {
            'source': 'Source Cell Type',
            'target': 'Target Cell Type',
            'ligand_complex': 'Ligand',
            'receptor_complex': 'Receptor',
            'specificity_rank': '-log10 Specificity',
            'magnitude_rank': '-log10 Magnitude'
        }
        display_df = display_df.rename(columns=column_names)
        
        # Configure and display AgGrid

        display_df = add_searchbar_to_aggrid(
        display_df,
        key_prefix=f"{key_prefix}_marker"
    )
        

        grid_options = configure_grid_options(display_df, key_prefix)
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            height=600,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid"
        )
        
        filtered_data = grid_response['data']
        
        
        st.download_button(
            label="Download Interaction Data",
            data=filtered_data.to_csv(index=False),
            file_name="ligand_receptor_interactions.csv",
            mime="text/csv",
            help="Download the current filtered interaction dataset",
            key=f"{key_prefix}_download"
        )
        
        return filtered_data
        
    except Exception as e:
        st.error(f"Error displaying ligand-receptor table: {str(e)}")
        return None
    

def display_enrichment_table(enrichment_df, key_prefix=""):
    """
    Display an interactive enrichment results table using AgGrid with native filtering
    """
    try:
        if enrichment_df.empty:
            st.warning("No enrichment data available.")
            return None
        
        #turn pvalue to -log10
        #where p.adjust is 0, make it 1e-300
        enrichment_df['p.adjust'] = enrichment_df['p.adjust'].replace(0, 1e-300)
        enrichment_df['-log10_pval'] = -np.log10(enrichment_df['p.adjust'])
        
        #remove pval column
        enrichment_df.drop(columns=['pvalue', 'p.adjust'], inplace=True)
        # Configure grid options with custom column widths and filtering
        gb = GridOptionsBuilder.from_dataframe(enrichment_df)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sorteable=True,
            minWidth=150
        )
        
        # Enable floating filters and advanced filter features
        gb.configure_grid_options(
            enableRangeSelection=True,
            floatingFilter=True,
            defaultColDef={
                'floatingFilter': True,
                'filter': True
            }
        )
        
        grid_options = gb.build()
        
        # Display AgGrid
        grid_response = AgGrid(
            enrichment_df,
            gridOptions=grid_options,
            height=600,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid"
        )
        
        filtered_data = grid_response['data']
        st.info(f"Showing {len(filtered_data)} of {len(enrichment_df)} results")

        # Add download button
        st.download_button(
            label="Download Enrichment Data",
            data=filtered_data.to_csv(index=False),
            file_name="enrichment_results.csv",
            mime="text/csv",
            help="Download the current filtered enrichment results",
            key=f"{key_prefix}_download"
        )
        

        st.markdown("### Grouping Definitions")
        st.markdown("""
                    Each grouping represents a specific comparison between cell populations:
                    
                    **Grouping 1: Stem Cells vs Hormone-Producing Cells**
                    - Group A: Stem cells
                    - Group B: All hormone-producing cells (Melanotrophs, Corticotrophs, Somatotrophs, Lactotrophs, Thyrotrophs, Gonadotrophs)
                    
                    **Grouping 2: Gonadotrophs vs Others**
                    - Group A: Gonadotrophs
                    - Group B: All other cell types (including stem cells)
                    
                    **Grouping 3: TPIT (TBX19)-lineage vs Others**
                    - Group A: TPIT (TBX19)-lineage cells (Melanotrophs, Corticotrophs)
                    - Group B: All other cell types
                    
                    **Grouping 4: Melanotrophs vs Corticotrophs**
                    - Group A: Melanotrophs
                    - Group B: Corticotrophs
                    
                    **Grouping 5: PIT1 (POU1F1)-lineage vs Others**
                    - Group A: PIT1 (POU1F1)-lineage cells (Somatotrophs, Lactotrophs, Thyrotrophs)
                    - Group B: All other cell types
                    
                    **Grouping 6: Lactotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Lactotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Somatotrophs, Thyrotrophs)
                    
                    **Grouping 7: Somatotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Somatotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Lactotrophs, Thyrotrophs)
                    
                    **Grouping 8: Thyrotrophs vs Other PIT1 (POU1F1)-lineage**
                    - Group A: Thyrotrophs
                    - Group B: Other PIT1 (POU1F1)-lineage cells (Lactotrophs, Somatotrophs)
                    
                    *Note: For each grouping, positive log2FC values indicate higher expression in Group A.*
                    In each grouping 1 cell type vs 1 cell type tests were performed and aggregated.
                    We display the geometric mean of P-values and mean of log2FC values.
                """)
        
        return filtered_data
        
    except Exception as e:
        st.error(f"Error displaying enrichment table: {str(e)}")
        return None