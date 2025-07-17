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
        "🔍 Search across all columns:", key=f"{key_prefix}_searchbar"
    )

    # If search term is provided, filter the dataframe
    if search_term:
        columns_to_search = filter_columns if filter_columns else df.columns

        # Create a filter mask that matches the search term in any selected column
        mask = pd.Series(False, index=df.index)
        for col in columns_to_search:
            # Convert column to string and check if it contains the search term (case-insensitive)
            mask = mask | df[col].astype(str).str.contains(
                search_term, case=False, na=False
            )

        filtered_df = df[mask]

        # Show how many results were found
        st.caption(
            f"Found {len(filtered_df)} matching results out of {len(df)} total entries"
        )

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
    gb.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=50
    )

    # Configure default column settings with left-aligned text
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sorteable=True,
        width=default_col_width,
        minWidth=default_col_width,
        cellStyle={"text-align": "right"},  # Left-align cell contents
    )

    # Configure column definitions manually to ensure header alignment
    for col in df.columns:
        gb.configure_column(
            col, headerClass="right-aligned-header"  # Apply custom class to headers
        )

    # Enable floating filters and range selection
    gb.configure_grid_options(
        enableRangeSelection=True,
        defaultColDef={
            "width": default_col_width,
            "minWidth": default_col_width,
            "maxWidth": default_col_width
            * 2,  # Allow columns to expand up to double width
            "resizable": True,
            "sortable": True,
            "filter": True,
            "cellStyle": {
                "text-align": "right"
            },  # Set left alignment for cell contents
        },
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
            key=f"{key_prefix}_marker_type",
        )

        cell_typing_markers, grouping_lineage_markers = load_marker_data_func(
            version=version
        )

        cell_typing_markers = cell_typing_markers.loc[
            :, ~cell_typing_markers.columns.str.contains("^Unnamed")
        ]
        grouping_lineage_markers = grouping_lineage_markers.loc[
            :, ~grouping_lineage_markers.columns.str.contains("^Unnamed")
        ]

        marker_data = (
            cell_typing_markers
            if marker_type == "Cell Type Markers"
            else grouping_lineage_markers
        )

        # Format numeric columns
        if marker_type == "Cell Type Markers":
            marker_data["log2fc"] = marker_data["log2fc"].round(2)
            # -log10 pval
            marker_data["pval"] = -np.log10(marker_data["pval"] + 1e-300)
            # rename pval to -log10 pval
            marker_data.rename(columns={"pval": "-log10 pval"}, inplace=True)

            #get only two decimals for both
            marker_data["log2fc"] = marker_data["log2fc"].round(2)
            marker_data["-log10 pval"] = marker_data["-log10 pval"].round(2)

        else:
            marker_data["mean_log2fc"] = marker_data["mean_log2fc"].round(2)
            # -log10 geom_mean_adj_pval
            marker_data["geom_mean_adj_pval"] = -np.log10(
                marker_data["geom_mean_adj_pval"] + 1e-300
            )
            marker_data["geom_mean_adj_pval"] = marker_data["geom_mean_adj_pval"].round(2)

            # rename geom_mean_adj_pval to -log10 geom_mean_adj_pval
            marker_data.rename(
                columns={"geom_mean_adj_pval": "-log10 geom_mean_adj_pval"},
                inplace=True,
            )
            if "AveExpr" in marker_data.columns:
                marker_data["AveExpr"] = marker_data["AveExpr"].round(2)
        

        # Apply search filtering to the data
        filtered_marker_data = add_searchbar_to_aggrid(
            marker_data, key_prefix=f"{key_prefix}_marker"
        )

        #

        # Configure and display AgGrid with the filtered data
        grid_options = configure_grid_options(filtered_marker_data, key_prefix)
        grid_response = AgGrid(
            filtered_marker_data,
            gridOptions=grid_options,
            height=600,
            width="100%",
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid",
        )

        filtered_data = grid_response["data"]
        st.info(f"Showing {len(filtered_data)} of {len(marker_data)} markers")

        st.download_button(
            label="Download Marker Data",
            data=filtered_data.to_csv(index=False),
            file_name=f"marker_data_{marker_type.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            help="Download the current filtered marker dataset",
            key=f"{key_prefix}_download",
        )

        if marker_type == "Grouping/Lineage Markers":
            st.markdown("### Grouping Definitions")
            st.markdown(
                """
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
                """
            )

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
    #rename some cols Logfc, Aveexpr, T, P.Value, Adj.P.Val, B, Genes
    aging_genes_df = aging_genes_df.rename(
        columns={
            "Logfc": "log2FC",
            "Aveexpr": "AveExpr",
            "T": "t",
            "P.Value": "pvalue",
            "Adj.P.Val": "adj.P.Val",
            "B": "B",
            "Genes": "gene",
        }
    )
    #make -log10 adj.P.Val
    aging_genes_df["-log10_adj_pval"] = -np.log10(aging_genes_df["adj.P.Val"] + 1e-300)  
    #remove pvalue and adj.P.Val
    aging_genes_df = aging_genes_df.drop(columns=["pvalue", "adj.P.Val"])
    #round log2FC, AveExpr, t, B, -log10_adj_pval to 2 decimals
    aging_genes_df["log2FC"] = aging_genes_df["log2FC"].round(2)
    aging_genes_df["AveExpr"] = aging_genes_df["AveExpr"].round(2)
    aging_genes_df["t"] = aging_genes_df["t"].round(2)
    aging_genes_df["B"] = aging_genes_df["B"].round(2)
    aging_genes_df["-log10_adj_pval"] = aging_genes_df["-log10_adj_pval"].round(2)

    # Configure and display AgGrid

    # Apply search filtering to the data
    aging_genes_df = add_searchbar_to_aggrid(
        aging_genes_df, key_prefix=f"{key_prefix}_marker"
    )

    grid_options = configure_grid_options(aging_genes_df, key_prefix)
    grid_response = AgGrid(
        aging_genes_df,
        gridOptions=grid_options,
        height=600,
        width="100%",
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        key=f"{key_prefix}_grid",
    )

    filtered_data = grid_response["data"]
    st.info(f"Showing {len(filtered_data)} of {len(aging_genes_df)} genes")

    st.download_button(
        label="Download Aging Genes Data",
        data=filtered_data.to_csv(index=False),
        file_name="aging_genes_data.csv",
        mime="text/csv",
        key="download_aging_genes",
        help="Download the current filtered aging genes dataset",
    )

    return filtered_data


def display_curation_table(curation_data, key_prefix=""):
    """
    Display an interactive curation table using AgGrid
    """
    try:
        st.info(
            "We are humans too. Did we get something wrong? Did we miss your dataset?\n"
            "Please let us know by supplying the relevant information through email.\n"
        )

        # Configure and display AgGrid

        curation_data = add_searchbar_to_aggrid(
            curation_data, key_prefix=f"{key_prefix}_marker"
        )

        grid_options = configure_grid_options(curation_data, key_prefix)
        grid_response = AgGrid(
            curation_data,
            gridOptions=grid_options,
            height=800,
            width="100%",
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid",
        )

        filtered_data = grid_response["data"]
        st.info(f"Showing {len(filtered_data)} of {len(curation_data)} entries")

        st.download_button(
            label="Download Curated Metadata",
            data=filtered_data.to_csv(index=False),
            file_name="curated_metadata.csv",
            mime="text/csv",
            help="Download the current filtered dataset",
            key=f"{key_prefix}_download_curation",
        )

        # Display metadata column explanations
        st.markdown(
            """
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
        - **Modality**: Type of data
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
                    
        """
        )

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
        display_df["Interaction"] = (
            display_df["ligand_complex"] + " → " + display_df["receptor_complex"]
        )
        display_df["Cell_Pair"] = display_df["source"] + " → " + display_df["target"]

        # -log10
        display_df["specificity_rank"] = -np.log10(display_df["specificity_rank"] + 1e-300)
        display_df["magnitude_rank"] = -np.log10(display_df["magnitude_rank"] + 1e-300)

        # Select and reorder columns
        display_cols = [
            "Interaction",
            "Cell_Pair",
            "source",
            "target",
            "ligand_complex",
            "receptor_complex",
            "specificity_rank",
            "magnitude_rank",
        ]
        display_df = display_df[display_cols]

        # Rename columns for display
        column_names = {
            "source": "Source Cell Type",
            "target": "Target Cell Type",
            "ligand_complex": "Ligand",
            "receptor_complex": "Receptor",
            "specificity_rank": "-log10 Specificity",
            "magnitude_rank": "-log10 Magnitude",
        }
        display_df = display_df.rename(columns=column_names)

        # Configure and display AgGrid

        display_df = add_searchbar_to_aggrid(
            display_df, key_prefix=f"{key_prefix}_marker"
        )

        grid_options = configure_grid_options(display_df, key_prefix)
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            height=600,
            width="100%",
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid",
        )

        filtered_data = grid_response["data"]

        st.download_button(
            label="Download Interaction Data",
            data=filtered_data.to_csv(index=False),
            file_name="ligand_receptor_interactions.csv",
            mime="text/csv",
            help="Download the current filtered interaction dataset",
            key=f"{key_prefix}_download",
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

        # turn pvalue to -log10
        # where p.adjust is 0, make it 1e-300
        enrichment_df["p.adjust"] = enrichment_df["p.adjust"].replace(0, 1e-300)
        enrichment_df["-log10_pval"] = -np.log10(enrichment_df["p.adjust"])

        #round to 4 decimals: percent.observed, percent.background, fold.enrichment, -log10_pval
        enrichment_df["percent.observed"] = enrichment_df["percent.observed"].round(4)
        enrichment_df["percent.background"] = enrichment_df["percent.background"].round(4)
        enrichment_df["fold.enrichment"] = enrichment_df["fold.enrichment"].round(4)
        enrichment_df["-log10_pval"] = enrichment_df["-log10_pval"].round(2)

        # remove pval column
        enrichment_df.drop(columns=["pvalue", "p.adjust"], inplace=True)
        # Configure grid options with custom column widths and filtering
        gb = GridOptionsBuilder.from_dataframe(enrichment_df)
        gb.configure_pagination(
            enabled=True, paginationAutoPageSize=False, paginationPageSize=50
        )
        gb.configure_default_column(
            resizable=True, filterable=True, sorteable=True, minWidth=150
        )

        # Enable floating filters and advanced filter features
        gb.configure_grid_options(
            enableRangeSelection=True,
            floatingFilter=True,
            defaultColDef={"floatingFilter": True, "filter": True},
        )

        grid_options = gb.build()

        # Display AgGrid
        grid_response = AgGrid(
            enrichment_df,
            gridOptions=grid_options,
            height=600,
            width="100%",
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid",
        )

        filtered_data = grid_response["data"]
        st.info(f"Showing {len(filtered_data)} of {len(enrichment_df)} results")

        # Add download button
        st.download_button(
            label="Download Enrichment Data",
            data=filtered_data.to_csv(index=False),
            file_name="enrichment_results.csv",
            mime="text/csv",
            help="Download the current filtered enrichment results",
            key=f"{key_prefix}_download",
        )

        st.markdown("### Grouping Definitions")
        st.markdown(
            """
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
                """
        )

        return filtered_data

    except Exception as e:
        st.error(f"Error displaying enrichment table: {str(e)}")
        return None


def display_sex_dimorphism_table(sex_dim_data, key_prefix=""):
    """
    Display and handle sex dimorphism table functionality using AgGrid.
    """
    
    try:
        # Format numeric columns if present
        if 'log2fc' in sex_dim_data.columns:
            sex_dim_data['log2fc'] = sex_dim_data['log2fc'].round(2)
        
        if 'pvalue' in sex_dim_data.columns:
            # -log10 transform p-values for better visualization
            sex_dim_data["-log10_pval"] = -np.log10(sex_dim_data["adj.P.Val"] + 1e-300)
            sex_dim_data["-log10_pval"] = sex_dim_data["-log10_pval"].round(2)

        #logFC round 2
        sex_dim_data['logFC'] = sex_dim_data['logFC'].round(2)
        # round 2 AveExpr, t, B, -log10_pval
        sex_dim_data['AveExpr'] = sex_dim_data['AveExpr'].round(2)
        sex_dim_data['t'] = sex_dim_data['t'].round(2)
        sex_dim_data['B'] = sex_dim_data['B'].round(2)
        sex_dim_data['-log10_pval'] = sex_dim_data['-log10_pval'].round(2)

        #rename occurs to "Occurs in n cell types"
        sex_dim_data = sex_dim_data.rename(
            columns={
                "occurs": "Occurs in n cell types"})
        #remove these cols Unnamed: 0
        sex_dim_data = sex_dim_data.drop(
            columns=[
                "Unnamed: 0"])
        #Move Occurs in n cell types to the end. Move these to the start: cell_type, sex, gene
        cols = sex_dim_data.columns.tolist()
        cols.remove("Occurs in n cell types")
        cols.remove("cell_type")
        cols.remove("sex")
        cols.remove("gene")
        cols = ["cell_type","sex","gene"] + cols + ["Occurs in n cell types"]
        sex_dim_data = sex_dim_data[cols]


        # Add searchbar to filter data
        
        filtered_data = add_searchbar_to_aggrid(
            sex_dim_data, 
            key_prefix=f"{key_prefix}_sex_dim"
        )
        
        # Configure and display AgGrid
        
        grid_options = configure_grid_options(filtered_data, key_prefix)
        
        grid_response = AgGrid(
            filtered_data,
            gridOptions=grid_options,
            height=600,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid_sex_dim"
        )
        
        filtered_result = grid_response['data']
        st.info(f"Showing {len(filtered_result)} of {len(sex_dim_data)} sexually dimorphic genes")
        
        # Download button for the data
        st.download_button(
            label="Download Sexually Dimorphic Genes Data",
            data=filtered_result.to_csv(index=False),
            file_name="sexually_dimorphic_genes.csv",
            mime="text/csv",
            help="Download the current filtered sexually dimorphic genes dataset",
            key=f"{key_prefix}_download_sex_dim"
        )
        
        # Add explanation text
        st.markdown("""
        ### About Sexually Dimorphic Genes
        
        This table shows genes that are differentially expressed between male and female mice in pituitary cell types.
        
        Key metrics:
        - **log2fc**: Log2 fold change between male and female expression (positive values indicate higher expression in males)
        - **-log10_pval**: -log10 transformed p-value (higher values indicate greater statistical significance)
        - **Cell Type**: Pituitary cell type where sexual dimorphism was detected
        
        These genes may be involved in sex-specific functions, reproductive processes, or hormonal regulation.
        """)
        
        return filtered_result
        
    except Exception as e:
        st.error(f"Error loading sexually dimorphic genes data: {str(e)}")
        return None
    


def display_enhancers_table(enhancers_data, key_prefix=""):
    """
    Display and handle nhancers table functionality using AgGrid.
    """
    
    try:
        # Format numeric columns if present
        if 'log2fc' in enhancers_data.columns:
            enhancers_data['log2fc'] = enhancers_data['log2fc'].round(2)
        
        if 'pvalue' in enhancers_data.columns:
            # -log10 transform p-values for better visualization
            enhancers_data["-log10_pval"] = -np.log10(enhancers_data["padj"]+ 1e-300)
            enhancers_data["-log10_pval"] = enhancers_data["-log10_pval"].round(2)
        

        filtered_data = add_searchbar_to_aggrid(
            enhancers_data, 
            key_prefix=f"{key_prefix}_enhancers"
        )
        
        # Configure and display AgGrid
        
        grid_options = configure_grid_options(filtered_data, key_prefix)
        
        grid_response = AgGrid(
            filtered_data,
            gridOptions=grid_options,
            height=600,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_grid_sex_dim"
        )
        
        filtered_result = grid_response['data']
        st.info(f"Showing {len(filtered_result)} of {len(enhancers_data)} enhancers-TF pairs")
        
        # Download button for the data
        st.download_button(
            label="Download Enhancer table",
            data=filtered_result.to_csv(index=False),
            file_name="enhancers.csv",
            mime="text/csv",
            help="Download the current filtered enhancers dataset",
            key=f"{key_prefix}_download_enhancers"
        )
        
        # Add explanation text
        st.markdown("""
        ### About enhancers
        
        This table shows enhancer-TF pairs associated with given target genes.
                    
        Key metrics:
        - **cor**: Correlation between enhancer and target gene expression (positive values indicate higher expression in the same direction)
        - **-log10_pval**: -log10 transformed p-value (higher values indicate greater statistical significance)
        - **TF**: Transcription factor associated with the enhancer
        - **gene**: Target gene associated with the enhancer
        """)
        
        return filtered_result
        
    except Exception as e:
        st.error(f"Error loading sexually dimorphic genes data: {str(e)}")
        return None