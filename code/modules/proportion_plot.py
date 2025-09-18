import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .utils import create_color_mapping

def create_figure(
    prop_df, cell_types, color_map, title, x_values=None, connect_bars=False
):
    data = []

    if connect_bars and x_values is not None:
        # Create continuous lines first
        for cell_type in cell_types:
            cumsum = np.zeros(len(prop_df.columns))
            for ct in cell_types:
                if ct == cell_type:
                    break
                cumsum += prop_df.loc[ct].values

            y_values = cumsum + prop_df.loc[cell_type].values
            data.append(
                go.Scatter(
                    name=f"{cell_type} (trend)",
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line=dict(color=color_map[cell_type], width=1),
                    showlegend=False,
                )
            )

    # Add the bars
    for cell_type in cell_types:
        data.append(
            go.Bar(
                name=cell_type,
                x=x_values if x_values is not None else prop_df.columns,
                y=prop_df.loc[cell_type],
                marker_color=color_map[cell_type],
                hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            )
        )

    layout = {
        "barmode": "stack",
        "title": title,
        "xaxis_title": "log10(Age)" if x_values is not None else "Sample ID",
        "yaxis_title": "Percentage",
        "bargap": 0.1,
        "bargroupgap": 0.05,
        "xaxis": {"tickangle": 45, "automargin": True, "tickfont": {"size": 8}},
        "height": 800,
        "showlegend": True,
        "legend_title": "Cell Types",
        "yaxis": {"range": [0, 100]},
        "hovermode": "x",
        "margin": {"b": 100},
    }

    return go.Figure(data=data, layout=layout)


def process_age_grouped_data(prop_df, meta_subset):
    # Calculate log10 age and round to 2 decimal places
    meta_subset["log10_age"] = np.round(np.log10(meta_subset["Age_numeric"]), 2)

    # Group by log10 age and average proportions
    grouped_props = []
    unique_ages = sorted(meta_subset["log10_age"].unique())

    for age in unique_ages:
        age_samples = meta_subset[meta_subset["log10_age"] == age]["SRA_ID"]
        age_props = prop_df[age_samples].mean(axis=1)
        age_props = age_props * 100 / age_props.sum()  # Normalize to 100%
        grouped_props.append(age_props)

    # Create new DataFrame with log10 ages as columns
    grouped_df = pd.DataFrame(grouped_props).T
    grouped_df.columns = unique_ages

    return grouped_df, unique_ages


import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(y, sigma=0.05):
    return gaussian_filter1d(y, sigma=sigma)


def create_stacked_area_plot(prop_df, cell_types, color_map, ages):
    data = []
    total = prop_df.sum(axis=0)

    for cell_type in cell_types:
        data.append(
            go.Scatter(
                x=ages,
                y=smooth(prop_df.loc[cell_type] / total * 100, sigma=0.05),
                name=cell_type,
                stackgroup="one",
                fillcolor=color_map[cell_type],
                hovertemplate=f"{cell_type}: %{{y:.1f}}%<extra></extra>",
                hoverinfo="name+y",
            )
        )

    fig = go.Figure(data=data)
    fig.update_traces(line=dict(width=0))

    # Add vertical lines for observations
    for age in ages:
        fig.add_shape(
            type="line",
            x0=age,
            x1=age,
            y0=0,
            y1=100,
            line=dict(
                color="black",
                width=1,
                dash="dash",
            ),
            opacity=0.1,
        )

    fig.update_layout(
        title="Cell Type Proportions by Age",
        xaxis_title="log10(Age)",
        yaxis_title="Percentage",
        xaxis={"tickangle": 45, "tickfont": {"size": 8}},
        yaxis={"range": [0, 100]},
        height=800,
        width=1200,
        hovermode="x",
        margin={"b": 100},
    )

    return fig


def create_proportion_plot(
    matrix,
    rows,
    columns,
    meta_data,
    selected_samples=None,
    selected_authors=None,
    only_normal=False,
    only_whole=False,
    group_by_sex=False,
    order_by_age=False,
    show_mean=False,
    use_log_age=False,
    atac_rna="rna",
):  
    #in age numeric, convert , to 0. then convert all to float
    if "Age_numeric" in meta_data.columns:
        meta_data["Age_numeric"] = (
            meta_data["Age_numeric"].astype(str).str.replace(",", ".").astype(float)
        )
        
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    prop_df = pd.DataFrame(matrix, index=rows.iloc[:, 0], columns=columns.iloc[:, 0])
    meta_data["SRA_ID"] = meta_data["SRA_ID"].astype(str)
    meta_data = meta_data.drop_duplicates(subset="SRA_ID")
    if atac_rna == "rna":
        valid_sra_ids = meta_data["SRA_ID"].unique()
    elif atac_rna == "atac":
        # rename cols of prop_df to corresponding GEO value for a given SRA_ID
        prop_df.columns = [
            meta_data[meta_data["GEO"] == sra_id]["SRA_ID"].values[0]
            for sra_id in prop_df.columns
        ]
        valid_sra_ids = meta_data["SRA_ID"].unique()

    print("First printing propdf")
    print(prop_df)

    # Apply filters
    if selected_samples:
        valid_sra_ids = np.intersect1d(
            valid_sra_ids,
            meta_data[meta_data["Name"].isin(selected_samples)]["SRA_ID"].unique(),
        )
    if selected_authors:
        valid_sra_ids = np.intersect1d(
            valid_sra_ids,
            meta_data[meta_data["Author"].isin(selected_authors)]["SRA_ID"].unique(),
        )
    if only_normal:
        valid_sra_ids = np.intersect1d(
            valid_sra_ids, meta_data[meta_data["Normal"] == 1]["SRA_ID"].unique()
        )
    if only_whole:
        valid_sra_ids = np.intersect1d(
            valid_sra_ids, meta_data[meta_data["Sorted"] == 0]["SRA_ID"].unique()
        )

    valid_sra_ids = [sra_id for sra_id in valid_sra_ids if sra_id in prop_df.columns]
    prop_df = prop_df[valid_sra_ids]

    print("Second printing propdf")
    print(prop_df)

    if prop_df.empty:
        return None, None, None, "No data available for the selected filters"

    # Normalize to percentages
    colsums = prop_df.sum(axis=0)
    prop_df = prop_df * 100 / colsums

    meta_subset = meta_data[meta_data["SRA_ID"].isin(valid_sra_ids)]
    cell_types = list(prop_df.index)
    color_map = create_color_mapping(cell_types)

    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": "cell_type_proportions",
            "height": 1000,
            "width": 2000,
            "scale": 2,
        },
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
    }

    if group_by_sex:
        #convert Comp_sex to int
        meta_subset["Comp_sex"] = meta_subset["Comp_sex"].astype(float).astype(int)
        male_subset = meta_subset[meta_subset["Comp_sex"] == 1]
        female_subset = meta_subset[meta_subset["Comp_sex"] == 0]
        male_data = prop_df[
            [id for id in prop_df.columns if id in male_subset["SRA_ID"].values]
        ]
        female_data = prop_df[
            [id for id in prop_df.columns if id in female_subset["SRA_ID"].values]
        ]

        if show_mean:
            if not male_data.empty:
                male_mean = male_data.mean(axis=1)
                male_mean = male_mean * 100 / male_mean.sum()
                male_data = pd.DataFrame(male_mean, columns=["Mean"])

            if not female_data.empty:
                female_mean = female_data.mean(axis=1)
                female_mean = female_mean * 100 / female_mean.sum()
                female_data = pd.DataFrame(female_mean, columns=["Mean"])

        elif use_log_age and order_by_age:
            if not male_data.empty:
                male_data, male_ages = process_age_grouped_data(male_data, male_subset)
                fig_male = create_stacked_area_plot(
                    male_data, cell_types, color_map, male_ages
                )
            else:
                fig_male = None

            if not female_data.empty:
                female_data, female_ages = process_age_grouped_data(
                    female_data, female_subset
                )
                fig_female = create_stacked_area_plot(
                    female_data, cell_types, color_map, female_ages
                )
            else:
                fig_female = None

            return fig_male, fig_female, config, None

        elif order_by_age:
            if not male_data.empty:
                male_ids = male_subset.sort_values("Age_numeric")["SRA_ID"].values
                male_data = male_data[male_ids]

            if not female_data.empty:
                female_ids = female_subset.sort_values("Age_numeric")["SRA_ID"].values
                female_data = female_data[female_ids]

        fig_male = (
            create_figure(
                male_data, cell_types, color_map, "Cell Type Proportions - Male"
            )
            if not male_data.empty
            else None
        )
        fig_female = (
            create_figure(
                female_data, cell_types, color_map, "Cell Type Proportions - Female"
            )
            if not female_data.empty
            else None
        )
        return fig_male, fig_female, config, None

    else:
        if show_mean:
            mean_props = prop_df.mean(axis=1)
            mean_props = mean_props * 100 / mean_props.sum()
            prop_df = pd.DataFrame(mean_props, columns=["Mean"])

        elif use_log_age and order_by_age:
            prop_df, ages = process_age_grouped_data(prop_df, meta_subset)
            fig = create_stacked_area_plot(prop_df, cell_types, color_map, ages)
            return fig, None, config, None

        elif order_by_age:
            ordered_ids = meta_subset.sort_values("Age_numeric")["SRA_ID"].values
            prop_df = prop_df[ordered_ids]

        title = (
            "Mean Cell Type Proportions"
            if show_mean
            else "Cell Type Proportions by Sample"
        )
        fig = create_figure(prop_df, cell_types, color_map, title)
        return fig, None, config, None