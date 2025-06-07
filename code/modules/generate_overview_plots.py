import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

BASE_PATH = Config.BASE_PATH


def load_curation_data(base_path, version="v_0.01"):
    """
    Load curation data from parquet file
    """
    file_path = f"{base_path}/data/curation/{version}/cpa.parquet"
    print(f"Attempting to load curation data from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Curation data file not found at {file_path}")

    return pd.read_parquet(file_path)


def preprocess_data(df):
    """
    Preprocess the curation dataframe for plotting
    """
    # Restrict species to mouse
    df = df[df["species"] == "mouse"]

    # Extract year from Author column
    df["Year"] = df["Author"].str.extract(r"\((.*?)\)")
    df["Year"] = df["Year"].fillna("Unpublished")
    df["Year"] = df["Year"].apply(
        lambda x: "Unpublished" if "unpublished" in x.lower() else x
    )

    # Clean up Sex column
    df["Sex"] = df["Sex"].apply(
        lambda x: "mixed" if isinstance(x, str) and "mixed" in x.lower() else x
    )
    df["Sex"] = df["Sex"].apply(
        lambda x: "unclear" if isinstance(x, str) and "?" in x else x
    )

    # Add RNA/ATAC label for easier filtering
    df["rna_atac"] = df["Modality"].apply(
        lambda x: "ATAC" if x in ["atac", "multi_atac"] else "RNA"
    )

    # Label multiome data
    df.loc[df["Modality"].isin(["multi_atac", "multi_rna"]), "10X version"] = (
        "Multiome"
    )

    # Convert Age_numeric to numeric
    df["Age_numeric"] = pd.to_numeric(
        df["Age_numeric"].astype(str).str.replace(",", "."), errors="coerce"
    )

    return df


def create_broken_axis_histogram(data, modality, n_bins=30, output_path=None):
    """
    Create histogram with broken y-axis to handle peaks of different magnitudes

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data
    modality : str
        'RNA' or 'ATAC' - used for filtering and title
    n_bins : int
        Number of bins for histogram
    output_path : str, optional
        Path to save the figure
    """
    # Filter based on modality
    if modality == "RNA":
        df_filtered = data[data["Modality"].isin(["sc", "sn", "multi_rna"])]
    else:  # ATAC
        df_filtered = data[data["Modality"].isin(["atac", "multi_atac"])]

    # Calculate histogram to find peak heights
    hist, bin_edges = np.histogram(df_filtered["Age_numeric"].dropna(), bins=n_bins)
    max_height = np.max(hist)  # Height of the highest peak

    # Get second highest peak (if there's only one bin, use half of max)
    if len(hist) > 1:
        sorted_hist = np.sort(hist)[::-1]
        second_highest = sorted_hist[1] if len(sorted_hist) > 1 else max_height / 2
    else:
        second_highest = max_height / 2

    # Round to nearest integer
    max_height = int(round(max_height))
    second_highest = int(round(second_highest))

    # Set y-limits based on the actual peak heights
    top_ylim = (
        max(max_height - 5, second_highest + 1),
        max_height + 1,
    )  # Top plot: around the highest peak
    bottom_ylim = (
        0,
        min(second_highest + 5, max_height - 1),
    )  # Bottom plot: up to the second peak + buffer

    # Create figure with 2/3 height/width ratio
    width = 3  # Width in inches
    height = width * (2 / 3)  # Height in inches (2/3 of width)
    fig = plt.figure(figsize=(width, height), dpi=300)

    # Define the heights ratio for the two axes
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 3], hspace=0.05)

    # Create the two axes
    ax1 = fig.add_subplot(gs[0])  # top subplot for high values
    ax2 = fig.add_subplot(gs[1])  # bottom subplot for the main distribution

    # Create histograms on both axes with the same number of bins
    sns.histplot(
        data=df_filtered,
        x="Age_numeric",
        kde=False,
        color="#4C72B0",
        ax=ax1,
        bins=n_bins,
    )
    sns.histplot(
        data=df_filtered,
        x="Age_numeric",
        kde=False,
        color="#4C72B0",
        ax=ax2,
        bins=n_bins,
    )

    # Set the y-limits for each axis
    ax1.set_ylim(top_ylim)
    ax2.set_ylim(bottom_ylim)

    # Hide the x-axis labels and ticks of the top subplot
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_xlabel("")

    # Only show y-ticks that are within the y-limits for each axis, skipping the first tick on the top plot
    ax1_ticks = [y for y in ax1.get_yticks() if top_ylim[0] <= y <= top_ylim[1]]
    ax1.set_yticks(ax1_ticks[1:] if len(ax1_ticks) > 1 else ax1_ticks)

    ax2_ticks = [y for y in ax2.get_yticks() if bottom_ylim[0] <= y <= bottom_ylim[1]]
    ax2.set_yticks(ax2_ticks)

    # make sure all y ticks are integers, if not find the closest integer
    ax1.set_yticklabels(
        [int(y) if y.is_integer() else int(round(y)) for y in ax1.get_yticks()]
    )
    ax2.set_yticklabels(
        [int(y) if y.is_integer() else int(round(y)) for y in ax2.get_yticks()]
    )

    # Customize the bottom subplot
    ax2.set_xlabel("Age (days)", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)

    # Add ylabel to the first subplot but make it empty (to align with the bottom plot)
    ax1.set_ylabel("")

    # Add title to the entire figure
    fig.suptitle(
        f"Distribution of Age ({modality})", fontsize=8, fontweight="bold", y=0.98
    )

    # Set tick sizes for both axes
    ax1.tick_params(axis="both", which="major", labelsize=6)
    ax2.tick_params(axis="both", which="major", labelsize=6)

    # Remove top and right spines for both subplots
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add the break symbols
    d = 0.015  # Size of the diagonal lines
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved {modality} age distribution to {output_path}")

    plt.close(fig)
    return fig


def create_cumulative_plots(df, base_path, version):
    """Create combined cumulative cell count plot for RNA and ATAC data"""
    # Set font to Arial
    plt.rcParams["font.family"] = "Arial"

    # Prepare data for cumulative plots
    df_rna = df[df["rna_atac"] == "RNA"].copy()
    df_atac = df[df["rna_atac"] == "ATAC"].copy()

    # Convert year to numeric, handling unpublished data
    df_rna["Year"] = pd.to_numeric(
        df_rna["Year"].replace("Unpublished", "2024"), errors="coerce"
    )
    df_atac["Year"] = pd.to_numeric(
        df_atac["Year"].replace("Unpublished", "2024"), errors="coerce"
    )

    # Process RNA data
    rna_yearly = (
        df_rna.groupby("Year")
        .agg({"n_cells": "sum", "Author": "nunique", "SRA_ID": "nunique"})
        .reset_index()
    )
    rna_yearly = rna_yearly.sort_values("Year")
    rna_yearly["Cumulative_N_cell"] = rna_yearly["n_cells"].cumsum()

    # Process ATAC data
    atac_yearly = (
        df_atac.groupby("Year")
        .agg({"n_cells": "sum", "Author": "nunique", "SRA_ID": "nunique"})
        .reset_index()
    )
    atac_yearly = atac_yearly.sort_values("Year")
    atac_yearly["Cumulative_N_cell"] = atac_yearly["n_cells"].cumsum()

    # Normalize author counts for scatter plot sizes
    max_authors_rna = max(rna_yearly["Author"].max(), 1)  # Avoid division by zero
    sizes_rna = (rna_yearly["Author"] / max_authors_rna) * 100  # Scale to 0-100 range

    max_authors_atac = max(atac_yearly["Author"].max(), 1)  # Avoid division by zero
    sizes_atac = (
        atac_yearly["Author"] / max_authors_atac
    ) * 100  # Scale to 0-100 range

    # Plot both datasets
    plt.figure(figsize=(5, 4), dpi=300)
    sns.set_style("whitegrid")

    # RNA plot (blue)
    plt.plot(
        rna_yearly["Year"],
        rna_yearly["Cumulative_N_cell"],
        "-",
        color="#0000ff",
        label="RNA",
    )
    scatter_rna = plt.scatter(
        rna_yearly["Year"],
        rna_yearly["Cumulative_N_cell"],
        s=sizes_rna,
        color="#0000ff",
        alpha=0.7,
    )

    # ATAC plot (magenta)
    plt.plot(
        atac_yearly["Year"],
        atac_yearly["Cumulative_N_cell"],
        "-",
        color="#ff2eff",
        label="ATAC",
    )
    scatter_atac = plt.scatter(
        atac_yearly["Year"],
        atac_yearly["Cumulative_N_cell"],
        s=sizes_atac,
        color="#ff2eff",
        alpha=0.7,
    )

    plt.title(
        "Cumulative cells over the years (RNA vs ATAC)", fontsize=12, weight="bold"
    )
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Cumulative cells", fontsize=11)

    # Ensure y-axis values are displayed as integers
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Add legend
    plt.legend(fontsize=8)

    # Add total papers and cells to the plot for both datasets
    rna_total_samples = int(rna_yearly["SRA_ID"].sum())
    rna_total_cells = int(rna_yearly["n_cells"].sum())
    rna_total_papers = int(rna_yearly["Author"].sum())

    atac_total_samples = int(atac_yearly["SRA_ID"].sum())
    atac_total_cells = int(atac_yearly["n_cells"].sum())
    atac_total_papers = int(atac_yearly["Author"].sum())

    text_info = (
        f"RNA:\n"
        f"  Total papers: {rna_total_papers}\n"
        f"  Total samples: {rna_total_samples}\n"
        f"  Total cells: {rna_total_cells:,}\n\n"
        f"ATAC:\n"
        f"  Total papers: {atac_total_papers}\n"
        f"  Total samples: {atac_total_samples}\n"
        f"  Total cells: {atac_total_cells:,}"
    )

    plt.text(
        0.02,
        0.98,
        text_info,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save the combined plot
    combined_path = os.path.join(
        base_path,
        "data",
        "figures",
        version,
        "cumulative_ncell_over_years_combined.png",
    )
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")

    # Also save as SVG for better quality
    combined_svg_path = os.path.join(
        base_path,
        "data",
        "figures",
        version,
        "cumulative_ncell_over_years_combined.svg",
    )
    plt.savefig(combined_svg_path, bbox_inches="tight")

    plt.close()

    print(f"Saved combined cumulative plot to {combined_path}")
    print(
        f"RNA summary: {rna_total_papers} papers, {rna_total_samples} samples, {rna_total_cells:,} cells"
    )
    print(
        f"ATAC summary: {atac_total_papers} papers, {atac_total_samples} samples, {atac_total_cells:,} cells"
    )


def generate_overview_plots(base_path, version="v_0.01"):
    """
    Generate all overview plots and save them to the figures directory

    Parameters:
    -----------
    base_path : str
        Base path to data directory
    version : str
        Version string (default: "v_0.01")
    """
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(base_path, "data", "figures", version)
    os.makedirs(figures_dir, exist_ok=True)

    # Load and preprocess data
    print(f"Loading curation data...")
    curation_data = load_curation_data(base_path, version)
    df = preprocess_data(curation_data)

    # Generate and save age distribution plots
    print(f"Generating age distribution plots...")
    rna_output_path = os.path.join(figures_dir, "age_distribution_histogram_small.svg")
    create_broken_axis_histogram(df, "RNA", output_path=rna_output_path)
    #also png
    rna_output_png_path = os.path.join(
        figures_dir, "age_distribution_histogram_small.png"
    )
    create_broken_axis_histogram(
        df, "RNA", output_path=rna_output_png_path
    )

    atac_output_path = os.path.join(
        figures_dir, "age_distribution_histogram_small_atac.svg"
    )
    create_broken_axis_histogram(df, "ATAC", output_path=atac_output_path)
    #also png
    atac_output_png_path = os.path.join(
        figures_dir, "age_distribution_histogram_small_atac.png"
    )
    create_broken_axis_histogram(
        df, "ATAC", output_path=atac_output_png_path
    )

    # Generate and save cumulative plots (now a combined plot)
    print(f"Generating combined cumulative plot...")
    create_cumulative_plots(df, base_path, version)

    print(f"All overview plots have been generated and saved to {figures_dir}")


if __name__ == "__main__":
    # Import configuration to get BASE_PATH
    # Try different approaches to find the config module
    config_found = False
    from config import Config

    BASE_PATH = Config.BASE_PATH

    # Set available versions
    AVAILABLE_VERSIONS = ["v_0.01"]

    # Generate plots for all available versions
    for version in AVAILABLE_VERSIONS:
        generate_overview_plots(BASE_PATH, version)
