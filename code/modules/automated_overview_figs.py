"""Regenerate overview figures from the curation table for post-legacy releases.

``v_0.01`` and ``v_0.02`` figures are frozen artefacts of the original
publication, so they are skipped: only newer curation releases are redrawn.

Run from anywhere::

    python code/modules/automated_overview_figs.py            # all eligible versions
    python code/modules/automated_overview_figs.py v_0.03     # one version
"""

import os
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

plt.rcParams["font.family"] = "Arial"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from modules.versioning import sorted_versions  # noqa: E402

# Figures for these versions shipped with the original publication; leave them be.
FROZEN_FIGURE_VERSIONS = {"v_0.01", "v_0.02"}

CUMULATIVE_STEM = "cumulative_ncell_over_years_combined"
AGE_HISTOGRAM_STEMS = {
    "RNA": "age_distribution_histogram_small",
    "ATAC": "age_distribution_histogram_small_atac",
}

RNA_MODALITIES = ["sc", "sn", "multi_rna"]
ATAC_MODALITIES = ["atac", "multi_atac"]

RNA_COLOR = "#0000ff"
ATAC_COLOR = "#ff2eff"

AGE_BINS = 30


def eligible_versions(base_path):
    """Curation versions that should have freshly generated figures."""
    curation_root = Path(base_path) / "data" / "curation"
    found = [
        path.name
        for path in curation_root.iterdir()
        if path.is_dir()
        and path.name.startswith("v_")
        and (path / "cpa.parquet").exists()
        and path.name not in FROZEN_FIGURE_VERSIONS
    ]
    return sorted_versions(found, reverse=False)


def load_curation(base_path, version):
    """Curation rows for the mouse atlas, labelled by modality and publication year."""
    path = Path(base_path) / "data" / "curation" / version / "cpa.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Curation data not found at {path}")

    df = pd.read_parquet(path)
    df = df[df["species"] == "mouse"].copy()

    # Author strings are usually "Name et al. (2021)" but not always parenthesised,
    # so match a bare four-digit year anywhere in the string.
    df["Year"] = pd.to_numeric(
        df["Author"].str.extract(r"(\d{4})", flags=re.ASCII)[0], errors="coerce"
    )
    undated = int(df["Year"].isna().sum())
    if undated:
        # Unpublished/undated entries belong to the newest year rather than being
        # dropped, which would silently remove their cells from the totals.
        latest = df["Year"].max()
        print(f"  {undated} rows without a year in Author; counting them as {latest:.0f}")
        df["Year"] = df["Year"].fillna(latest)

    missing_cells = int(df["n_cells"].isna().sum())
    if missing_cells:
        print(f"  {missing_cells} rows without n_cells; counted as 0 cells")
    df["n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce").fillna(0)

    df["Age_numeric"] = pd.to_numeric(
        df["Age_numeric"].astype(str).str.replace(",", "."), errors="coerce"
    )

    df["rna_atac"] = df["Modality"].apply(
        lambda modality: "ATAC" if modality in ATAC_MODALITIES else "RNA"
    )
    return df


def yearly_cumulative(df):
    """Per-year sample/paper counts with a running cell total."""
    yearly = (
        df.groupby("Year")
        .agg({"n_cells": "sum", "Author": "nunique", "SRA_ID": "nunique"})
        .reset_index()
        .sort_values("Year")
    )
    yearly["Cumulative_N_cell"] = yearly["n_cells"].cumsum()
    return yearly


def _totals_text(df_rna, df_atac):
    lines = []
    for label, frame in (("RNA", df_rna), ("ATAC", df_atac)):
        lines.append(
            f"{label}:\n"
            f"  Total papers: {frame['Author'].nunique()}\n"
            f"  Total samples: {frame['SRA_ID'].nunique()}\n"
            f"  Total cells: {int(frame['n_cells'].sum()):,}"
        )
    return "\n\n".join(lines)


def _save(figures_dir, stem):
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"  wrote {png_path}")
    print(f"  wrote {svg_path}")


def plot_age_distribution(df, figures_dir, modality):
    """Age histogram with a broken y-axis, so one dominant bin cannot flatten the rest."""
    color = ATAC_COLOR if modality == "ATAC" else RNA_COLOR
    subset = df[df["rna_atac"] == modality]
    ages = subset["Age_numeric"].dropna()
    if ages.empty:
        print(f"  no {modality} ages available; skipping age histogram")
        return

    hist, _ = np.histogram(ages, bins=AGE_BINS)
    sorted_hist = np.sort(hist)[::-1]
    max_height = int(round(sorted_hist[0]))
    second_highest = int(
        round(sorted_hist[1]) if len(sorted_hist) > 1 else max_height / 2
    )

    # The top axis frames the tallest bin, the bottom one everything else; keep the
    # two ranges from overlapping when the peaks are close together.
    top_ylim = (max(max_height - 5, second_highest + 1), max_height + 1)
    bottom_ylim = (0, min(second_highest + 5, max_height - 1))

    width = 2.5
    fig = plt.figure(figsize=(width, width * (2 / 4)), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 3], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        sns.histplot(
            data=subset, x="Age_numeric", kde=False, color=color, ax=ax, bins=AGE_BINS
        )

    ax1.set_ylim(top_ylim)
    ax2.set_ylim(bottom_ylim)

    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticks([max_height])

    ax2.set_yticks(
        [
            int(round(y))
            for y in ax2.get_yticks()
            if bottom_ylim[0] <= y <= bottom_ylim[1] and abs(y - round(y)) < 1e-6
        ]
    )
    ax2.set_xlabel("Age (days)", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)

    fig.suptitle(
        f"Distribution of Age ({modality})", fontsize=8, fontweight="bold", y=0.98
    )

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # No tight_layout here: it cannot handle this gridspec (and only warns), while
    # bbox_inches="tight" on save already trims the figure identically.
    print(
        f"  {modality} ages: {len(subset)} samples, "
        f"{ages.min():.0f}-{ages.max():.0f} days, tallest bin {max_height}"
    )
    _save(figures_dir, AGE_HISTOGRAM_STEMS[modality])


def plot_cumulative(df, figures_dir):
    """Cumulative cells per year for RNA and ATAC, dot size scaled by publications."""
    df_rna = df[df["rna_atac"] == "RNA"]
    df_atac = df[df["rna_atac"] == "ATAC"]
    rna_yearly = yearly_cumulative(df_rna)
    atac_yearly = yearly_cumulative(df_atac)

    plt.figure(figsize=(5, 4), dpi=300)
    sns.set_style("whitegrid")

    for yearly, color, label in (
        (rna_yearly, RNA_COLOR, "RNA"),
        (atac_yearly, ATAC_COLOR, "ATAC"),
    ):
        sizes = (yearly["Author"] / max(yearly["Author"].max(), 1)) * 100
        plt.plot(
            yearly["Year"],
            yearly["Cumulative_N_cell"],
            "-",
            color=color,
            label=label,
        )
        plt.scatter(
            yearly["Year"],
            yearly["Cumulative_N_cell"],
            s=sizes,
            color=color,
            alpha=0.7,
        )

    plt.title(
        "Cumulative cells over the years (RNA vs ATAC)", fontsize=12, weight="bold"
    )
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Cumulative cells", fontsize=11)

    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=8)

    plt.text(
        0.02,
        0.98,
        _totals_text(df_rna, df_atac),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    for label, frame in (("RNA", df_rna), ("ATAC", df_atac)):
        print(
            f"  {label}: {frame['Author'].nunique()} papers, "
            f"{frame['SRA_ID'].nunique()} samples, "
            f"{int(frame['n_cells'].sum()):,} cells"
        )
    _save(figures_dir, CUMULATIVE_STEM)


def generate_overview_figs(base_path, version):
    print(f"{version}: loading curation")
    df = load_curation(base_path, version)
    figures_dir = Path(base_path) / "data" / "figures" / version
    plot_age_distribution(df, figures_dir, "RNA")
    plot_age_distribution(df, figures_dir, "ATAC")
    plot_cumulative(df, figures_dir)


def main(argv=None):
    base_path = Config.BASE_PATH
    versions = list(argv or []) or eligible_versions(base_path)
    frozen = [v for v in versions if v in FROZEN_FIGURE_VERSIONS]
    if frozen:
        raise SystemExit(
            f"Refusing to overwrite frozen publication figures for {', '.join(frozen)}"
        )
    if not versions:
        print("No eligible versions found.")
        return
    print(f"Generating overview figures for: {', '.join(versions)}")
    for version in versions:
        generate_overview_figs(base_path, version)


if __name__ == "__main__":
    main(sys.argv[1:])
