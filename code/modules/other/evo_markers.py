"""Ancestral reconstruction of cell type marker genes across species.

Markers are binary per species (a gene either is or is not a marker of the cell
type), so ancestral states are reconstructed with a binary Fitch parsimony pass
over a hardcoded taxonomy of the v_0.05 atlas species. Species that were never
assayed stay in the tree but carry no state, and cannot pull an ancestor either
way.
"""

import warnings
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ete3 import Tree

MARKER_COLOR = "#d62728"
OTHER_COLOR = "#cccccc"
MISSING_COLOR = "#4d4d4d"


# --------------------------------------------------------------------- table

def load_dge(path):
    """Load the DGE table and harmonise its two halves.

    Human and mouse were tested with limma-voom (gene / direction / pvalue /
    grouping filled, names empty), every other species with wilcoxon (names /
    pvals_adj filled, gene empty). The table only holds significant hits, so a
    row existing means that gene is a marker in that species.
    """
    res = pd.read_csv(path, index_col=0, low_memory=False)
    res["species"] = res["species"].replace({"mus_musculus": "Mus musculus",
                                             "homo_sapiens": "Homo sapiens"})

    # limma rows carry the gene symbol in `gene`, wilcoxon rows in `names`
    limma = res["species"].isin(["Mus musculus", "Homo sapiens"])
    res.loc[limma, "names"] = res.loc[limma, "gene"]

    # human_symbol names the orthogroup and is 1:1 with gene_tree_id, so all
    # paralogs of a family share it (Nfia/Nfib/Nfix -> NFIA). Rows without one
    # have no ortholog mapping and cannot be compared across species.
    # NB the old notebook overwrote human_symbol with the first human `gene` of
    # each orthogroup and back-filled macaque from `names`; both are dropped
    # here as they rename orthogroups by row order and collide with each other.
    return res.dropna(subset=["human_symbol"])


def load_background(path, gene_col="representative_human_gene_symbol",
                    species_col="species"):
    """Orthogroups present in each species, whether or not they are markers.

    `mapped` is often a proxy genome (mouse ids for sheep and cat, cow ids for
    buffalo). The orthogroup label is `representative_human_gene_symbol`, which
    matches DGE `human_symbol`. Rows with no orthogroup, and the stray Bos
    taurus fragment, are dropped.
    """
    bg = pd.read_csv(path, low_memory=False)
    bg = bg[bg[species_col] != "Bos taurus"]
    return bg.dropna(subset=[gene_col])


# ---------------------------------------------------------------------- tree

# Collapsed NCBI topology for the current Other atlas species only. Internal
# names are the clade labels shown on the plot. Expand this Newick when a
# species is added; do not call NCBI/SQLite at runtime (Streamlit threads).
SPECIES_TREE_NEWICK = (
    "(((((Danio rerio,Ctenopharyngodon idella)Cyprinoidei,"
    "Tachysurus fulvidraco)Otophysi,((Larimichthys crocea,"
    "Gasterosteus aculeatus)Eupercaria,Oryzias latipes)Percomorphaceae)"
    "Clupeocephala,Anguilla japonica)Teleostei,"
    "((((Macaca fascicularis,Homo sapiens)Catarrhini,"
    "(Rattus norvegicus,Mus musculus)Murinae)Euarchontoglires,"
    "(((Ovis aries,Bubalus bubalis)Bovidae,Sus scrofa)Artiodactyla,"
    "(Felis catus,Panthera tigris altaica)Felidae)Laurasiatheria)"
    "Boreoeutheria,Gallus gallus)Amniota)Euteleostomi;"
)


def _label_clades(tree):
    for node in tree.traverse():
        node.sci_name = node.name


def _collapse_unifurcations(tree):
    """Drop unary internals so every remaining clade node actually branches."""
    for node in list(tree.traverse("postorder")):
        if not node.is_root() and not node.is_leaf() and len(node.children) == 1:
            node.delete(prevent_nondicotomic=False)
    while not tree.is_leaf() and len(tree.children) == 1 and not tree.children[0].is_leaf():
        child = tree.children[0]
        grandchildren = list(child.children)
        tree.name = child.name
        child.detach()
        for grandchild in grandchildren:
            tree.add_child(grandchild)
    return tree


def build_species_tree(species, collapse=True):
    """Prune the hardcoded atlas phylogeny to the requested species.

    Names missing from the fixed tree are warned about and dropped. With
    collapse=True, unary internals are removed so each remaining internal node
    is a named clade (Amniota, Boreoeutheria, ...). Tips keep the requested
    scientific names so they match the DGE table.
    """
    requested = list(dict.fromkeys(species))
    full = Tree(SPECIES_TREE_NEWICK, format=8)
    known = {leaf.name for leaf in full}
    unresolved = [name for name in requested if name not in known]
    if unresolved:
        warnings.warn(f"not in the hardcoded atlas phylogeny, dropped: {unresolved}")
    keep = [name for name in requested if name in known]
    if not keep:
        raise ValueError("none of the requested species are in the hardcoded atlas phylogeny")

    tree = full
    tree.prune(keep, preserve_branch_length=False)
    if collapse:
        _collapse_unifurcations(tree)
    _label_clades(tree)
    return tree


def annotate_markers(tree, df, background=None, gene_col="human_symbol",
                     species_col="species", instance_col="names",
                     bg_gene_col="representative_human_gene_symbol",
                     bg_species_col="species"):
    """Attach marker hits, and optionally the orthogroups each species has.

    leaf.markers    orthogroups that are markers in that species
    leaf.instances  orthogroup -> paralogs of it that are markers there
    leaf.has_data   False when the species is absent from this cell type's table
    leaf.orthologs  orthogroups present in the background for that species, or
                    None when no background was passed

    Fitch only sees True or False for an assayed species. A hit is True even if
    the background has no row for it. Everything else in an assayed species is
    False, including an orthogroup the species simply does not have. That
    distinction is recorded afterwards on leaf.reason, for the figure, and is
    not an input to the parsimony. A species with has_data False should not be
    in the tree; cell_type_tree leaves it out.
    """
    markers = defaultdict(set)
    instances = defaultdict(lambda: defaultdict(set))
    for species, gene, instance in zip(df[species_col], df[gene_col], df[instance_col]):
        if pd.isna(gene):
            continue
        markers[species].add(gene)
        if not pd.isna(instance):
            instances[species][gene].add(instance)

    orthologs = None
    if background is not None:
        orthologs = defaultdict(set)
        for species, gene in zip(background[bg_species_col], background[bg_gene_col]):
            if not pd.isna(gene):
                orthologs[species].add(gene)

    for leaf in tree:
        leaf.has_data = leaf.name in markers
        leaf.markers = markers.get(leaf.name, set())
        leaf.instances = {g: sorted(v) for g, v in instances.get(leaf.name, {}).items()}
        leaf.orthologs = None if orthologs is None else orthologs.get(leaf.name, set())

    no_data = [leaf.name for leaf in tree if not leaf.has_data]
    if no_data:
        warnings.warn(f"no rows in this cell type, left out of the reconstruction: {no_data}")

    return tree


def assayed_species(df, species_col="species"):
    """Species that have marker rows for this cell type."""
    return sorted(df[species_col].dropna().unique())


def cell_type_tree(df, background=None, **kwargs):
    """Phylogeny of the species assayed for this cell type, with markers hung on it.

    Species absent from `df` are not in the tree, so they cannot affect Fitch.
    """
    tree = build_species_tree(assayed_species(df), **kwargs)
    return annotate_markers(tree, df, background=background)


def _tip_states(node, gene):
    """Binary tip state. Only a species that was never assayed is missing."""
    if not node.has_data:
        return None
    return {gene in node.markers}


def _assign_reasons(tree, gene):
    """Why a tip is negative. Not used by the parsimony."""
    for node in tree.traverse():
        if not node.is_leaf():
            node.reason = None
            continue
        if not node.has_data:
            node.reason = "no data"
        elif gene in node.markers:
            node.reason = "marker"
        elif getattr(node, "orthologs", None) is not None and gene not in node.orthologs:
            node.reason = "orthogroup missing"
        else:
            node.reason = "not significant"


# --------------------------------------------------------------------- fitch

def fitch(tree, gene, root_state=False):
    """Binary Fitch parsimony for one gene.

    Every assayed tip is True or False. Not a marker is False whether the
    orthogroup was tested and not significant or is absent from that species.
    A tip is skipped only when the whole species was not assayed. Downpass: a
    node keeps the states held by most of its children and costs one change per
    child lacking them. For two children that is Fitch's intersection-or-union;
    the majority rule is also correct for polytomies.

    Uppass: a node keeps its parent's state whenever that state is in its own
    set. In the binary case that resolves every node below the root. Only the
    root can stay ambiguous, and Fitch (1971) leaves that choice arbitrary, so
    `root_state` fixes it (default False).

    Sets node.state, node.change, node.reason and tree.cost. `reason` is filled
    in after the reconstruction and does not feed back into it.
    """
    cost = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.down = _tip_states(node, gene)
            continue
        sets = [c.down for c in node.children if c.down is not None]
        if not sets:
            node.down = None
            continue
        votes = {state: sum(state in s for s in sets) for state in (False, True)}
        best = max(votes.values())
        node.down = {state for state, v in votes.items() if v == best}
        cost += len(sets) - best

    for node in tree.traverse("preorder"):
        if node.down is None:
            node.state = node.change = None
        elif node.is_root():
            node.state = root_state if len(node.down) > 1 else next(iter(node.down))
            node.change = None
        else:
            parent = node.up.state
            # a missing parent is not False, and the branch to it is not a change
            if parent is None:
                node.state = root_state if len(node.down) > 1 else next(iter(node.down))
                node.change = None
            else:
                node.state = parent if parent in node.down else next(iter(node.down))
                node.change = ("gain" if node.state and not parent else
                               "loss" if parent and not node.state else None)

    tree.cost = cost
    _assign_reasons(tree, gene)
    return tree


def fitch_all(tree, genes=None, root_state=False):
    """Run fitch() for every gene and store the result in node.states[gene]."""
    if genes is None:
        genes = sorted(set().union(*[leaf.markers for leaf in tree]))

    for node in tree.traverse():
        node.states = {}
    for gene in genes:
        fitch(tree, gene, root_state=root_state)
        for node in tree.traverse():
            node.states[gene] = node.state

    return tree


def ancestral_table(tree, genes=None, root_state=False):
    """Fitch states for every gene as a genes x nodes frame, pd.NA where missing."""
    fitch_all(tree, genes, root_state=root_state)
    table = pd.DataFrame({n.name: n.states for n in tree.traverse("preorder")})
    return table.astype("boolean")


# ---------------------------------------------------------------------- plot

def _node_color(node):
    """Colour from the reconstruction. Orthogroup-missing tips are marked later."""
    if node.is_leaf() and getattr(node, "reason", None) == "orthogroup missing":
        return MISSING_COLOR
    return (MISSING_COLOR if node.state is None else
            MARKER_COLOR if node.state else OTHER_COLOR)


def _layout(tree):
    """Rectangular cladogram coordinates, tips aligned on the right."""
    tree.depth = 0
    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.depth = node.up.depth + 1

    for i, leaf in enumerate(tree.get_leaves()):
        leaf.y = -float(i)

    xmax = max(leaf.depth for leaf in tree)
    for node in tree.traverse("postorder"):
        node.x = float(xmax) if node.is_leaf() else float(node.depth)
        if not node.is_leaf():
            node.y = sum(c.y for c in node.children) / len(node.children)

    return xmax


def plot_gene_tree(tree, gene, figsize=(7, 5), root_state=False, show_clades=True,
                   title=None, ax=None):
    """Plot the tree for one gene, with every reconstructed marker node red.

    Internal nodes follow the Fitch state. A tip that is negative because the
    orthogroup is absent is dark gray and labelled 'orthogroup missing'; a tip
    that was tested and not significant stays light gray. Both entered Fitch as
    False. Marker tips list the paralogs that are hits.
    """
    fitch(tree, gene, root_state=root_state)
    tree.ladderize()
    xmax = _layout(tree)

    plt.rcParams["font.family"] = "Arial"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # branches as elbows
    for node in tree.traverse():
        for child in node.children:
            ax.plot([node.x, node.x], [node.y, child.y], color="#444444", lw=0.9, zorder=1)
            ax.plot([node.x, child.x], [child.y, child.y], color="#444444", lw=0.9, zorder=1)

    for node in tree.traverse():
        ax.scatter(node.x, node.y, s=70 if node.is_leaf() else 50,
                   facecolor=_node_color(node), edgecolor="white",
                   linewidth=0.7, zorder=3)
        # clade name above and left of the node, clear of both branch lines
        if show_clades and not node.is_leaf():
            ax.text(node.x - 0.07, node.y + 0.09, node.sci_name, fontsize=6,
                    ha="right", va="bottom", color="#555555")

    pad = 0.12
    for leaf in tree:
        if leaf.reason == "orthogroup missing":
            color, sub, sub_color = MISSING_COLOR, "orthogroup missing", MISSING_COLOR
        elif leaf.reason == "no data":
            color, sub, sub_color = MISSING_COLOR, "no data", MISSING_COLOR
        else:
            color = MARKER_COLOR if leaf.state else "#2c2c2c"
            sub = ", ".join(leaf.instances.get(gene, []))
            sub_color = MARKER_COLOR if leaf.state else "#999999"
        ax.text(leaf.x + pad, leaf.y, leaf.name, fontsize=9, style="italic",
                va="center", color=color)
        if sub:
            ax.text(leaf.x + pad, leaf.y - 0.33, sub, fontsize=6, va="center",
                    color=sub_color)

    n_marker = sum(leaf.reason == "marker" for leaf in tree)
    n_other = sum(leaf.reason == "not significant" for leaf in tree)
    n_no_og = sum(leaf.reason == "orthogroup missing" for leaf in tree)
    n_absent = sum(leaf.reason == "no data" for leaf in tree)
    entries = [("marker", MARKER_COLOR, n_marker),
               ("not a marker", OTHER_COLOR, n_other),
               ("orthogroup missing", MISSING_COLOR, n_no_og),
               ("no data", MISSING_COLOR, n_absent)]
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=7,
                      label=f"{label} (n={n})")
               for label, c, n in entries if n]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5)

    ax.set_title(title or gene, fontsize=11, fontweight="bold", color="#2c2c2c")
    ax.set_xlim(-1.8 if show_clades else -0.4, xmax + 3.0)
    ax.set_ylim(-len(tree) + 0.2, 0.9)
    ax.axis("off")
    fig.tight_layout()

    return fig, ax
