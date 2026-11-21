"""Gene label placement for Plotly scatter plots drawn at a fixed pixel size.

Each label chooses one spot from a fixed candidate set (discrete point-feature
labelling, Christensen, Marks & Shieber 1995):

* direct: the text touches its point (8 positions) and no leader is drawn;
* side leader: the text sits beside its point and the leader meets the middle of the
  text's near side; anchored edges snap to shared columns so stacked labels line up;
* short vertical leader: the text sits just above / below and the leader meets the
  middle of its bottom / top edge (labels squeezed against the plot edge).

A spot must keep the label inside the plot and entirely on its point's side of a
vertical divider (e.g. logFC = 0). The layout minimises per-label costs (leader
length and steepness, covered points, leader over points, direction) plus pairwise
costs (overlap, crossing leaders, leaders through other labels, reversed order of
neighbouring labels, minus a bonus for aligned neighbours): greedy start, best
response, simulated annealing with relocation and swap moves, best response again;
labels still overlapping then search all their candidates and, failing that, a wider
area. A fixed seed and move count make the layout reproducible.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

# Approximate Arial advance widths in em; anything not listed uses the class default.
_NARROW = {"I": 0.28, "J": 0.5, "i": 0.22, "j": 0.22, "l": 0.22, "f": 0.28, "t": 0.28,
           "r": 0.33, "-": 0.33, ".": 0.28, ",": 0.28, "'": 0.19, "(": 0.33, ")": 0.33}
_WIDE = {"M": 0.83, "W": 0.94, "m": 0.83, "w": 0.72, "@": 1.0}
LABEL_FONT_FAMILY = "Arial"

# Candidate geometry (px).
GAP_PX = 2.0
DX_PX = np.array([10.0, 16.0, 24.0, 34.0, 46.0, 62.0, 82.0, 106.0, 136.0])
COLUMN_PX = 8.0
DY_STEP_PX = 7.0
DY_MAX_PX = 160.0
MAX_LEADER_PX = 200.0
VERTICAL_MAX_PX = 46.0
VERTICAL_OFFSETS_PX = np.arange(-60.0, 60.1, 10.0)
PAD_PX = 2.0
SIDE_GAP_PX = 3.0

# Per-label costs.
DIRECT_COST = {"out": 0.0, "up": 3.0, "out_up": 2.0, "out_down": 5.0, "down": 7.0,
               "in": 9.0, "in_up": 7.0, "in_down": 11.0}
LEADER_BASE_COST = 14.0
VERTICAL_LEADER_COST = 12.0
STEEP_COST = 2.0
DOWN_LEADER_COST = 4.0
INWARD_COST = 10.0
POINT_COVER_COST = 2.0
ANCHOR_COVER_COST = 400.0
LEADER_OVER_POINT_COST = 0.5
LEADER_OVER_ANCHOR_COST = 40.0

# Pairwise costs.
OVERLAP_COST = 1e4
CROSS_COST = 220.0
THROUGH_COST = 220.0
ORDER_COST = 10.0
ORDER_REACH_PX = 160.0
ALIGN_BONUS = 5.0

# Search.
PAIR_BUDGET = 1e7   # entries of the n x n x K x K pairwise table
K_MAX = 150
SECTORS = 8
RINGS_PX = np.array([20.0, 45.0, 80.0, 130.0])
MAX_ROUNDS = 12
SA_T0, SA_T1 = 200.0, 0.5
SA_MOVES_PER_LABEL = 4000
SA_PATIENCE = 1500
SWAP_FRACTION = 0.3
TIME_CAP_S = 4.0
REFRESH_ROUNDS = 2
REFRESH_FRACTION = 3   # a refresh replaces up to 1/3 of a label's shortlist
REFRESH_EFFORT = 0.4
REFRESH_EFFORT = 0.4
CACHE_SIZE = 64


def text_box_size(text: str, font_size: float, pad: float = 2.0) -> tuple[float, float]:
    """Width and height in px of a single-line label (slightly generous)."""
    width = 0.0
    for ch in str(text):
        if ch in _NARROW:
            width += _NARROW[ch]
        elif ch in _WIDE:
            width += _WIDE[ch]
        elif ch.isupper():
            width += 0.70
        else:
            width += 0.56
    return width * font_size + 2 * pad, font_size * 1.2 + pad


@dataclass(frozen=True)
class PlotFrame:
    """Linear data <-> pixel mapping of the plot area (pixel y grows upwards)."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    width_px: float
    height_px: float

    def to_px(self, x, y) -> np.ndarray:
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        px = (np.asarray(x, dtype=float) - x0) / (x1 - x0) * self.width_px
        py = (np.asarray(y, dtype=float) - y0) / (y1 - y0) * self.height_px
        return np.column_stack([px, py])

    def to_data(self, px) -> np.ndarray:
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        x = x0 + px[:, 0] / self.width_px * (x1 - x0)
        y = y0 + px[:, 1] / self.height_px * (y1 - y0)
        return np.column_stack([x, y])


@dataclass(frozen=True)
class LabelLayout:
    """Label boxes in px. ``xa``: +1 the text's left edge is anchored (it grows right),
    -1 its right edge, 0 its centre. ``ya``: +1 its bottom edge, -1 its top, 0 its middle.
    ``direct`` labels touch their point and get no leader."""

    centers: np.ndarray
    sizes: np.ndarray
    xa: np.ndarray
    ya: np.ndarray
    direct: np.ndarray

    @property
    def anchors(self) -> np.ndarray:
        """The anchored point of each box; leaders meet the box here."""
        return self.centers - np.column_stack([self.xa * self.sizes[:, 0] / 2, self.ya * self.sizes[:, 1] / 2])


def _candidates(anchor, size, side, divider, bounds, r, reach=1.0):
    """Candidate boxes for one label: centres, anchors, direct flags, leader ends, base costs."""
    w, h = size
    px, py = anchor
    out = side if side != 0 else 1.0
    d = r + GAP_PX
    k = 0.75 * d
    rows = []  # (cx, cy, xa, ya, direct, base cost)
    for name, xa, ya, cx, cy in [
        ("out", out, 0, px + out * d, py), ("in", -out, 0, px - out * d, py),
        ("up", 0, 1, px, py + d), ("down", 0, -1, px, py - d),
        ("out_up", out, 1, px + out * k, py + k), ("in_up", -out, 1, px - out * k, py + k),
        ("out_down", out, -1, px + out * k, py - k), ("in_down", -out, -1, px - out * k, py - k),
    ]:
        rows.append((cx + xa * w / 2, cy + ya * h / 2, xa, ya, 1.0, DIRECT_COST[name]))
    dys = np.arange(-DY_MAX_PX * reach, DY_MAX_PX * reach + 1e-9, DY_STEP_PX * reach)
    for g in (out, -out):
        edges = np.unique(np.round((px + g * DX_PX * reach) / COLUMN_PX) * COLUMN_PX)
        edges = edges[g * (edges - px) >= DX_PX[0] - COLUMN_PX / 2]
        for edge in edges:
            dx = abs(edge - px)
            lengths = np.hypot(dx, dys)
            keep = lengths <= MAX_LEADER_PX * reach
            for dy, length in zip(dys[keep], lengths[keep]):
                rows.append((edge + g * w / 2, py + dy, g, 0.0, 0.0,
                             LEADER_BASE_COST + length - r + STEEP_COST * max(0.0, abs(dy) - dx)
                             + (INWARD_COST if g != out else 0.0) + (DOWN_LEADER_COST if dy < 0 else 0.0)))
    for ya in (1.0, -1.0):
        for dy in DX_PX[DX_PX <= VERTICAL_MAX_PX]:
            for ox in VERTICAL_OFFSETS_PX:
                length = np.hypot(ox, dy)
                rows.append((px + ox, py + ya * (dy + h / 2), 0.0, ya, 0.0,
                             LEADER_BASE_COST + VERTICAL_LEADER_COST + length - r
                             + STEEP_COST * max(0.0, abs(ox) - dy) + (DOWN_LEADER_COST if ya < 0 else 0.0)))
    a = np.array(rows, dtype=float)
    c, xa, ya, direct, base = a[:, :2], a[:, 2], a[:, 3], a[:, 4].astype(bool), a[:, 5]
    lo, hi = c - size / 2, c + size / 2
    ok = (lo[:, 0] >= 0) & (hi[:, 0] <= bounds[0]) & (lo[:, 1] >= 0) & (hi[:, 1] <= bounds[1])
    # Text growing towards the divider keeps a full label width of room, so it stays
    # on its side even when the chart is drawn narrower.
    if side > 0:
        ok &= (lo[:, 0] >= divider + SIDE_GAP_PX) & ~((xa < 0) & (lo[:, 0] - divider < w))
    elif side < 0:
        ok &= (hi[:, 0] <= divider - SIDE_GAP_PX) & ~((xa > 0) & (divider - hi[:, 0] < w))
    ends = c - np.column_stack([xa * w / 2, ya * h / 2])
    return dict(c=c[ok], x=xa[ok], y=ya[ok], d=direct[ok], t=ends[ok], u=base[ok])


class _Density:
    """Summed-area table of point counts, so any box count costs four lookups."""

    def __init__(self, points, bounds, cell=2.0):
        self.cell = cell
        self.nx = int(np.ceil(bounds[0] / cell)) + 2
        self.ny = int(np.ceil(bounds[1] / cell)) + 2
        grid = np.zeros((self.nx, self.ny))
        if len(points):
            ix = np.clip((points[:, 0] / cell).astype(int), 0, self.nx - 1)
            iy = np.clip((points[:, 1] / cell).astype(int), 0, self.ny - 1)
            np.add.at(grid, (ix, iy), 1.0)
        self.sat = np.zeros((self.nx + 1, self.ny + 1))
        self.sat[1:, 1:] = grid.cumsum(0).cumsum(1)

    def count(self, lo, hi):
        x0 = np.clip(np.floor(lo[..., 0] / self.cell).astype(int), 0, self.nx)
        y0 = np.clip(np.floor(lo[..., 1] / self.cell).astype(int), 0, self.ny)
        x1 = np.clip(np.ceil(hi[..., 0] / self.cell).astype(int), 0, self.nx)
        y1 = np.clip(np.ceil(hi[..., 1] / self.cell).astype(int), 0, self.ny)
        s = self.sat
        return s[x1, y1] - s[x0, y1] - s[x1, y0] + s[x0, y0]

    def along(self, a, b, radius):
        """Approximate number of points within ``radius`` of each segment a -> b."""
        lengths = np.hypot(*(b - a).T)
        n = max(2, int(np.ceil(lengths.max() / (2 * radius))) + 1) if len(lengths) else 2
        t = np.linspace(0, 1, n)
        pts = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
        step = np.maximum(lengths / (n - 1), 1e-9)
        return self.count(pts - radius, pts + radius).sum(axis=1) * np.minimum(1.0, step / (2 * radius))


def _in_boxes(points, centers, size, pad=0.0):
    if len(points) == 0:
        return np.zeros((len(centers), 0), dtype=bool)
    return (np.abs(points[None, :, :] - centers[:, None, :]) < size / 2 + pad).all(axis=2)


def _near_segments(points, a, b, radius):
    if len(points) == 0 or len(a) == 0:
        return np.zeros((len(a), len(points)), dtype=bool)
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    t = ((points[None, :, :] - a[:, None, :]) * ab[:, None, :]).sum(axis=2) / l2[:, None]
    proj = a[:, None, :] + np.clip(t, 0.0, 1.0)[..., None] * ab[:, None, :]
    return ((points[None, :, :] - proj) ** 2).sum(axis=2) < radius ** 2


def _segments_cross(a0, a1, b0, b1):
    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                       - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    return (orient(a0, a1, b0) * orient(a0, a1, b1) < 0) & (orient(b0, b1, a0) * orient(b0, b1, a1) < 0)


def _segment_hits_box(a, b, lo, hi):
    """Whether segment a -> b passes through the open box [lo, hi] (Liang-Barsky)."""
    d = b - a
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1], lo.shape[:-1], hi.shape[:-1])
    t0, t1 = np.zeros(shape), np.ones(shape)
    for k in range(2):
        dk = np.where(np.abs(d[..., k]) < 1e-9, 1e-9, d[..., k])
        ta = (lo[..., k] - a[..., k]) / dk
        tb = (hi[..., k] - a[..., k]) / dk
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))
    return t0 < t1 - 1e-6


def _pair_costs(pa, sa, ca, side_a, pb, sb, cb, side_b, divider):
    """Cost of every candidate of label a (rows) against every candidate of label b.

    ``p`` anchor point, ``s`` box size, ``c`` candidate dict, ``side`` point side.
    """
    xa, xb = ca["c"][:, None, :], cb["c"][None, :, :]
    overlap = (np.abs(xa - xb) < (sa + sb) / 2 + PAD_PX).all(axis=2)
    lead_a, lead_b = ~ca["d"][:, None], ~cb["d"][None, :]
    ta, tb = ca["t"][:, None, :], cb["t"][None, :, :]
    cross = _segments_cross(pa, ta, pb, tb) & lead_a & lead_b
    through = (_segment_hits_box(pa, ta, xb - sb / 2, xb + sb / 2) & lead_a) | (
        _segment_hits_box(pb, tb, xa - sa / 2, xa + sa / 2) & lead_b)
    cost = OVERLAP_COST * overlap + CROSS_COST * cross + THROUGH_COST * through
    if (np.abs(pa - pb) < ORDER_REACH_PX).all():
        # Neighbouring labels keep the vertical (significance) order of their points
        # and, on the same side, the outward (effect size) order.
        ddy = pa[1] - pb[1]
        flips = ((abs(ddy) > 3.0) & (ddy * (xa[..., 1] - xb[..., 1]) < 0)).astype(float)
        if side_a != 0 and side_a == side_b:
            ddx = abs(pa[0] - divider) - abs(pb[0] - divider)
            if abs(ddx) > 3.0:
                flips += ddx * (np.abs(xa[..., 0] - divider) - np.abs(xb[..., 0] - divider)) < 0
        cost += ORDER_COST * flips
    ga, gb = ca["x"][:, None], cb["x"][None, :]
    aligned = (lead_a & lead_b & (ga == gb) & (ga != 0) & (np.abs(ta[..., 0] - tb[..., 0]) < 1.0)
               & (np.abs(xa[..., 1] - xb[..., 1]) < 1.6 * max(sa[1], sb[1])))
    return cost - ALIGN_BONUS * aligned


def _shortlist(f, anchor, k):
    """The cheapest candidate in every direction / distance sector, then the cheapest rest.

    Crowded labels keep far-away free spots that a plain cost ranking would drop.
    """
    order = np.argsort(f["u"], kind="stable")
    off = f["c"] - anchor
    sector = np.floor((np.arctan2(off[:, 1], off[:, 0]) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    ring = np.searchsorted(RINGS_PX, np.hypot(*(f["t"] - anchor).T))
    key = (f["d"].astype(int) * 1000 + (f["x"] + 1) * 100 + (f["y"] + 1) * 30
           + sector * len(RINGS_PX) + ring)[order]
    _, first = np.unique(key, return_index=True)
    picks = order[np.sort(first)][:k]
    rest = np.setdiff1d(order, picks, assume_unique=True)
    rest = rest[np.argsort(f["u"][rest], kind="stable")][:max(0, k - len(picks))]
    keep = np.concatenate([picks, rest])
    return {key_: v[keep] for key_, v in f.items()}


def _solve(anchors, sizes, sides, divider, bounds, others, r, priority, seed):
    t_start = time.perf_counter()
    n = len(anchors)
    idx = np.arange(n)
    dens = _Density(others, bounds)

    def unary(i, f):
        rest = np.delete(anchors, i, axis=0)
        heads = np.repeat(anchors[i][None], len(f["c"]), axis=0)
        half = sizes[i] / 2 + r * 0.6
        return (f["u"] + POINT_COVER_COST * dens.count(f["c"] - half, f["c"] + half)
                + ANCHOR_COVER_COST * _in_boxes(rest, f["c"], sizes[i], r).sum(axis=1)
                + np.where(f["d"], 0.0, LEADER_OVER_POINT_COST * dens.along(heads, f["t"], r * 0.8)
                           + LEADER_OVER_ANCHOR_COST * _near_segments(rest, heads, f["t"], r).sum(axis=1)))

    full = []
    for i in range(n):
        f = _candidates(anchors[i], sizes[i], sides[i], divider, bounds, r)
        if len(f["c"]) == 0:
            # No room at all (tiny chart): sit above the point, clipped into the plot.
            c = np.array([[np.clip(anchors[i, 0], sizes[i, 0] / 2, bounds[0] - sizes[i, 0] / 2),
                           np.clip(anchors[i, 1] + r + sizes[i, 1] / 2, sizes[i, 1] / 2,
                                   bounds[1] - sizes[i, 1] / 2)]])
            f = dict(c=c, x=np.zeros(1), y=np.zeros(1), d=np.ones(1, bool), t=c.copy(), u=np.zeros(1))
        f["u"] = unary(i, f)
        full.append(f)

    k = int(np.clip(np.sqrt(PAIR_BUDGET) / n, 12, K_MAX))
    cands = [_shortlist(f, anchors[i], k) for i, f in enumerate(full)]
    counts = [len(c["u"]) for c in cands]
    kmax = max(counts)
    U = np.full((n, kmax), np.inf)
    M = np.zeros((n, n, kmax, kmax), dtype=np.float32)

    def fill(i, js):
        """Per-label costs of label i and its pairwise table rows against labels ``js``."""
        U[i] = np.inf
        U[i, :counts[i]] = cands[i]["u"]
        reach_i = np.abs(cands[i]["c"] - anchors[i]).max(axis=0) + sizes[i]
        for j in js:
            if j == i:
                continue
            reach_j = np.abs(cands[j]["c"] - anchors[j]).max(axis=0) + sizes[j]
            if (np.abs(anchors[i] - anchors[j]) > reach_i + reach_j).any():
                M[i, j] = M[j, i] = 0.0
                continue
            P = _pair_costs(anchors[i], sizes[i], cands[i], sides[i], anchors[j], sizes[j], cands[j],
                            sides[j], divider)
            M[i, j] = M[j, i] = 0.0
            M[i, j, :counts[i], :counts[j]] = P
            M[j, i, :counts[j], :counts[i]] = P.T

    for i in range(n):
        fill(i, range(i + 1, n))

    def context(choice):
        """(n, kmax): pairwise cost of each label's candidates against the others' choices."""
        return M[:, idx, :, choice].sum(axis=0)

    def total(choice):
        return U[idx, choice].sum() + 0.5 * M[idx[:, None], idx[None, :], choice[:, None], choice[None, :]].sum()

    def best_response(choice, order):
        ctx = context(choice)
        for _ in range(MAX_ROUNDS):
            moved = False
            for i in order:
                cost = U[i] + ctx[i]
                b = int(np.argmin(cost))
                if cost[b] < cost[choice[i]] - 1e-9:
                    ctx += M[:, i, :, b] - M[:, i, :, choice[i]]
                    choice[i], moved = b, True
            if not moved:
                break
        return choice

    def anneal(choice, rng, t0, effort=1.0):
        """Simulated annealing from ``choice``. ``ctx`` changes only when a move is
        accepted, so pricing a proposal costs a few lookups."""
        best, best_cost = choice.copy(), total(choice)
        cur, cur_cost = choice.copy(), best_cost
        ctx = context(cur)
        centres = [c["c"] for c in cands]
        neighbours = [np.flatnonzero(np.abs(M[i]).sum(axis=(1, 2)) > 0) for i in range(n)]
        n_moves = int(effort * np.clip(SA_MOVES_PER_LABEL * n, 4000, 200000))
        batch = 4096
        last_gain = 0
        for step in range(n_moves):
            kb = step % batch
            if kb == 0:
                if step - last_gain > effort * SA_PATIENCE * n or time.perf_counter() - t_start > TIME_CAP_S:
                    break
                r_label, r_cand, r_kind, r_nb, r_acc = (
                    rng.integers(0, n, batch), rng.random(batch), rng.random(batch), rng.random(batch),
                    rng.random(batch))
            temp = t0 * (SA_T1 / t0) ** (step / n_moves)
            i = int(r_label[kb])
            ai = int(cur[i])
            nb = neighbours[i]
            if len(nb) and r_kind[kb] < SWAP_FRACTION:
                # Swap: i moves to its spot nearest j's box and j to its spot nearest i's.
                j = int(nb[int(r_nb[kb] * len(nb))])
                bj = int(cur[j])
                a = int(np.argmin(np.abs(centres[i] - centres[j][bj]).sum(axis=1)))
                b = int(np.argmin(np.abs(centres[j] - centres[i][ai]).sum(axis=1)))
                if a == ai and b == bj:
                    continue
                delta = (U[i, a] - U[i, ai] + ctx[i, a] - ctx[i, ai] - (M[i, j, a, bj] - M[i, j, ai, bj])
                         + U[j, b] - U[j, bj] + ctx[j, b] - ctx[j, bj] - (M[j, i, b, ai] - M[j, i, bj, ai])
                         + M[i, j, a, b] - M[i, j, ai, bj])
                if delta < 0 or r_acc[kb] < np.exp(-delta / temp):
                    ctx += M[:, i, :, a] - M[:, i, :, ai]
                    cur[i] = a
                    ctx += M[:, j, :, b] - M[:, j, :, bj]
                    cur[j] = b
                    cur_cost += delta
            else:
                a = int(r_cand[kb] * counts[i])
                if a == ai:
                    continue
                delta = U[i, a] - U[i, ai] + ctx[i, a] - ctx[i, ai]
                if delta < 0 or r_acc[kb] < np.exp(-delta / temp):
                    ctx += M[:, i, :, a] - M[:, i, :, ai]
                    cur[i] = a
                    cur_cost += delta
            if cur_cost < best_cost - 1e-9:
                best, best_cost, last_gain = cur.copy(), cur_cost, step
        return best

    def current(choice):
        return {key: np.array([cands[i][key][choice[i]] for i in range(n)]) for key in cands[0]}

    def context_cost(i, f, pick):
        """Cost of label i's candidates ``f`` against the other labels' spots in ``pick``."""
        cost = f["u"].copy()
        for j in idx[idx != i]:
            other = {key: v[j:j + 1] for key, v in pick.items()}
            cost += _pair_costs(anchors[i], sizes[i], f, sides[i], anchors[j], sizes[j], other,
                                sides[j], divider)[:, 0]
        return cost

    order = idx if priority is None else np.asarray(priority)
    rng = np.random.default_rng(seed)
    choice = np.zeros(n, dtype=int)
    placed = np.zeros(n, dtype=bool)
    for i in order:
        choice[i] = int(np.argmin(U[i] + (M[i, idx, :, choice] * placed[:, None]).sum(axis=0)))
        placed[i] = True
    choice = best_response(anneal(best_response(choice, order), rng, SA_T0), order)

    # Labels still in a hard conflict (overlap, crossing, leader through a label) swap
    # their least useful shortlisted spots for the best of all their candidates given
    # everyone else's spot; then anneal again from the current layout.
    for _ in range(REFRESH_ROUNDS):
        ctx = context(choice)
        stuck = [i for i in range(n) if ctx[i, choice[i]] >= min(CROSS_COST, THROUGH_COST) - 4 * ALIGN_BONUS]
        if not stuck:
            break
        pick = current(choice)
        for i in stuck:
            f, c = full[i], cands[i]
            cost = context_cost(i, f, pick)
            known = {tuple(xy) for xy in np.round(c["c"], 3)}
            fresh = [b for b in np.argsort(cost, kind="stable") if tuple(np.round(f["c"][b], 3)) not in known]
            fresh = fresh[:max(1, counts[i] // REFRESH_FRACTION)]
            spare = [a for a in np.argsort(-c["u"], kind="stable") if a != choice[i]][:len(fresh)]
            for a, b in zip(spare, fresh):
                for key in c:
                    c[key][a] = f[key][b]
        for i in stuck:
            fill(i, idx)
        choice = best_response(anneal(choice, rng, SA_T0 / 4, REFRESH_EFFORT), order)

    # Labels still overlapping another search all their candidates, then a wider area.
    pick = current(choice)
    for _ in range(3):
        changed = False
        for i in range(n):
            rest = idx != i
            if not (np.abs(pick["c"][rest] - pick["c"][i]) < (sizes[rest] + sizes[i]) / 2 + PAD_PX).all(axis=1).any():
                continue
            f = full[i]
            cost = context_cost(i, f, pick)
            if cost.min() >= OVERLAP_COST:
                wide = _candidates(anchors[i], sizes[i], sides[i], divider, bounds, r, reach=2.0)
                if len(wide["c"]):
                    wide["u"] = unary(i, wide)
                    wide_cost = context_cost(i, wide, pick)
                    if wide_cost.min() < cost.min():
                        f, cost = wide, wide_cost
            b = int(np.argmin(cost))
            if not np.allclose(f["c"][b], pick["c"][i]):
                for key in pick:
                    pick[key][i] = f[key][b]
                changed = True
        if not changed:
            break
    return pick


_CACHE: OrderedDict[str, dict] = OrderedDict()


def place_labels(
    anchors: np.ndarray,
    sizes: np.ndarray,
    sides: np.ndarray,
    *,
    divider_px: float,
    bounds_px: tuple[float, float],
    obstacles: np.ndarray | None = None,
    point_radius: float = 5.0,
    priority: np.ndarray | None = None,
    seed: int = 0,
) -> LabelLayout:
    """Label boxes for points ``anchors`` (n x 2 px) with text ``sizes`` (n x 2 px).

    ``sides`` is +1 / -1 for points right / left of ``divider_px`` (0 = either side); a
    label never crosses the divider. ``obstacles`` are the other plotted points, which
    labels and leaders prefer not to cover. ``priority`` (default: as given) is the
    greedy starting order. Identical inputs give the identical layout.
    """
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
    sizes = np.asarray(sizes, dtype=float).reshape(-1, 2)
    sides = np.asarray(sides, dtype=float).reshape(-1)
    others = np.empty((0, 2)) if obstacles is None else np.asarray(obstacles, dtype=float).reshape(-1, 2)
    n = len(anchors)
    if n == 0:
        empty = np.zeros(0)
        return LabelLayout(anchors.copy(), sizes.copy(), empty, empty, np.zeros(0, dtype=bool))
    key = hashlib.sha1(b"".join(np.ascontiguousarray(a).tobytes() for a in (
        anchors, sizes, sides, others, np.array([divider_px, *bounds_px, point_radius, seed], dtype=float),
        np.asarray(priority if priority is not None else [], dtype=float),
    ))).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = _solve(anchors, sizes, sides, float(divider_px), tuple(bounds_px), others,
                             float(point_radius), priority, seed)
        if len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    pick = _CACHE[key]
    return LabelLayout(pick["c"].copy(), sizes, pick["x"].copy(), pick["y"].copy(), pick["d"].copy())
