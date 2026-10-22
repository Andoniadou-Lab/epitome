"""ggrepel-style label placement for Plotly scatter plots drawn at a fixed pixel size.

Labels are placed one by one at the cheapest free spot around their point. Candidates
are rings of growing radius plus slots touching labels already placed nearby; a spot
must keep the label inside the plot, off other labels and labelled points, and
entirely on the same side of a vertical divider (e.g. logFC = 0) as its point.

The cost of a spot is its gap to the point (outward horizontal distance counts half),
plus penalties for leaning away from "outward and slightly up", for sitting on the
inner side of or below the point, for reversing the vertical (significance) or
outward (effect size) order of neighbouring labels, for covering other points, and
for leader lines that cross labels, points or other leaders. Greedy placement
depends on the order, so a few orders are tried and the cheapest layout is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Approximate Arial advance widths in em; anything not listed uses the class default.
_NARROW = {"I": 0.28, "J": 0.5, "i": 0.22, "j": 0.22, "l": 0.22, "f": 0.28, "t": 0.28,
           "r": 0.33, "-": 0.33, ".": 0.28, ",": 0.28, "'": 0.19, "(": 0.33, ")": 0.33}
_WIDE = {"M": 0.83, "W": 0.94, "m": 0.83, "w": 0.72, "@": 1.0}
LABEL_FONT_FAMILY = "Arial"

MIN_LEADER_PX = 8.0
INNER_FRACTION = 0.4
RADII_PX = (MIN_LEADER_PX, 12.0, 16.0, 21.0, 27.0, 34.0, 42.0, 51.0, 61.0, 72.0, 85.0,
            100.0, 118.0, 140.0, 165.0, 195.0, 230.0)
N_ANGLES = 48
SLOT_REACH_PX = 250.0
CANDIDATE_BATCH = 64
FULL_REFINE_ROUNDS = 1
CROWD_REACH_PX = 60.0
OTHER_POINT_COST = 1.5
SEGMENT_CROSS_COST = 60.0
CROSS_OWN_TEXT_COST = 10.0
# Preferred offset is outward (away from the divider) and slightly up; going the
# opposite way costs DIRECTION_COST, and a label centred on the inner side of its
# point costs INWARD_COST on top.
DIRECTION_COST = 30.0
UP_WEIGHT = 0.4
INWARD_COST = 45.0
DOWN_COST = 25.0
LINE_CROSS_COST = 40.0
LINE_OVER_POINT_COST = 15.0
# Outward horizontal distance counts this much of vertical distance, so crowded
# labels fan out into the empty space beside a cluster instead of stacking over it.
OUTWARD_DISTANCE_WEIGHT = 0.5
# Per nearby placed label whose vertical (significance) or outward (effect size)
# order is reversed relative to the points.
ORDER_COST = 12.0
ORDER_REACH_PX = 120.0


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


def _candidates(anchor, size, point_radius):
    """Label centres around ``anchor``: the box edge sits ``r`` px beyond the point."""
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    # Start straight above the point and sweep alternately left / right of it.
    order = np.argsort(np.abs(((angles - np.pi / 2) + np.pi) % (2 * np.pi) - np.pi), kind="stable")
    angles = angles[order]
    cos, sin = np.cos(angles), np.sin(angles)
    rows = []
    for r in RADII_PX:
        reach = r + point_radius
        dx = cos * (reach + size[0] / 2)
        dy = sin * (reach + size[1] / 2)
        rows.append(np.column_stack([anchor[0] + dx, anchor[1] + dy]))
    return np.vstack(rows)


def _slot_candidates(anchor, size, side, fixed_c, fixed_s, point_radius, pad):
    """Label centres touching placed labels near ``anchor`` (above, below, beside).

    Crowded labels then stack into tidy columns instead of jumping to the next ring.
    """
    near = (np.abs(fixed_c - anchor) < SLOT_REACH_PX).all(axis=1)
    fc, fs = fixed_c[near], fixed_s[near]
    if len(fc) == 0:
        return np.empty((0, 2))
    step = (fs + size) / 2 + pad + 0.5
    if side > 0:
        aligned_x = fc[:, 0] - fs[:, 0] / 2 + size[0] / 2
    elif side < 0:
        aligned_x = fc[:, 0] + fs[:, 0] / 2 - size[0] / 2
    else:
        aligned_x = fc[:, 0]
    beside_x = anchor[0] + (side or 1) * (point_radius + MIN_LEADER_PX + size[0] / 2)
    xs = [aligned_x, np.full(len(fc), beside_x)]
    rows = []
    for x in xs:
        rows += [np.column_stack([x, fc[:, 1] + step[:, 1]]), np.column_stack([x, fc[:, 1] - step[:, 1]])]
    for dx in (step[:, 0], -step[:, 0]):
        rows += [np.column_stack([fc[:, 0] + dx, fc[:, 1]]),
                 np.column_stack([fc[:, 0] + dx, np.full(len(fc), anchor[1])])]
    return np.vstack(rows)


def _boxes_overlap(centers, size, other_centers, other_sizes, pad):
    """(n_candidates x n_others) boolean overlap of one label size against boxes."""
    if len(other_centers) == 0:
        return np.zeros((len(centers), 0), dtype=bool)
    d = np.abs(centers[:, None, :] - other_centers[None, :, :])
    reach = (size[None, None, :] + other_sizes[None, :, :]) / 2 + pad
    return (d < reach).all(axis=2)


def _segment_crosses(anchor, ends, boxes_c, boxes_s):
    """Whether the straight leader from ``anchor`` to each end passes through a box."""
    if len(boxes_c) == 0:
        return np.zeros(len(ends), dtype=bool)
    t = np.linspace(0.15, 0.85, 6)
    samples = anchor[None, None, :] + (ends[:, None, :] - anchor[None, None, :]) * t[None, :, None]
    d = np.abs(samples[:, :, None, :] - boxes_c[None, None, :, :])
    inside = (d < boxes_s[None, None, :, :] / 2).all(axis=3)
    return inside.any(axis=(1, 2))


def _lines_cross(p0, p1s, q0s, q1s):
    """(n_candidates,) count of leader lines p0->p1s crossing the segments q0s->q1s."""
    if len(q0s) == 0:
        return np.zeros(len(p1s))

    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                       - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    p0 = np.broadcast_to(p0, (len(p1s), 1, 2))
    p1 = p1s[:, None, :]
    q0, q1 = q0s[None, :, :], q1s[None, :, :]
    cross = (orient(p0, p1, q0) * orient(p0, p1, q1) < 0) & (orient(q0, q1, p0) * orient(q0, q1, p1) < 0)
    return cross.sum(axis=1)


def repel_labels(
    anchors: np.ndarray,
    sizes: np.ndarray,
    sides: np.ndarray,
    *,
    divider_px: float,
    bounds_px: tuple[float, float],
    obstacles: np.ndarray | None = None,
    point_radius: float = 5.0,
    side_gap: float = 3.0,
    pad: float = 1.5,
    priority: np.ndarray | None = None,
    max_obstacles: int = 600,
    refine_rounds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Label centres in px and text growth directions for points ``anchors`` (n x 2).

    ``sides`` is +1 / -1 for points right / left of ``divider_px`` (0 = either side);
    a label box never crosses the divider. ``obstacles`` are other points (m x 2) that
    labels prefer not to cover. Labels are placed in ``priority`` order (default: as
    given). If no free spot exists within the largest radius, the spot with the fewest
    overlaps is used, still on the correct side.

    The returned growth is +1 when the text is anchored at its left edge and grows
    right, -1 for the mirror case. It normally points away from the divider.
    """
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
    sizes = np.asarray(sizes, dtype=float).reshape(-1, 2)
    sides = np.asarray(sides, dtype=float).reshape(-1)
    n = len(anchors)
    centers = anchors.copy()
    grows = sides.copy()
    if n == 0:
        return centers, grows
    width, height = bounds_px
    point_box = np.array([2 * point_radius, 2 * point_radius])
    others = (
        np.empty((0, 2)) if obstacles is None else np.asarray(obstacles, dtype=float).reshape(-1, 2)
    )
    if len(others) > max_obstacles:
        nearest = np.min(np.abs(others[:, None, :] - anchors[None, :, :]).max(axis=2), axis=1)
        others = others[np.argsort(nearest)[:max_obstacles]]
    other_sizes = np.repeat(point_box[None, :], len(others), axis=0)
    point_sizes = np.repeat(point_box[None, :], n, axis=0)

    def order_inversions(i, xy, fixed):
        """Order flips with placed labels of nearby points: vertical (significance) and
        outward (effect size, same side only)."""
        if len(fixed) == 0:
            return np.zeros(len(xy))
        fc, fa = centers[fixed], anchors[fixed]
        near = (np.abs(fa - anchors[i]) < ORDER_REACH_PX).all(axis=1)[None, :]
        dot_dy = anchors[i, 1] - fa[:, 1]
        lab_dy = xy[:, None, 1] - fc[None, :, 1]
        inv = (np.abs(dot_dy) > 2.0) & (dot_dy * lab_dy < 0)
        if sides[i] != 0:
            same_side = sides[fixed] == sides[i]
            dot_dx = np.abs(anchors[i, 0] - divider_px) - np.abs(fa[:, 0] - divider_px)
            lab_dx = np.abs(xy[:, None, 0] - divider_px) - np.abs(fc[None, :, 0] - divider_px)
            inv = inv.astype(int) + (same_side & (np.abs(dot_dx) > 2.0) & (dot_dx * lab_dx < 0))
        return (near * inv).sum(axis=1)

    def place(i, fixed, at=None):
        """Best centre, growth and cost for label ``i`` (or the cost of putting it ``at``)."""
        fixed_c, fixed_s = centers[fixed], sizes[fixed]
        size = sizes[i]
        half = size / 2
        xy = np.vstack([
            _candidates(anchors[i], size, point_radius),
            _slot_candidates(anchors[i], size, sides[i], fixed_c, fixed_s, point_radius, pad),
        ]) if at is None else np.asarray(at, dtype=float).reshape(1, 2)
        ok = (
            (xy[:, 0] - half[0] >= 0)
            & (xy[:, 0] + half[0] <= width)
            & (xy[:, 1] - half[1] >= 0)
            & (xy[:, 1] + half[1] <= height)
        )
        if sides[i] > 0:
            ok &= xy[:, 0] - half[0] >= divider_px + side_gap
        elif sides[i] < 0:
            ok &= xy[:, 0] + half[0] <= divider_px - side_gap
        gap = np.clip(np.abs(xy - anchors[i]) - half, 0.0, None)
        dist = np.hypot(gap[:, 0], gap[:, 1])
        ok &= dist >= point_radius + MIN_LEADER_PX - 1e-6
        if not ok.any():
            # The plot edge leaves no room: keep the side, clip into bounds.
            fallback = anchors[i] + np.array([0.0, half[1] + point_radius + MIN_LEADER_PX])
            fallback[0] = np.clip(fallback[0], half[0], width - half[0])
            fallback[1] = np.clip(fallback[1], half[1], height - half[1])
            if sides[i] > 0:
                fallback[0] = max(fallback[0], divider_px + side_gap + half[0])
            elif sides[i] < 0:
                fallback[0] = min(fallback[0], divider_px - side_gap - half[0])
            return fallback, sides[i], 1000.0
        xy, gap = xy[ok], gap[ok]
        side = sides[i]

        # Cheap terms first: distance (outward horizontal distance counts less),
        # direction, and hard collisions with labels / labelled points.
        offset = xy - anchors[i]
        offset /= np.linalg.norm(offset, axis=1, keepdims=True)
        preferred = np.array([side, UP_WEIGHT if side else 1.0])
        preferred /= np.linalg.norm(preferred)
        outward = side * (xy[:, 0] - anchors[i, 0]) > 0
        gap_x = gap[:, 0] * np.where(outward, OUTWARD_DISTANCE_WEIGHT, 1.0)
        hard = (
            _boxes_overlap(xy, size, fixed_c, fixed_s, pad).sum(axis=1)
            + _boxes_overlap(xy, size, anchors, point_sizes, pad).sum(axis=1)
        )
        cheap = (
            np.hypot(gap_x, gap[:, 1])
            + DIRECTION_COST * (1.0 - offset @ preferred) / 2.0
            + INWARD_COST * (side * (xy[:, 0] - anchors[i, 0]) < 0)
            + DOWN_COST * np.clip(-offset[:, 1], 0.0, None)
            + 1000.0 * hard
        )

        def facing(grow):
            # The point faces the anchored end of the text, so the leader reaches the
            # text without running through it.
            if grow > 0:
                return anchors[i][0] <= xy[:, 0] - half[0] + INNER_FRACTION * size[0]
            if grow < 0:
                return anchors[i][0] >= xy[:, 0] + half[0] - INNER_FRACTION * size[0]
            return np.ones(len(xy), dtype=bool)

        # Text growing towards the divider is only allowed with a full label width of
        # room, so it cannot cross even on a chart drawn half as wide.
        room = np.abs(xy[:, 0] - divider_px) - half[0]
        cheap_away = cheap + CROSS_OWN_TEXT_COST * ~facing(side)
        cheap_toward = np.where(
            (side != 0) & (room >= size[0]),
            cheap + 1.0 + CROSS_OWN_TEXT_COST * ~facing(-side),
            np.inf,
        )
        lower = np.minimum(cheap_away, cheap_toward)

        others_labelled = np.flatnonzero(np.arange(n) != i)
        fixed_ends = np.clip(anchors[fixed], fixed_c - fixed_s / 2, fixed_c + fixed_s / 2)

        def extra(idx):
            """The costlier, non-negative terms for candidates ``idx``."""
            c = xy[idx]
            ends = np.clip(anchors[i], c - half, c + half)
            return (
                OTHER_POINT_COST * _boxes_overlap(c, size, others, other_sizes, 0.0).sum(axis=1)
                + SEGMENT_CROSS_COST * _segment_crosses(anchors[i], ends, fixed_c, fixed_s)
                + LINE_OVER_POINT_COST * _segment_crosses(
                    anchors[i], ends, anchors[others_labelled], point_sizes[others_labelled])
                + LINE_CROSS_COST * _lines_cross(anchors[i], ends, anchors[fixed], fixed_ends)
                + ORDER_COST * order_inversions(i, c, fixed)
            )

        # Evaluate the costly terms in batches, cheapest first, until no remaining
        # candidate can beat the best total found.
        by_lower = np.argsort(lower, kind="stable")
        best_total, pick, grow = np.inf, int(by_lower[0]), side
        for start in range(0, len(by_lower), CANDIDATE_BATCH):
            idx = by_lower[start:start + CANDIDATE_BATCH]
            if lower[idx[0]] >= best_total:
                break
            e = extra(idx)
            away, toward = cheap_away[idx] + e, cheap_toward[idx] + e
            total = np.minimum(away, toward)
            j = int(np.argmin(total))
            if total[j] < best_total:
                best_total, pick = total[j], int(idx[j])
                grow = side if away[j] <= toward[j] else -side
        return xy[pick], grow, best_total

    def layout(order):
        centers[:], grows[:] = anchors, sides
        placed = np.zeros(n, dtype=bool)
        for i in order:
            centers[i], grows[i], _ = place(i, np.flatnonzero(placed))
            placed[i] = True
        # Re-place labels now that all others are known: first every label (pulls in
        # labels pushed out by neighbours that later settled elsewhere), then only
        # colliding ones.
        for round_ in range(refine_rounds):
            moved = False
            for i in order:
                rest = np.flatnonzero(np.arange(n) != i)
                if round_ >= FULL_REFINE_ROUNDS and not _boxes_overlap(
                    centers[i:i + 1], sizes[i], centers[rest], sizes[rest], pad
                ).any():
                    continue
                new, grows[i], _ = place(i, rest)
                moved |= not np.allclose(new, centers[i])
                centers[i] = new
            if not moved:
                break
        score = sum(place(i, np.flatnonzero(np.arange(n) != i), at=centers[i])[2] for i in range(n))
        return centers.copy(), grows.copy(), score

    # Greedy placement depends on the order, so try a few and keep the cheapest layout:
    # by priority, outermost points first, and most crowded points first.
    crowding = (np.abs(anchors[:, None, :] - anchors[None, :, :]) < CROWD_REACH_PX).all(axis=2).sum(axis=1)
    orders = [
        np.arange(n) if priority is None else np.asarray(priority),
        np.argsort(-np.abs(anchors[:, 0] - divider_px), kind="stable"),
        np.argsort(-crowding, kind="stable"),
    ]
    best = min((layout(order) for order in orders), key=lambda result: result[2])
    return best[0], best[1]


def anchor_edge_x(centers: np.ndarray, sizes: np.ndarray, grows: np.ndarray) -> np.ndarray:
    """x where each label's text starts (left edge if it grows right; centre for 0)."""
    grows = np.asarray(grows, dtype=float)
    return centers[:, 0] - grows * sizes[:, 0] / 2
