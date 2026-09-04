#!/usr/bin/env python3
"""Standalone full-graph version (all 14 zoodles claims) for comparison against the
zoomed panel (a) used in the paper figure. Same palette/conventions, own sizing since
10 topological generations don't fit the paper figure's compact 2-panel layout."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ACCENT = "#1B6B93"
ACCENT_SOFT = "#DCEAF2"
ERROR_RED = "#B23A48"
SIBLING_GRAY = "#B9B9B4"
SIBLING_FILL = "#EDEDEA"
DEEP_ANC_GRAY = "#8A8A84"
DEEP_ANC_FILL = "#D8D8D3"
DOWNSTREAM_GRAY = "#CFCFCA"
DOWNSTREAM_FILL = "#F5F5F3"
INK = "#1A1A1A"
MUTED = "#6B6B66"

fig, ax = plt.subplots(figsize=(4.2, 8.4))
ax.set_xlim(0.4, 8.8)
ax.set_ylim(-4.4, 8.4)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")
ax.set_title("Full zoodles dependency graph (14 claims)", loc="left", pad=8)


def node(xy, label, sublabel, facecolor, edgecolor, textcolor, r=0.5, sublabel_color=None,
         linestyle="-"):
    circ = plt.Circle(xy, r, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3, zorder=4,
                       linestyle=linestyle)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=7.6, color=textcolor,
            weight="bold", zorder=5)
    if sublabel:
        ax.text(xy[0], xy[1] - r - 0.22, sublabel, ha="center", va="top", fontsize=6.2,
                color=sublabel_color or INK, zorder=5)


def arrow(p_from, p_to, r_from=0.5, r_to=0.5, color=INK):
    p_from, p_to = np.array(p_from), np.array(p_to)
    d = p_to - p_from
    u = d / np.linalg.norm(d)
    start = p_from + u * r_from
    end = p_to - u * (r_to + 0.05)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                  linewidth=1.0, color=color, zorder=3))


# All 14 real nodes, positioned by the graph's own topological generation (10 levels).
pos = {
    9: (3.05, 7.6), 12: (5.75, 7.6),
    5: (4.4, 6.3),
    13: (4.4, 5.0),
    8: (4.4, 3.7),
    11: (4.4, 2.4),
    2: (1.4, 1.1), 3: (3.4, 1.1), 10: (5.5, 1.1), 7: (7.9, 1.1),
    1: (5.5, -0.2),
    6: (5.5, -1.5),
    4: (5.5, -2.8),
    14: (5.5, -4.1),
}
real_edges = [
    (9, 5), (12, 5), (5, 13), (13, 8), (8, 11), (11, 2), (11, 3), (11, 10), (11, 7),
    (2, 1), (3, 1), (7, 1), (10, 1), (1, 6), (6, 4), (4, 14),
]

inner = FancyBboxPatch((3.75, 2.15), 1.3, 1.8, boxstyle="round,pad=0.1,rounding_size=0.18",
                        linewidth=1.3, edgecolor=ACCENT, facecolor=ACCENT_SOFT, zorder=1, alpha=0.55)
ax.add_patch(inner)

for u, v in real_edges:
    if (u, v) == (8, 11):
        color = ACCENT
    elif (u, v) == (11, 10):
        color = INK
    elif u in (1, 6, 4) or v in (6, 4, 14):
        color = DOWNSTREAM_GRAY
    else:
        color = SIBLING_GRAY
    arrow(pos[u], pos[v], color=color)

node(pos[8], "#8", None, ACCENT, ACCENT, "white")
node(pos[11], "#11", None, ACCENT, ACCENT, "white")
node(pos[10], "#10", None, ERROR_RED, ERROR_RED, "white")
ax.text(pos[10][0] + 0.62, pos[10][1], "gold: error", fontsize=6.2, color=ERROR_RED,
        ha="left", va="center")
for sid in [9, 12, 5, 13]:
    node(pos[sid], f"#{sid}", None, DEEP_ANC_FILL, DEEP_ANC_GRAY, MUTED)
for sid in [2, 3, 7]:
    node(pos[sid], f"#{sid}", None, SIBLING_FILL, SIBLING_GRAY, MUTED, linestyle=(0, (3, 2)))
for sid in [1, 6, 4, 14]:
    node(pos[sid], f"#{sid}", None, DOWNSTREAM_FILL, DOWNSTREAM_GRAY, MUTED)

ax.text(6.0, -3.6, "descendants of #10\n(never precede it)", fontsize=6.5, color=MUTED,
        ha="left", va="center", style="italic")

legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ERROR_RED, markeredgecolor=ERROR_RED,
           markersize=7, label="claim #10 (evaluated, gold: error)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT_SOFT, markeredgecolor=ACCENT,
           markersize=7, label=r"SAGER context $\{$#8, #11$\}$"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=DEEP_ANC_FILL, markeredgecolor=DEEP_ANC_GRAY,
           markersize=7, label="deeper true ancestors (beyond depth 2)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SIBLING_FILL, markeredgecolor=SIBLING_GRAY,
           markersize=7, label="true siblings of #10 (order-dependent)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=DOWNSTREAM_FILL, markeredgecolor=DOWNSTREAM_GRAY,
           markersize=7, label="descendants of #10"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1,
          handletextpad=0.5, fontsize=7, frameon=False)

fig.savefig(os.path.join(OUTPUT_DIR, "fig_zoodles_full_graph.png"), dpi=300)
print("Saved fig_zoodles_full_graph.png")
