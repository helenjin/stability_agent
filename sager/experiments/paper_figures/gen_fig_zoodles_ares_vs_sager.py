#!/usr/bin/env python3
"""Publication figure: ARES vs. SAGER on the zoodles/node-10 example.

Two panels:
  (a) The local dependency structure -- claim #10's two true depth-2
      ancestors (#8, #11) vs. the pool of irrelevant sibling claims that
      ARES's prefix rule can pull in depending on which valid serialization
      is observed.
  (b) The five ARES scores (one per observed serialization rho_0..rho_4)
      against the fixed 0.5 verdict threshold, vs. SAGER's single invariant
      score.

All numbers below are real, pulled from this project's own 15-recipe
ARES-vs-SAGER run (zoodles, node 10) -- see sager/experiments/README.md /
the ares_vs_sager_topodev.py results. Not illustrative placeholders.

Usage: python gen_fig_zoodles_ares_vs_sager.py
Outputs: fig_zoodles_ares_vs_sager.pdf (vector, for LaTeX) and .png (300dpi).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Publication defaults ---------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "lines.linewidth": 1.4,
})

# --- Restrained academic palette -------------------------------------------
ACCENT = "#1B6B93"        # SAGER / true-ancestor context (steel blue)
ACCENT_SOFT = "#DCEAF2"
ERROR_RED = "#B23A48"     # error claim / ARES wrong verdicts
NEUTRAL_DARK = "#3A3A3A"  # ARES correct verdicts (deliberately not green)
SIBLING_GRAY = "#B9B9B4"     # true siblings of #10 (order-dependent)
SIBLING_FILL = "#EDEDEA"
DEEP_ANC_GRAY = "#8A8A84"    # deeper true ancestors, beyond SAGER's depth-2 cutoff
DEEP_ANC_FILL = "#D8D8D3"
INK = "#1A1A1A"
MUTED = "#6B6B66"

FIG = plt.figure(figsize=(7.0, 2.75))
gs = FIG.add_gridspec(1, 2, width_ratios=[0.86, 1.0], wspace=0.30)
axA = FIG.add_subplot(gs[0, 0])
axB = FIG.add_subplot(gs[0, 1])

# =============================================================================
# Panel (a): dependency structure
# =============================================================================
axA.set_xlim(0.4, 8.4)
axA.set_ylim(0, 8.7)
axA.set_aspect("equal", adjustable="box")
axA.set_anchor("C")
axA.axis("off")
axA.set_title("(a) Claim #10's local neighborhood (zoomed in)", loc="left", pad=6)


def node(ax, xy, label, sublabel, facecolor, edgecolor, textcolor, r=0.5, sublabel_color=None,
         linestyle="-"):
    circ = plt.Circle(xy, r, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3, zorder=4,
                       linestyle=linestyle)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=7.6, color=textcolor,
            weight="bold", zorder=5)
    if sublabel:
        ax.text(xy[0], xy[1] - r - 0.22, sublabel, ha="center", va="top", fontsize=6.2,
                color=sublabel_color or INK, zorder=5)


def arrow(ax, p_from, p_to, r_from=0.5, r_to=0.5, color=INK):
    p_from, p_to = np.array(p_from), np.array(p_to)
    d = p_to - p_from
    u = d / np.linalg.norm(d)
    start = p_from + u * r_from
    end = p_to - u * (r_to + 0.05)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                  linewidth=1.0, color=color, zorder=3))


# Real zoodles subgraph, restricted to the local neighborhood of claim #10 (its full
# ancestor closure {12,9,5,13,8,11} and its true siblings {2,3,7}, i.e. other children
# of #11) -- omits only the four nodes strictly downstream of #10 (#1,#6,#4,#14), which
# can never precede it in any valid ordering and so are irrelevant to this example.
# Positions follow the graph's own topological generations (real edges below), not a
# hand-arranged layout.
pos = {
    9: (3.05, 6.85), 12: (5.75, 6.85),
    5: (4.4, 5.55),
    13: (4.4, 4.35),
    8: (4.4, 3.15),
    11: (4.4, 1.95),
    2: (1.4, 0.6), 3: (3.4, 0.6), 10: (5.5, 0.6), 7: (7.5, 0.6),
}
real_edges = [(9, 5), (12, 5), (5, 13), (13, 8), (8, 11), (11, 2), (11, 3), (11, 10), (11, 7)]

# solid box: SAGER's fixed depth-2 context {#8, #11}
inner = FancyBboxPatch((3.75, 1.4), 1.3, 2.25, boxstyle="round,pad=0.1,rounding_size=0.18",
                        linewidth=1.3, edgecolor=ACCENT, facecolor=ACCENT_SOFT, zorder=1, alpha=0.55)
axA.add_patch(inner)

for u, v in real_edges:
    edge_color = ACCENT if (u, v) == (8, 11) else (INK if (u, v) == (11, 10) else SIBLING_GRAY)
    arrow(axA, pos[u], pos[v], color=edge_color)

node(axA, pos[8], "#8", None, ACCENT, ACCENT, "white")
node(axA, pos[11], "#11", None, ACCENT, ACCENT, "white")
node(axA, pos[10], "#10", "gold: error", ERROR_RED, ERROR_RED, "white", sublabel_color=ERROR_RED)
for sid in [9, 12, 5, 13]:
    node(axA, pos[sid], f"#{sid}", None, DEEP_ANC_FILL, DEEP_ANC_GRAY, MUTED)
for sid in [2, 3, 7]:
    node(axA, pos[sid], f"#{sid}", None, SIBLING_FILL, SIBLING_GRAY, MUTED, linestyle=(0, (3, 2)))

# explicit zoom note -- this is an excerpt, not the whole recipe graph
axA.text(0.4, 8.6, "excerpt of the 14-claim zoodles graph\n(4 claims strictly downstream of #10 omitted)",
          fontsize=6.2, color=MUTED, ha="left", va="top", style="italic")

# legend strip below the diagram, mirroring panel (b)'s legend style
legend_a = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT_SOFT, markeredgecolor=ACCENT,
           markersize=7, label=r"SAGER context $\{$#8, #11$\}$"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=DEEP_ANC_FILL, markeredgecolor=DEEP_ANC_GRAY,
           markersize=7, label="deeper true ancestors (beyond depth 2)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SIBLING_FILL, markeredgecolor=SIBLING_GRAY,
           markersize=7, linestyle="None", label="true siblings of #10 (order-dependent)"),
]
axA.legend(handles=legend_a, loc="upper center", bbox_to_anchor=(0.47, -0.04), ncol=1,
           handletextpad=0.5, fontsize=6.6, frameon=False)

# =============================================================================
# Panel (b): score variation across serializations
# =============================================================================
rho_labels = [r"$\rho_0$", r"$\rho_1$", r"$\rho_2$", r"$\rho_3$", r"$\rho_4$"]
ares_scores = [0.4028, 0.7702, 0.7702, 0.3128, 0.4085]
ares_correct = [True, False, False, True, True]  # vs. gold = error
sager_score = 0.468
threshold = 0.5

x = list(range(5))
axB.set_title("(b) ARES score vs. serialization; SAGER is invariant", loc="left", pad=6)

axB.axhline(threshold, color=MUTED, linestyle=(0, (5, 3)), linewidth=1.1, zorder=1)
axB.text(4.62, threshold + 0.045, "threshold", fontsize=6.6, color=MUTED, ha="right", va="bottom")

# SAGER: single invariant reference line (no repeated markers -- one clean label instead).
# Bounded to the actual data span (not axB.axhline, which would span the full axes
# width including the left margin where the "SAGER" label lives, running the line
# directly behind/through the label text).
axB.plot([0, 4], [sager_score, sager_score], color=ACCENT, linestyle="-", linewidth=1.3,
          alpha=0.85, zorder=1, solid_capstyle="butt")
axB.text(-0.32, sager_score, "SAGER", fontsize=6.8, color=ACCENT, ha="right", va="center", weight="bold")

for xi, (s, ok) in enumerate(zip(ares_scores, ares_correct)):
    color = NEUTRAL_DARK if ok else ERROR_RED
    axB.scatter(xi, s, marker="o", s=46, facecolor=color, edgecolor="white", linewidth=0.7, zorder=4)
    tag = "correct" if ok else "wrong"
    # keep labels clear of the crowded 0.45-0.52 threshold/SAGER band
    if s > 0.55:
        dy, va = 0.075, "bottom"
    else:
        dy, va = -0.075, "top"
    axB.text(xi, s + dy, tag, ha="center", va=va, fontsize=6.6, color=color, style="italic")

# swing annotation -- text only, placed in genuinely empty plot area (no overlap risk)
axB.text(0.98, 0.965, r"ARES range: $\Delta$=0.457", transform=axB.transAxes, fontsize=7,
          color=INK, ha="right", va="top",
          bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=SIBLING_GRAY, linewidth=0.8))

axB.set_xticks(x)
axB.set_xticklabels(rho_labels)
axB.set_ylabel("score")
axB.set_ylim(0.15, 0.95)
axB.set_xlim(-0.55, 4.5)

legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL_DARK, markeredgecolor="white",
           markersize=6, label="ARES, correct verdict"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ERROR_RED, markeredgecolor="white",
           markersize=6, label="ARES, wrong verdict"),
    Line2D([0], [0], color=ACCENT, linewidth=1.6, label="SAGER (invariant, correct)"),
]
axB.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
           columnspacing=1.1, handletextpad=0.5)

FIG.savefig(os.path.join(OUTPUT_DIR, "fig_zoodles_ares_vs_sager.pdf"))
FIG.savefig(os.path.join(OUTPUT_DIR, "fig_zoodles_ares_vs_sager.png"), dpi=300)
print("Saved fig_zoodles_ares_vs_sager.pdf and .png to", OUTPUT_DIR)
