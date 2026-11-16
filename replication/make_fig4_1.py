"""
make_fig4_1.py -- Figure 4-1: simulation task dependency pipeline.

Methodology-chapter version: shows WHAT each task does and WHAT it depends on.
Deliberately contains no numerical results and no forward chapter references,
so that Chapter 4 does not pre-empt Chapters 6 and 7.

Portrait aspect, sized for a single A4 page at ~16 cm text width.
Outputs: fig4_1_task_pipeline.pdf / .png / .svg
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

C = {
    "val":    ("#E8F1FB", "#2F6FB5"),
    "base":   ("#EAF4EC", "#3B8C4E"),
    "main":   ("#FDF0E3", "#D07C1E"),
    "judge":  ("#F3EEF9", "#7A4FA3"),
    "robust": ("#FFF8DC", "#B8860B"),
    "dead":   ("#F4F4F4", "#9A9A9A"),
}

fig, ax = plt.subplots(figsize=(9.2, 15.1))
ax.set_xlim(0, 120)
ax.set_ylim(0, 197)
ax.axis("off")


def band(y0, y1, label, colour):
    ax.add_patch(Rectangle((2, y0), 116, y1 - y0, facecolor=colour,
                           edgecolor="none", alpha=0.34, zorder=0))
    ax.text(6, y1 - 3.6, label, ha="left", va="center",
            fontsize=10.5, weight="bold", color="#3A3A3A", zorder=1)


def card(x, y, w, h, title, code, what, kind, fs=8.2):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.45,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=1.5,
                                zorder=3))
    ax.text(x, y + h / 2 - 3.2, title, ha="center", va="center",
            fontsize=fs + 1.5, weight="bold", color=ec, zorder=4)
    ax.text(x, y + h / 2 - 6.4, code, ha="center", va="center",
            fontsize=fs - 1.6, color="#5A5A5A", family="monospace", zorder=4)
    ax.text(x, y - h / 2 + 5.0, what, ha="center", va="center",
            fontsize=fs, color="#1A1A1A", zorder=4, linespacing=1.45)


def arr(x1, y1, x2, y2, color="#555555", lw=1.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


# ------------------------------------------------------------------ bands
band(166, 194, "Component validation", C["val"][0])
band(138, 165.5, "Uncoded reference", C["base"][0])
band(72, 137, "Coded experiments   ($L_c$ fixed, $D$ swept)", C["main"][0])
band(42.5, 69, "Cost accounting", C["judge"][0])
band(12.5, 39, "Robustness check", C["robust"][0])

# ------------------------------------------------------------------ layer 1
card(22, 178, 36, 18, "Task 01", "01_channel_model_validation.py",
     "Gamma-Gamma, pointing and\nAWGN components checked\nagainst closed-form theory",
     "val")
card(60, 178, 36, 18, "Task 03", "03_rs_awgn_validation.py",
     "RS encoder / decoder\nself-test and AWGN\nreference curve", "val")
card(98, 178, 36, 18, "Task 06", "06_ldpc_awgn_validation.py",
     "Rate-matched LDPC(512,449)\nconstructed, verified and\nstored for reuse", "val")

# ------------------------------------------------------------------ layer 2
card(32, 150, 50, 17, "Task 02", "02_uncoded_fso_ber.py",
     "Uncoded FSO BER across\nturbulence and pointing severity", "base")
card(88, 150, 50, 17, "Task 09", "09_uncoded_fixed_coherence_coding_gain.py",
     "Uncoded baseline under fixed $L_c$;\npooled reference threshold\n"
     "for the coding-gain definition", "base")

# ------------------------------------------------------------------ layer 3
card(32, 122, 50, 17, "Task 04C", "04c_rs_fso_fixed_coherence_interleaving.py",
     "RS,  $L_c$ = 64 bits\n$D \\in \\{1,\\ 8,\\ 32,\\ 128\\}$", "main")
card(32, 102, 50, 17, "Task 04D", "04d_rs_fso_fixed_coherence32_interleaving.py",
     "RS,  $L_c$ = 256 bits\n$D \\in \\{1,\\ 8,\\ 32,\\ 128\\}$", "main")
card(32, 83, 50, 15, "Task 05", "05_rs_coherence_depth_analysis.py",
     "Threshold extraction and\ninterleaving gain (§4.6.1)", "main")
card(88, 122, 50, 17, "Task 07B", "07b_ldpc_fso_group_aware_refinement.py",
     "LDPC, both $L_c$, same $D$ sweep;\n$G \\geq 40$ independent groups (§4.5)",
     "main")
card(88, 100, 50, 15, "Task 07", "07_ldpc_fso_fixed_coherence_interleaving.py",
     "Frame-count stopping rule;\nsuperseded — see §4.5", "dead")

# ------------------------------------------------------------------ layer 4
card(60, 54, 96, 16, "Task 08", "08_rs_ldpc_final_comparison.py",
     "Joint accounting of required $E_b/N_0$, interleaver buffer,\n"
     "latency and decoding complexity on a common basis (§4.6)",
     "judge", fs=8.6)

# ------------------------------------------------------------------ layer 5
card(60, 24, 96, 16, "Task 10",
     "run_task10.py  ·  fso_lib.py  ·  analyse_task10.py",
     "Coded sweep repeated at $\\sigma_R^2 \\in \\{0.20,\\ 0.75,\\ 2.00\\}$, "
     "with $L_c$ and\npointing severity held fixed (§4.1.2)",
     "robust", fs=8.6)

# ------------------------------------------------------------------ arrows
arr(22, 169, 30, 158.7, rad=-0.08)                 # 01 -> 02
arr(60, 169, 78, 158.7, rad=-0.10)                 # 03 -> 09
arr(98, 169, 96, 158.7, rad=0.06)                  # 06 -> 09
arr(28, 141.3, 30, 130.7)                          # 02 -> 04C
arr(45, 143, 74, 130.7, rad=-0.14)                 # 02 -> 07B
arr(32, 113.3, 32, 110.7)                          # 04C -> 04D
arr(88, 107.7, 88, 113.3, color="#9A9A9A",
    ls="--", lw=1.2)                               # 07 superseded by 07B
ax.text(92, 110.5, "superseded", ha="left", va="center", fontsize=7.4,
        style="italic", color="#8A8A8A", zorder=4)

arr(23, 93.3, 25, 90.7, ls="--")                   # 04C -> 05 (via 04D column)
arr(41, 93.3, 39, 90.7, ls="--")                   # 04D -> 05
arr(32, 75.3, 44, 62.3, rad=0.12, ls="--")         # 05 -> 08
arr(64, 113.5, 68, 62.3, rad=0.26, ls="--")        # 07B -> 08 (left of 07)
arr(111, 141.3, 92, 62.3, rad=-0.13, ls="--")      # 09 -> 08
arr(60, 45.8, 60, 32.2, lw=2.0, color="#B8860B")   # 08 -> 10

# ------------------------------------------------------------------ legend
ax.text(60, 5.5,
        "Solid arrow: the receiving task re-simulates using the validated model.        "
        "Dashed arrow: the receiving task re-reads a frozen CSV output.\n"
        "Held fixed throughout: IM/DD OOK, perfect CSI, Severe pointing error "
        "($\\xi$ = 1.52), target BER = $10^{-3}$.  Turbulence is fixed at "
        "$\\sigma_R^2$ = 0.75 except in Task 10.",
        ha="center", va="center", fontsize=7.8, color="#4A4A4A",
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#CCCCCC", lw=0.9))

fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"fig4_1_task_pipeline.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("saved fig4_1_task_pipeline.{pdf,png,svg}")
