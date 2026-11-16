"""
make_task_map.py -- one-page research architecture map:
Task -> code -> chapter -> conclusion.

Outputs: fso_task_map.pdf / .png / .svg
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
    "dead":   ("#F2F2F2", "#9A9A9A"),
}

fig, ax = plt.subplots(figsize=(17.6, 15.2))
ax.set_xlim(0, 178)
ax.set_ylim(0, 152)
ax.axis("off")

LEFT = 34.0          # cards must not start left of this


def band(y0, y1, label, sub, colour):
    ax.add_patch(Rectangle((2, y0), 174, y1 - y0, facecolor=colour,
                           edgecolor="none", alpha=0.32, zorder=0))
    ax.text(5, (y0 + y1) / 2 + 2.2, label, ha="left", va="center",
            fontsize=12.5, weight="bold", color="#333333", zorder=1)
    ax.text(5, (y0 + y1) / 2 - 2.6, sub, ha="left", va="center",
            fontsize=8.6, color="#707070", style="italic", zorder=1,
            linespacing=1.4)


def card(x, y, w, h, task, code, chap, concl, kind, fs=8.6):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.1",
                                facecolor=fc, edgecolor=ec, linewidth=1.7,
                                zorder=3))
    ax.text(x, y + h / 2 - 2.8, task, ha="center", va="center",
            fontsize=fs + 1.9, weight="bold", color=ec, zorder=4)
    ax.text(x, y + h / 2 - 6.3, code, ha="center", va="center",
            fontsize=fs - 1.3, color="#555555", family="monospace", zorder=4)
    ax.text(x, y - h / 2 + 6.6, concl, ha="center", va="center",
            fontsize=fs, color="#1a1a1a", zorder=4, linespacing=1.4)
    ax.text(x, y - h / 2 + 2.1, chap, ha="center", va="center",
            fontsize=fs - 0.2, weight="bold", color=ec, zorder=4)


def arr(x1, y1, x2, y2, color="#555555", lw=1.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=lw, color=color,
                                 linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


ax.text(89, 148.5, "FSO dissertation — research architecture",
        ha="center", va="center", fontsize=18.5, weight="bold")
ax.text(89, 144.4,
        "Task  →  code  →  chapter  →  conclusion          "
        "solid arrow = simulation dependency,   dashed arrow = re-reads a frozen CSV",
        ha="center", va="center", fontsize=10.4, color="#666666", style="italic")

# ---------------------------------------------------------------- bands
band(124, 141, "LAYER 1   Validation",
     "prove the simulator is correct\nbefore trusting any curve", C["val"][0])
band(100, 121, "LAYER 2   Reference",
     "establish what happens with no coding —\nthe yardstick for coding gain",
     C["base"][0])
band(51, 97, "LAYER 3   Main experiments",
     "fix the coherence length $L_c$,\nsweep the depth $D$, measure both codecs",
     C["main"][0])
band(28, 48, "LAYER 4   Engineering judgement",
     "reliability vs buffer\nvs latency vs complexity", C["judge"][0])
band(4, 25, "LAYER 5   Robustness",
     "does the recommendation survive\na change of turbulence?", C["robust"][0])

# ---------------------------------------------------------------- layer 1
card(62, 132.5, 42, 16, "Task 01   Channel validation",
     "01_channel_model_validation.py", "→ Ch 5.1–5.3",
     "AWGN BER, Gamma-Gamma PDF and\npointing CDF all match closed-form theory",
     "val")
card(110, 132.5, 42, 16, "Task 03   RS on AWGN",
     "03_rs_awgn_validation.py", "→ Ch 6.1",
     "RS corrects ≤ 16 symbol errors;\nsharp waterfall between 8 and 9 dB", "val")
card(156, 132.5, 38, 16, "Task 06   LDPC on AWGN",
     "06_ldpc_awgn_validation.py", "→ Ch 6.2",
     "Rate-matched LDPC(512,449)\nR = 0.8770 ≈ R$_{RS}$ = 0.8745", "val")

# ---------------------------------------------------------------- layer 2
card(70, 110.5, 44, 17, "Task 02   Uncoded FSO",
     "02_uncoded_fso_ber.py", "→ Ch 5.4–5.5",
     "Both turbulence and pointing error\ndegrade BER substantially;\n"
     "FSO curves fall far slower than AWGN", "base")
card(136, 110.5, 48, 17, "Task 09   Common uncoded baseline",
     "09_uncoded_fixed_coherence_coding_gain.py", "→ Ch 5.6 / 6.6",
     "Pooled threshold 33.172 dB\n(95% CI 32.204–34.306)\n"
     "→ true coding gain becomes definable", "base")

# ---------------------------------------------------------------- layer 3
card(60, 86, 42, 17, "Task 04C   RS,  $L_c$ = 64 bits",
     "04c_rs_fso_fixed_coherence_interleaving.py", "→ Ch 6.3",
     "D = 1 → 27.19 dB,   D = 8 → 24.05 dB\ngain saturates at D = 8", "main")
card(60, 62, 42, 17, "Task 04D   RS,  $L_c$ = 256 bits",
     "04d_rs_fso_fixed_coherence32_interleaving.py", "→ Ch 6.4",
     "D = 1 → 33.21 dB ≈ uncoded!\nneeds D ≈ 32 to recover the gain", "main")
card(105, 74, 40, 18, "Task 05   RS depth analysis",
     "05_rs_coherence_depth_analysis.py", "→ Ch 6.4",
     "Re-reads 04C / 04D CSV and\nlog-interpolates at BER = 10$^{-3}$.\n"
     "Effective depth grows with $L_c$", "main")
card(150, 86, 42, 17, "Task 07B   LDPC + interleaving",
     "07b_ldpc_fso_group_aware_refinement.py", "→ Ch 6.5",
     "G ≥ 40 independent interleaver groups\n"
     "D = 128 → 21.62 dB;  loses to RS at D = 8", "main")
card(150, 62, 42, 15, "Task 07   development only",
     "07_ldpc_fso_fixed_coherence_interleaving.py", "not a final result",
     "Only 16 independent groups at D = 128\n→ statistically invalid", "dead")

# ---------------------------------------------------------------- layer 4
card(105, 38, 74, 16, "Task 08   Fair RS / LDPC comparison",
     "08_rs_ldpc_final_comparison.py", "→ Ch 7.2–7.6",
     "Pareto front:  RS D = 8 (1.99 KiB, ≈1.7 ms per 1000 info bits)\n"
     "vs  LDPC D = 128 (lowest threshold 21.62 dB, 8 KiB, ≈59× slower)\n"
     "→ no single winner; design recommendations R1–R4", "judge")

# ---------------------------------------------------------------- layer 5
card(105, 14.5, 96, 17,
     "Task 10   Turbulence sensitivity      (newly completed)",
     "run_task10.py  +  fso_lib.py  +  analyse_task10.py", "→ Ch 7.8",
     "RS saturates at D = 8 under Weak / Moderate / Strong turbulence "
     "(spread ≤ 0.23 dB)\n"
     "→ $D_{sat}$ is set by the coherence length, NOT by turbulence strength\n"
     "→ a single-point observation becomes a transferable design rule",
     "robust", fs=9.4)

# ---------------------------------------------------------------- arrows
arr(62, 124.5, 68, 119.2, rad=-0.06)             # 01 -> 02
arr(110, 124.5, 128, 119.2, rad=-0.10)           # 03 -> 09
arr(156, 124.5, 144, 119.2, rad=0.10)            # 06 -> 09
arr(66, 102, 60, 94.7, rad=0.06)                 # 02 -> 04C
arr(80, 104, 136, 94.7, rad=-0.16)               # 02 -> 07B
arr(60, 77.3, 60, 70.7)                          # 04C -> 04D
arr(150, 69.6, 150, 77.3, color="#9A9A9A",
    ls="--", lw=1.4)                             # 07 superseded by 07B
ax.text(154.5, 73.4, "superseded", ha="left", va="center", fontsize=8,
        style="italic", color="#8A8A8A", zorder=4)

arr(81.5, 84, 85, 78, rad=-0.12, ls="--")        # 04C -> 05
arr(81.5, 64, 85, 70, rad=0.12, ls="--")         # 04D -> 05
arr(105, 65, 105, 46.2, ls="--")                 # 05 -> 08
arr(129, 82.5, 140, 46.2, rad=0.30, ls="--")     # 07B -> 08 (routed clear of 07)
arr(136, 102, 122, 46.2, rad=-0.12, ls="--")     # 09 -> 08
arr(105, 30, 105, 23.2, lw=2.4, color="#B8860B")  # 08 -> 10

fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"fso_task_map.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("saved fso_task_map.{pdf,png,svg}")
