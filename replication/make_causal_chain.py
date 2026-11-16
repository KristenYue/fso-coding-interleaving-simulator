"""
make_causal_chain.py -- "why interleaving depth is the research question"
Physical causal chain figure for the supervisor meeting / dissertation Ch1.

Outputs: fso_causal_chain.pdf / .png / .svg
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- palette
C_CAUSE = "#E8F1FB"; E_CAUSE = "#2F6FB5"      # physical causes      (blue)
C_CHAN = "#EAF4EC";  E_CHAN = "#3B8C4E"       # channel model        (green)
C_IMP = "#FDF0E3";   E_IMP = "#D07C1E"        # impairment           (amber)
C_FAIL = "#FBE9E9";  E_FAIL = "#C0392B"       # failure branch       (red)
C_OK = "#E8F6EF";    E_OK = "#1E8A63"         # success branch       (teal)
C_COST = "#F3EEF9";  E_COST = "#7A4FA3"       # cost                 (purple)
C_Q = "#FFF8DC";     E_Q = "#B8860B"          # research question    (gold)

fig, ax = plt.subplots(figsize=(13.2, 16.8))
ax.set_xlim(0, 100)
ax.set_ylim(-12.5, 118)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=10.5, weight="normal", ls="solid", lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.55,rounding_size=1.4",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            weight=weight, color="#1a1a1a", zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, color="#444444",
          fs=9.5, style="-|>", lw=1.7, italic=True, connect="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=17,
        linewidth=lw, color=color, zorder=1, connectionstyle=connect,
        shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha="center", va="center", fontsize=fs, color=color,
                style="italic" if italic else "normal",
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec="none", alpha=0.94), zorder=4)


# ------------------------------------------------------- title
ax.text(50, 115.5, "Why interleaver depth is the design variable",
        ha="center", va="center", fontsize=17, weight="bold", color="#111111")
ax.text(50, 112.0,
        "Physical causal chain in an IM/DD OOK free-space optical link",
        ha="center", va="center", fontsize=11.5, color="#555555", style="italic")

# ------------------------------------------------------- row 1: causes
box(25, 105, 40, 7.4,
    "Atmospheric turbulence\n"
    r"random refractive-index fluctuations ($C_n^2$)",
    C_CAUSE, E_CAUSE, fs=10.5, weight="bold")
box(75, 105, 40, 7.4,
    "Platform jitter / misalignment\n"
    "transmitter–receiver boresight error",
    C_CAUSE, E_CAUSE, fs=10.5, weight="bold")

arrow(25, 101.0, 25, 95.5, "scintillation", lx=-11.5)
arrow(75, 101.0, 75, 95.5, "beam wander", lx=11.0)

# ------------------------------------------------------- row 2: channel components
box(25, 91.5, 40, 7.6,
    r"$h_a\ \sim$ Gamma–Gamma" "\n"
    "Al-Habash, Andrews & Phillips (2001)\n"
    r"$\alpha,\beta$ from Rytov variance $\sigma_R^2$",
    C_CHAN, E_CHAN, fs=10)
box(75, 91.5, 40, 7.6,
    r"$h_p\ \sim$ zero-boresight pointing error" "\n"
    "Farid & Hranilovic (2007)\n"
    r"$h_p/A_0 = U^{1/\xi^2}$",
    C_CHAN, E_CHAN, fs=10)

# merge
arrow(25, 87.2, 42, 82.5, connect="arc3,rad=-0.10")
arrow(75, 87.2, 58, 82.5, connect="arc3,rad=0.10")

box(50, 79.0, 44, 7.0,
    "Composite channel gain\n"
    r"$\mathbf{y = h\,x + n}$,   $\mathbf{h = h_a \cdot h_p}$",
    C_CHAN, E_CHAN, fs=12, weight="bold", lw=2.1)

arrow(50, 75.2, 50, 70.0)

# ------------------------------------------------------- deep fade
box(50, 66.5, 52, 7.0,
    "Deep fade\n"
    r"received optical power collapses for $L_c$ consecutive bits",
    C_IMP, E_IMP, fs=11, weight="bold")

arrow(50, 62.7, 50, 57.5, r"channel held constant over $L_c$", lx=17.5)

# ------------------------------------------------------- burst errors
box(50, 54.0, 52, 7.0,
    "Burst errors\n"
    "many consecutive bits / GF(256) symbols corrupted together",
    C_IMP, E_IMP, fs=11, weight="bold")

# ------------------------------------------------------- split
ax.plot([50, 50], [50.2, 47.6], color="#444444", lw=1.7, zorder=1)
ax.plot([24, 76], [47.6, 47.6], color="#444444", lw=1.7, zorder=1)
arrow(24, 47.6, 24, 43.6)
arrow(76, 47.6, 76, 43.6)
ax.text(24, 45.9, r"$D = 1$   (no interleaving)", ha="center", va="center",
        fontsize=10.5, weight="bold", color=E_FAIL,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"), zorder=4)
ax.text(76, 45.9, r"$D$ large   (deep interleaving)", ha="center",
        va="center", fontsize=10.5, weight="bold", color=E_OK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"), zorder=4)

# ---- failure branch
box(24, 39.5, 40, 6.4,
    "Whole burst falls inside\n one codeword", C_FAIL, E_FAIL, fs=10.5)
arrow(24, 36.0, 24, 32.4)
box(24, 28.8, 40, 6.4,
    "Errors exceed correction\n"
    r"capacity ($t = 16$ for RS)", C_FAIL, E_FAIL, fs=10.5)
arrow(24, 25.3, 24, 21.7)
box(24, 18.1, 40, 7.0,
    "Decoding fails\n"
    r"$\mathbf{coding\ gain \approx 0\ dB}$", C_FAIL, E_FAIL,
    fs=11, weight="bold", lw=2.0)

# ---- success branch
box(76, 39.5, 40, 6.4,
    "Burst is spread across\n"
    r"$D$ different codewords", C_OK, E_OK, fs=10.5)
arrow(76, 36.0, 76, 32.4)
box(76, 28.8, 40, 6.4,
    "Each codeword sees only\n a few symbol errors", C_OK, E_OK, fs=10.5)
arrow(76, 25.3, 76, 21.7)
box(76, 18.1, 40, 7.0,
    "Decoding succeeds\n"
    r"$\mathbf{coding\ gain\ realised}$", C_OK, E_OK,
    fs=11, weight="bold", lw=2.0)

# ---- evidence tag under the failure branch
arrow(24, 14.4, 24, 11.0, color=E_FAIL)
box(24, 7.4, 40, 6.4,
    r"Measured at $L_c$ = 256 bits, $D$ = 1:" "\n"
    r"RS $-0.04$ dB,   LDPC $-0.01$ dB" "\n"
    "FEC alone buys nothing",
    "#FFF5F5", E_FAIL, fs=9.4)

# ---- cost of the success branch
arrow(76, 14.4, 76, 11.0, color=E_COST)
box(76, 7.4, 40, 6.4,
    "Cost of depth\n"
    r"buffer $\uparrow$: 255$D$ B (RS) / 64$D$ B (LDPC)" "\n"
    r"interleaving latency $\uparrow$",
    C_COST, E_COST, fs=9.4)

# ---- research question: conclusion of BOTH branches
arrow(24, 4.0, 38, 0.8, color=E_Q, lw=1.9, connect="arc3,rad=0.10")
arrow(76, 4.0, 62, 0.8, color=E_Q, lw=1.9, connect="arc3,rad=-0.10")
box(50, -5.4, 88, 10.4,
    "RESEARCH QUESTION\n"
    r"Where does the benefit saturate — what is $D_{\rm sat}$, and what sets it?" "\n"
    "Does RS or LDPC make better use of the depth we pay for,\n"
    "once reliability, buffer, latency and decoding complexity are all counted?",
    C_Q, E_Q, fs=9.8, weight="bold", lw=2.2, ls="dashed")

fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"fso_causal_chain.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print("saved fso_causal_chain.{pdf,png,svg}")
