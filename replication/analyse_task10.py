"""
analyse_task10.py -- threshold extraction, coding gain and figures for Task 10.

Reads 10_turbulence_sensitivity.csv (produced by run_task10.py) and produces:
  10_turbulence_sensitivity_summary.csv
  10_turbulence_sensitivity.png / .pdf
  10_findings.txt
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fso_lib as F
from qfunc import Q

TARGET = 1e-3
DEPTHS = [1, 8, 32, 128]
TURBS = ["Weak", "Moderate", "Strong"]
CODECS = ["RS", "LDPC"]


def load():
    rows = []
    with open("10_turbulence_sensitivity.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["D"] = int(r["D"])
            r["ebn0_db"] = float(r["ebn0_db"])
            r["ber"] = float(r["ber"])
            r["ber_ci95"] = float(r["ber_ci95"])
            r["fer"] = float(r["fer"])
            rows.append(r)
    return rows


def threshold(snr, ber, target=TARGET):
    """Log-domain linear interpolation for the first crossing of `target`."""
    snr = np.asarray(snr, float)
    ber = np.asarray(ber, float)
    o = np.argsort(snr)
    snr, ber = snr[o], ber[o]
    for i in range(1, len(ber)):
        if ber[i] < target <= ber[i - 1]:
            y0, y1 = np.log10(max(ber[i - 1], 1e-12)), np.log10(max(ber[i], 1e-12))
            return float(snr[i - 1] + (np.log10(target) - y0) *
                         (snr[i] - snr[i - 1]) / (y1 - y0))
    return float("nan")


def uncoded_threshold(sR2, n=800_000, seed=20260727):
    """Exact marginal uncoded threshold: BER = E[Q(h sqrt(Eb/N0))], R = 1."""
    rng = np.random.default_rng(seed)
    a, b = F.gg_params(sR2)
    h = F.draw_h(n, a, b, F.POINTING["Severe"], rng)
    grid = np.arange(15.0, 55.0, 0.5)
    ber = np.array([Q(h * np.sqrt(10 ** (g / 10))).mean() for g in grid])
    return threshold(grid, ber)


def main():
    rows = load()
    unc = {t: uncoded_threshold(F.TURBULENCE[t]) for t in TURBS}

    thr = {}
    curves = {}
    for t in TURBS:
        for c in CODECS:
            for d in DEPTHS:
                sel = [r for r in rows
                       if r["turbulence"] == t and r["codec"] == c and r["D"] == d]
                sel.sort(key=lambda r: r["ebn0_db"])
                curves[(t, c, d)] = sel
                thr[(t, c, d)] = threshold([r["ebn0_db"] for r in sel],
                                           [r["ber"] for r in sel])

    # ---------------- summary CSV ----------------
    out = []
    for t in TURBS:
        a, b = F.gg_params(F.TURBULENCE[t])
        for c in CODECS:
            base = thr[(t, c, 1)]
            for d in DEPTHS:
                x = thr[(t, c, d)]
                out.append(dict(
                    turbulence=t, sigma_R2=F.TURBULENCE[t],
                    alpha=round(a, 4), beta=round(b, 4),
                    SI=round(F.scintillation_index(a, b), 4),
                    codec=c, D=d,
                    req_ebn0_db=("" if np.isnan(x) else round(x, 3)),
                    interleaving_gain_db=("" if np.isnan(x) or np.isnan(base)
                                          else round(base - x, 3)),
                    uncoded_ebn0_db=round(unc[t], 3),
                    true_coding_gain_db=("" if np.isnan(x)
                                         else round(unc[t] - x, 3)),
                    buffer_KiB=round((255 * d if c == "RS" else 64 * d) / 1024, 3),
                ))
    with open("10_turbulence_sensitivity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    colors = {1: "#1f77b4", 8: "#2ca02c", 32: "#ff7f0e", 128: "#d62728"}
    for i, t in enumerate(TURBS):
        # (a) RS BER curves, (b) LDPC BER curves
        for j, c in enumerate(CODECS):
            ax = axes[i, j]
            for d in DEPTHS:
                s = curves[(t, c, d)]
                if not s:
                    continue
                x = [r["ebn0_db"] for r in s]
                y = [max(r["ber"], 1e-7) for r in s]
                e = [r["ber_ci95"] for r in s]
                ax.semilogy(x, y, "o-", ms=3.5, lw=1.3, color=colors[d],
                            label=f"D = {d}")
                ax.fill_between(x, np.maximum(np.array(y) - np.array(e), 1e-8),
                                np.array(y) + np.array(e),
                                color=colors[d], alpha=0.18, lw=0)
            ax.axhline(TARGET, color="k", ls="--", lw=0.9)
            ax.axvline(unc[t], color="grey", ls=":", lw=1.2)
            ax.set_title(f"{t} turbulence ($\\sigma_R^2$={F.TURBULENCE[t]}) — {c}",
                         fontsize=10)
            ax.set_xlabel("Information-bit $E_b/N_0$ (dB)")
            ax.set_ylabel("Decoded information BER")
            ax.set_ylim(1e-6, 1)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

        # (c) required Eb/N0 and coding gain vs D
        ax = axes[i, 2]
        for c, mk, col in (("RS", "o--", "#1f77b4"), ("LDPC", "s-", "#d62728")):
            g = [unc[t] - thr[(t, c, d)] for d in DEPTHS]
            ax.plot(range(len(DEPTHS)), g, mk, color=col, label=c, ms=6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels([str(d) for d in DEPTHS])
        ax.set_xlabel("Interleaver depth $D$")
        ax.set_ylabel("True coding gain (dB)")
        ax.set_title(f"{t}: coding gain vs depth\n(uncoded ref "
                     f"{unc[t]:.2f} dB)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Task 10 — Turbulence sensitivity of the recommended interleaver "
                 "depth ($L_c$ = 64 bits, Severe pointing, $\\xi$ = 1.52)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("10_turbulence_sensitivity.png", dpi=160)
    fig.savefig("10_turbulence_sensitivity.pdf")

    # ---------------- findings ----------------
    lines = []
    A = lines.append
    A("TASK 10 - TURBULENCE SENSITIVITY OF THE RECOMMENDED INTERLEAVER DEPTH")
    A("=" * 78)
    A(f"Fixed: Lc = 64 transmitted bits, Severe pointing (xi = 1.52), "
      f"target BER = {TARGET:.0e}")
    A(f"RS(255,223) R = {F.RS_RATE:.6f}   LDPC(512,449) R = {449/512:.6f}")
    A("")
    A(f"{'Turb':9s}{'sR2':>6s}{'alpha':>8s}{'beta':>8s}{'SI':>8s}"
      f"{'uncoded thr':>14s}")
    for t in TURBS:
        a, b = F.gg_params(F.TURBULENCE[t])
        A(f"{t:9s}{F.TURBULENCE[t]:>6.2f}{a:>8.3f}{b:>8.3f}"
          f"{F.scintillation_index(a,b):>8.3f}{unc[t]:>12.2f} dB")
    A("")
    A("Required information-bit Eb/N0 at BER = 1e-3 (dB)")
    A(f"{'Turb':9s}{'codec':6s}" + "".join(f"{'D='+str(d):>10s}" for d in DEPTHS)
      + f"{'  D_sat':>9s}")
    for t in TURBS:
        for c in CODECS:
            vals = [thr[(t, c, d)] for d in DEPTHS]
            # saturation depth: smallest D within 0.5 dB of the best
            best = np.nanmin(vals)
            dsat = next((d for d, v in zip(DEPTHS, vals)
                         if not np.isnan(v) and v - best <= 0.5), None)
            A(f"{t:9s}{c:6s}" + "".join(
                f"{('nan' if np.isnan(v) else f'{v:.2f}'):>10s}" for v in vals)
              + f"{str(dsat):>9s}")
    A("")
    A("True coding gain vs the matched uncoded FSO baseline (dB)")
    A(f"{'Turb':9s}{'codec':6s}" + "".join(f"{'D='+str(d):>10s}" for d in DEPTHS))
    for t in TURBS:
        for c in CODECS:
            A(f"{t:9s}{c:6s}" + "".join(
                f"{(unc[t]-thr[(t,c,d)]):>10.2f}" for d in DEPTHS))
    A("")
    A("Observations")
    A("-" * 78)
    for t in TURBS:
        rs = [thr[(t, "RS", d)] for d in DEPTHS]
        ld = [thr[(t, "LDPC", d)] for d in DEPTHS]
        rs_best, ld_best = np.nanmin(rs), np.nanmin(ld)
        rs_sat = next(d for d, v in zip(DEPTHS, rs) if v - rs_best <= 0.5)
        ld_sat = next(d for d, v in zip(DEPTHS, ld) if v - ld_best <= 0.5)
        A(f"{t}: RS saturates at D = {rs_sat} ({rs_best:.2f} dB); "
          f"LDPC best at D = {ld_sat} ({ld_best:.2f} dB); "
          f"LDPC advantage at D = 128 = {rs[-1]-ld[-1]:+.2f} dB")
    with open("10_findings.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
