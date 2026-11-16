"""
Task 05: Analyse RS interleaving depth versus channel coherence length.

Inputs:
- results/04c_rs_fso_fixed_coherence_interleaving.csv
  Coherence length Lc = 8 RS symbols

- results/04d_rs_fso_fixed_coherence32_interleaving.csv
  Coherence length Lc = 32 RS symbols

Outputs:
- Combined BER comparison
- Required Eb/N0 at target BER = 1e-3
- Interleaving gain
- Buffering/latency proxy
- Summary CSV

This program does not run a new Monte Carlo simulation.
It analyses the existing raw data.
"""

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Settings
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_FILES = {
    8: (
        RESULTS_DIR
        / "04c_rs_fso_fixed_coherence_interleaving.csv"
    ),
    32: (
        RESULTS_DIR
        / "04d_rs_fso_fixed_coherence32_interleaving.csv"
    ),
}

INTERLEAVER_DEPTHS = [1, 8, 32, 128]

TARGET_BER = 1e-3

RS_N = 255
RS_K = 223

OUTPUT_CSV = (
    RESULTS_DIR
    / "05_rs_coherence_depth_summary.csv"
)

OUTPUT_PNG = (
    RESULTS_DIR
    / "05_rs_coherence_depth_analysis.png"
)

OUTPUT_PDF = (
    RESULTS_DIR
    / "05_rs_coherence_depth_analysis.pdf"
)


# ============================================================
# 2. Data loading
# ============================================================

def parse_optional_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return float(value)


def load_results(path, coherence_symbols):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for raw in reader:
            measured_ber = parse_optional_float(
                raw.get("BER")
            )

            ber_upper = parse_optional_float(
                raw.get("BER_upper")
            )

            measured_fer = parse_optional_float(
                raw.get("FER")
            )

            fer_upper = parse_optional_float(
                raw.get("FER_upper")
            )

            if measured_ber is None:
                displayed_ber = ber_upper
                ber_is_upper_bound = True
            else:
                displayed_ber = measured_ber
                ber_is_upper_bound = False

            if measured_fer is None:
                displayed_fer = fer_upper
                fer_is_upper_bound = True
            else:
                displayed_fer = measured_fer
                fer_is_upper_bound = False

            rows.append({
                "coherence_symbols": coherence_symbols,
                "D": int(raw["D"]),
                "ebn0_db": float(raw["EbN0_dB"]),
                "ber": displayed_ber,
                "ber_is_upper_bound": (
                    ber_is_upper_bound
                ),
                "fer": displayed_fer,
                "fer_is_upper_bound": (
                    fer_is_upper_bound
                ),
                "frames": int(raw["frames"]),
                "failures": int(raw["failures"]),
                "miscorrections": int(
                    raw["miscorr"]
                ),
                "runtime_s": float(raw["time_s"]),
                "decoded_bit_errors": int(
                    raw["decoded_bit_errors"]
                ),
                "frame_errors": int(
                    raw["frame_errors"]
                ),
            })

    return rows


# ============================================================
# 3. Curve extraction and checking
# ============================================================

def get_curve(
    rows,
    coherence_symbols,
    depth,
):
    curve = [
        row
        for row in rows
        if (
            row["coherence_symbols"]
            == coherence_symbols
            and row["D"] == depth
        )
    ]

    curve.sort(
        key=lambda row: row["ebn0_db"]
    )

    if len(curve) == 0:
        raise ValueError(
            f"No data for Lc={coherence_symbols}, "
            f"D={depth}."
        )

    return curve


def check_monotonicity(curve):
    """
    Return the number of statistically observed upward BER steps.

    Small Monte Carlo fluctuations are reported rather than hidden.
    """
    violations = 0

    for previous, current in zip(
        curve[:-1],
        curve[1:],
    ):
        if current["ber"] > previous["ber"]:
            violations += 1

    return violations


# ============================================================
# 4. Log-domain threshold interpolation
# ============================================================

def required_ebn0_at_target(
    curve,
    target_ber,
):
    """
    Estimate required Eb/N0 using linear interpolation
    in log10(BER).

    Returns:
        required_ebn0,
        uses_upper_bound
    """
    for first, second in zip(
        curve[:-1],
        curve[1:],
    ):
        y1 = first["ber"]
        y2 = second["ber"]

        if y1 is None or y2 is None:
            continue

        if y1 >= target_ber >= y2:
            x1 = first["ebn0_db"]
            x2 = second["ebn0_db"]

            log_y1 = math.log10(y1)
            log_y2 = math.log10(y2)
            log_target = math.log10(
                target_ber
            )

            if log_y1 == log_y2:
                required = x2
            else:
                fraction = (
                    (log_target - log_y1)
                    / (log_y2 - log_y1)
                )

                required = (
                    x1
                    + fraction * (x2 - x1)
                )

            censored = (
                first["ber_is_upper_bound"]
                or second["ber_is_upper_bound"]
            )

            return required, censored

    return None, False


# ============================================================
# 5. Plotting helpers
# ============================================================

def draw_ber_panel(
    axis,
    all_rows,
    coherence_symbols,
):
    colours = {
        1: "black",
        8: "royalblue",
        32: "forestgreen",
        128: "firebrick",
    }

    markers = {
        1: "o",
        8: "s",
        32: "^",
        128: "D",
    }

    for depth in INTERLEAVER_DEPTHS:
        curve = get_curve(
            all_rows,
            coherence_symbols,
            depth,
        )

        x = np.array([
            row["ebn0_db"]
            for row in curve
        ])

        y = np.array([
            row["ber"]
            for row in curve
        ])

        upper_mask = np.array([
            row["ber_is_upper_bound"]
            for row in curve
        ])

        axis.semilogy(
            x,
            y,
            color=colours[depth],
            marker=markers[depth],
            linewidth=1.6,
            markersize=4,
            label=f"D={depth}",
        )

        if np.any(upper_mask):
            axis.scatter(
                x[upper_mask],
                y[upper_mask],
                marker="v",
                facecolors="none",
                edgecolors=colours[depth],
                s=50,
                zorder=5,
            )

    axis.axhline(
        TARGET_BER,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Target BER={TARGET_BER:.0e}",
    )

    axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    axis.set_ylabel(
        "Decoded information BER"
    )

    axis.set_title(
        f"Coherence length Lc={coherence_symbols} "
        "RS symbols"
    )

    axis.set_ylim(1e-8, 1)
    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(fontsize=7)


# ============================================================
# 6. Main analysis
# ============================================================

def main():
    print("=" * 76)
    print(
        "TASK 05: RS COHERENCE LENGTH "
        "AND INTERLEAVER DEPTH ANALYSIS"
    )
    print("=" * 76)

    all_rows = []

    for coherence_symbols, path in (
        INPUT_FILES.items()
    ):
        loaded = load_results(
            path,
            coherence_symbols,
        )

        all_rows.extend(loaded)

        print(
            f"Loaded {len(loaded)} rows "
            f"for Lc={coherence_symbols} "
            f"from {path.name}"
        )

    summary = []

    for coherence_symbols in sorted(
        INPUT_FILES
    ):
        baseline_curve = get_curve(
            all_rows,
            coherence_symbols,
            depth=1,
        )

        baseline_required, baseline_censored = (
            required_ebn0_at_target(
                baseline_curve,
                TARGET_BER,
            )
        )

        print(
            "\n"
            + "-" * 76
        )

        print(
            f"Coherence length "
            f"Lc={coherence_symbols}"
        )

        print(
            "-" * 76
        )

        for depth in INTERLEAVER_DEPTHS:
            curve = get_curve(
                all_rows,
                coherence_symbols,
                depth,
            )

            required, censored = (
                required_ebn0_at_target(
                    curve,
                    TARGET_BER,
                )
            )

            monotonicity_violations = (
                check_monotonicity(curve)
            )

            if (
                required is not None
                and baseline_required is not None
            ):
                coding_gain = (
                    baseline_required - required
                )
            else:
                coding_gain = None

            # Interleaver storage/latency proxies
            buffered_rs_symbols = (
                depth * RS_N
            )

            buffered_bytes = (
                buffered_rs_symbols
            )

            buffered_bits = (
                buffered_bytes * 8
            )

            latency_codewords = depth

            total_runtime = sum(
                row["runtime_s"]
                for row in curve
            )

            total_frames = sum(
                row["frames"]
                for row in curve
            )

            runtime_per_frame_ms = (
                1000.0
                * total_runtime
                / total_frames
            )

            summary_row = {
                "coherence_symbols": (
                    coherence_symbols
                ),
                "interleaver_depth": depth,
                "target_ber": TARGET_BER,
                "required_ebn0_db": required,
                "required_ebn0_uses_upper_bound": (
                    censored
                ),
                "baseline_required_ebn0_db": (
                    baseline_required
                ),
                "baseline_uses_upper_bound": (
                    baseline_censored
                ),
                "interleaving_gain_db": (
                    coding_gain
                ),
                "buffered_rs_symbols": (
                    buffered_rs_symbols
                ),
                "buffered_bytes": (
                    buffered_bytes
                ),
                "buffered_bits": (
                    buffered_bits
                ),
                "latency_codewords_proxy": (
                    latency_codewords
                ),
                "runtime_per_frame_ms": (
                    runtime_per_frame_ms
                ),
                "monotonicity_violations": (
                    monotonicity_violations
                ),
            }

            summary.append(summary_row)

            required_text = (
                "not crossed"
                if required is None
                else f"{required:.3f} dB"
            )

            gain_text = (
                "N/A"
                if coding_gain is None
                else f"{coding_gain:.3f} dB"
            )

            print(
                f"D={depth:3d} | "
                f"required Eb/N0={required_text:>12s} | "
                f"gain={gain_text:>9s} | "
                f"buffer={buffered_bytes:6d} bytes | "
                f"runtime/frame="
                f"{runtime_per_frame_ms:.3f} ms | "
                f"monotonic violations="
                f"{monotonicity_violations}"
            )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary)

    # --------------------------------------------------------
    # Prepare summary arrays
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    draw_ber_panel(
        axes[0, 0],
        all_rows,
        coherence_symbols=8,
    )

    draw_ber_panel(
        axes[0, 1],
        all_rows,
        coherence_symbols=32,
    )

    colours = {
        8: "royalblue",
        32: "firebrick",
    }

    markers = {
        8: "o",
        32: "s",
    }

    # Required Eb/N0 panel
    required_axis = axes[1, 0]

    # Performance-cost panel
    tradeoff_axis = axes[1, 1]

    for coherence_symbols in (8, 32):
        selected = [
            row
            for row in summary
            if row["coherence_symbols"]
            == coherence_symbols
        ]

        selected.sort(
            key=lambda row:
            row["interleaver_depth"]
        )

        depths = np.array([
            row["interleaver_depth"]
            for row in selected
        ])

        required_values = np.array([
            (
                np.nan
                if row["required_ebn0_db"]
                is None
                else row["required_ebn0_db"]
            )
            for row in selected
        ])

        gains = np.array([
            (
                np.nan
                if row["interleaving_gain_db"]
                is None
                else row["interleaving_gain_db"]
            )
            for row in selected
        ])

        buffer_bytes = np.array([
            row["buffered_bytes"]
            for row in selected
        ])

        required_axis.plot(
            depths,
            required_values,
            color=colours[
                coherence_symbols
            ],
            marker=markers[
                coherence_symbols
            ],
            linewidth=1.8,
            label=(
                f"Lc={coherence_symbols}"
            ),
        )

        tradeoff_axis.plot(
            buffer_bytes,
            gains,
            color=colours[
                coherence_symbols
            ],
            marker=markers[
                coherence_symbols
            ],
            linewidth=1.8,
            label=(
                f"Lc={coherence_symbols}"
            ),
        )

        for depth, x, y in zip(
            depths,
            buffer_bytes,
            gains,
        ):
            if not np.isnan(y):
                tradeoff_axis.annotate(
                    f"D={depth}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    required_axis.set_xscale("log", base=2)

    required_axis.set_xticks(
        INTERLEAVER_DEPTHS
    )

    required_axis.set_xticklabels([
        str(depth)
        for depth in INTERLEAVER_DEPTHS
    ])

    required_axis.set_xlabel(
        "Interleaver depth D"
    )

    required_axis.set_ylabel(
        f"Required Eb/N0 at BER={TARGET_BER:.0e} (dB)"
    )

    required_axis.set_title(
        "(c) Reliability benefit"
    )

    required_axis.grid(
        True,
        alpha=0.3,
    )

    required_axis.legend()

    tradeoff_axis.set_xscale(
        "log",
        base=2,
    )

    tradeoff_axis.set_xlabel(
        "Interleaver buffer proxy (bytes)"
    )

    tradeoff_axis.set_ylabel(
        "Interleaving gain relative to D=1 (dB)"
    )

    tradeoff_axis.set_title(
        "(d) Performance-buffering trade-off"
    )

    tradeoff_axis.grid(
        True,
        alpha=0.3,
    )

    tradeoff_axis.legend()

    figure.suptitle(
        "Task 05: Effect of Channel Coherence "
        "and Interleaver Depth on RS-Coded FSO\n"
        "Moderate Gamma-Gamma turbulence, "
        "severe normalised pointing error, perfect CSI",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PNG,
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    print("\n" + "=" * 76)
    print("TASK 05 FINISHED")
    print("=" * 76)

    print(f"Summary CSV: {OUTPUT_CSV}")
    print(f"Figure PNG: {OUTPUT_PNG}")
    print(f"Figure PDF: {OUTPUT_PDF}")

    print(
        "\nInterpretation rule:"
    )

    print(
        "A useful interleaver depth is the smallest D "
        "after which additional gain becomes small."
    )

    plt.show()


if __name__ == "__main__":
    main()