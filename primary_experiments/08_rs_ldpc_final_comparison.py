"""
TASK 08
Final rate-matched RS versus LDPC comparison.

Inputs
------
1. RS fixed-coherence results:
   - 04c: Lc = 8 RS symbols = 64 transmitted bits
   - 04d: Lc = 32 RS symbols = 256 transmitted bits

2. Group-aware LDPC refinement:
   - 07B: Lc = 64 and 256 transmitted bits

Outputs
-------
1. BER overlay figure.
2. Required-Eb/N0 and engineering trade-off figure.
3. Final numerical summary CSV.
4. Human-readable findings text file.

Important fairness notes
------------------------
- RS and LDPC rates are closely matched.
- RS and LDPC codeword lengths are different.
- RS uses hard-decision decoding.
- LDPC uses soft-decision iterative decoding.
- RS and LDPC therefore represent engineering implementations,
  not a pure coding-theory comparison.
"""

from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_BER = 1e-3

DEPTHS = [1, 8, 32, 128]

# Smallest depth within this many dB of the best depth is recommended.
BEST_DEPTH_TOLERANCE_DB = 0.5

# Example channel bit rate used only to translate buffer fill into time.
# Normalised coded-bit latency is also stored, so the conclusion does not
# depend on this particular example.
EXAMPLE_CHANNEL_BIT_RATE_BPS = 1e9

RS_N_SYMBOLS = 255
RS_K_SYMBOLS = 223
RS_BITS_PER_SYMBOL = 8

RS_CODEWORD_BITS = (
    RS_N_SYMBOLS * RS_BITS_PER_SYMBOL
)

RS_INFORMATION_BITS = (
    RS_K_SYMBOLS * RS_BITS_PER_SYMBOL
)

RS_RATE = RS_K_SYMBOLS / RS_N_SYMBOLS

LDPC_CODEWORD_BITS = 512
LDPC_INFORMATION_BITS = 449
LDPC_RATE = LDPC_INFORMATION_BITS / LDPC_CODEWORD_BITS


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RS_LC64_PATH = (
    RESULTS_DIR
    / "04c_rs_fso_fixed_coherence_interleaving.csv"
)

RS_LC256_PATH = (
    RESULTS_DIR
    / "04d_rs_fso_fixed_coherence32_interleaving.csv"
)

LDPC_PATH = (
    RESULTS_DIR
    / "07b_ldpc_fso_group_aware_refinement.csv"
)

SUMMARY_CSV_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_final_summary.csv"
)

COMPARISON_PNG_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_final_comparison.png"
)

COMPARISON_PDF_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_final_comparison.pdf"
)

OVERLAY_PNG_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_ber_overlay.png"
)

OVERLAY_PDF_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_ber_overlay.pdf"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_findings.txt"
)


# =============================================================================
# CSV UTILITIES
# =============================================================================

def read_csv_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def float_or_nan(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


# =============================================================================
# LOAD AND STANDARDISE RS DATA
# =============================================================================

def load_rs_file(path, coherence_bits):
    raw_rows = read_csv_rows(path)
    standardised = []

    for raw in raw_rows:
        ber = float_or_nan(raw["BER"])
        ber_upper = float_or_nan(raw["BER_upper"])

        if np.isfinite(ber) and ber > 0:
            ber_used = ber
            ber_is_upper_bound = False
        elif np.isfinite(ber_upper) and ber_upper > 0:
            ber_used = ber_upper
            ber_is_upper_bound = True
        else:
            raise ValueError(
                f"Invalid RS BER entry in {path.name}"
            )

        fer = float_or_nan(raw["FER"])
        fer_upper = float_or_nan(raw["FER_upper"])

        if np.isfinite(fer) and fer > 0:
            fer_used = fer
        elif np.isfinite(fer_upper) and fer_upper > 0:
            fer_used = fer_upper
        else:
            fer_used = np.nan

        frames = int(float(raw["frames"]))
        runtime_seconds = float(raw["time_s"])

        runtime_per_frame_ms = (
            1000.0 * runtime_seconds / frames
        )

        standardised.append({
            "code": "RS",
            "coherence_bits": coherence_bits,
            "depth": int(float(raw["D"])),
            "ebn0_db": float(raw["EbN0_dB"]),
            "ber": ber_used,
            "ber_ci_lower": np.nan,
            "ber_ci_upper": np.nan,
            "ber_is_upper_bound": ber_is_upper_bound,
            "fer": fer_used,
            "runtime_per_frame_ms": (
                runtime_per_frame_ms
            ),
            "frames": frames,
            "independent_groups": np.nan,
        })

    return standardised


# =============================================================================
# LOAD AND STANDARDISE LDPC DATA
# =============================================================================

def load_ldpc_file(path):
    raw_rows = read_csv_rows(path)
    standardised = []

    for raw in raw_rows:
        standardised.append({
            "code": "LDPC",
            "coherence_bits": int(
                float(raw["coherence_length_bits"])
            ),
            "depth": int(
                float(raw["interleaver_depth"])
            ),
            "ebn0_db": float(raw["ebn0_db"]),
            "ber": float(raw["information_ber"]),
            "ber_ci_lower": float(
                raw["information_ber_ci_lower"]
            ),
            "ber_ci_upper": float(
                raw["information_ber_ci_upper"]
            ),
            "ber_is_upper_bound": False,
            "fer": float(raw["fer"]),
            "runtime_per_frame_ms": float(
                raw["runtime_per_frame_ms"]
            ),
            "frames": int(float(raw["frames"])),
            "independent_groups": int(
                float(raw["independent_groups"])
            ),
        })

    return standardised


# =============================================================================
# INTERPOLATION
# =============================================================================

def log_crossing(x_values, y_values, target):
    """
    Find the x value where a decreasing positive y curve crosses target.

    Linear interpolation is performed in log10(y), which is appropriate
    for semilog BER plots.
    """

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (y > 0)
    )

    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for index in range(len(x) - 1):
        y_left = y[index]
        y_right = y[index + 1]

        crossed = (
            y_left >= target
            and y_right <= target
        )

        if crossed:
            if y_left == y_right:
                return float(
                    0.5 * (x[index] + x[index + 1])
                )

            log_left = np.log10(y_left)
            log_right = np.log10(y_right)
            log_target = np.log10(target)

            fraction = (
                (log_target - log_left)
                / (log_right - log_left)
            )

            return float(
                x[index]
                + fraction
                * (x[index + 1] - x[index])
            )

    return np.nan


def log_interpolate(x_values, y_values, x_target):
    """Interpolate a positive y value in log10(y)."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (y > 0)
    )

    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if not np.isfinite(x_target):
        return np.nan

    if x_target < x.min() or x_target > x.max():
        return np.nan

    return float(
        10.0 ** np.interp(
            x_target,
            x,
            np.log10(y),
        )
    )


def linear_interpolate(x_values, y_values, x_target):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if not np.isfinite(x_target):
        return np.nan

    if x_target < x.min() or x_target > x.max():
        return np.nan

    return float(
        np.interp(x_target, x, y)
    )


# =============================================================================
# CURVE ANALYSIS
# =============================================================================

def select_curve(rows, code, coherence_bits, depth):
    selected = [
        row for row in rows
        if row["code"] == code
        and row["coherence_bits"] == coherence_bits
        and row["depth"] == depth
    ]

    selected.sort(
        key=lambda row: row["ebn0_db"]
    )

    if not selected:
        raise ValueError(
            f"No curve found for "
            f"{code}, Lc={coherence_bits}, D={depth}"
        )

    return selected


def count_monotonicity_violations(ber_values):
    ber = np.asarray(ber_values, dtype=float)

    # BER should not increase as Eb/N0 increases.
    return int(
        np.count_nonzero(
            np.diff(ber) > 0
        )
    )


def analyse_curve(
    all_rows,
    code,
    coherence_bits,
    depth,
):
    curve = select_curve(
        all_rows,
        code,
        coherence_bits,
        depth,
    )

    ebn0 = np.array(
        [row["ebn0_db"] for row in curve]
    )

    ber = np.array(
        [row["ber"] for row in curve]
    )

    fer = np.array(
        [row["fer"] for row in curve]
    )

    runtime_ms = np.array(
        [
            row["runtime_per_frame_ms"]
            for row in curve
        ]
    )

    required_ebn0 = log_crossing(
        ebn0,
        ber,
        TARGET_BER,
    )

    if not np.isfinite(required_ebn0):
        raise RuntimeError(
            f"Target BER={TARGET_BER:.1e} is not "
            f"bracketed for {code}, "
            f"Lc={coherence_bits}, D={depth}."
        )

    # LDPC group-aware confidence bounds.
    if code == "LDPC":
        lower_curve = np.array(
            [
                row["ber_ci_lower"]
                for row in curve
            ]
        )

        upper_curve = np.array(
            [
                row["ber_ci_upper"]
                for row in curve
            ]
        )

        # Optimistic lower BER bound crosses first.
        required_ebn0_optimistic = log_crossing(
            ebn0,
            lower_curve,
            TARGET_BER,
        )

        # Pessimistic upper BER bound crosses later.
        required_ebn0_pessimistic = log_crossing(
            ebn0,
            upper_curve,
            TARGET_BER,
        )

    else:
        required_ebn0_optimistic = np.nan
        required_ebn0_pessimistic = np.nan

    fer_at_target = log_interpolate(
        ebn0,
        fer,
        required_ebn0,
    )

    runtime_at_target_ms = linear_interpolate(
        ebn0,
        runtime_ms,
        required_ebn0,
    )

    if code == "RS":
        codeword_bits = RS_CODEWORD_BITS
        information_bits = RS_INFORMATION_BITS
        code_rate = RS_RATE
        decoder_type = "Hard-decision algebraic"
    else:
        codeword_bits = LDPC_CODEWORD_BITS
        information_bits = LDPC_INFORMATION_BITS
        code_rate = LDPC_RATE
        decoder_type = "Soft-decision iterative"

    buffer_bits = depth * codeword_bits
    buffer_bytes = buffer_bits / 8.0
    buffer_kib = buffer_bytes / 1024.0

    # Waiting for D codewords instead of one.
    additional_latency_coded_bits = (
        (depth - 1) * codeword_bits
    )

    example_latency_microseconds = (
        1e6
        * additional_latency_coded_bits
        / EXAMPLE_CHANNEL_BIT_RATE_BPS
    )

    # Cross-code bit-goodput proxy. This is based on BER rather than FER,
    # because an RS information block contains 1784 bits while an LDPC
    # information block contains 449 bits. Their codeword FER values are
    # therefore not directly comparable as application-packet failure rates.
    bit_goodput_per_channel_bit = (
        code_rate * (1.0 - TARGET_BER)
    )

    # Per-frame runtime is also not a fair cross-code cost because the two
    # information blocks have different lengths. Normalise it to the time
    # required to process 1000 information bits.
    runtime_per_1000_information_bits_ms = (
        runtime_at_target_ms
        * 1000.0
        / information_bits
    )

    # bits/ms is numerically equal to kbit/s.
    software_information_throughput_kbps = (
        information_bits / runtime_at_target_ms
        if runtime_at_target_ms > 0
        else np.nan
    )

    violations = count_monotonicity_violations(
        ber
    )

    return {
        "code": code,
        "decoder_type": decoder_type,
        "coherence_bits": coherence_bits,
        "interleaver_depth": depth,
        "target_ber": TARGET_BER,
        "required_ebn0_db": required_ebn0,
        "required_ebn0_optimistic_db": (
            required_ebn0_optimistic
        ),
        "required_ebn0_pessimistic_db": (
            required_ebn0_pessimistic
        ),
        "interleaving_gain_db": np.nan,
        "code_rate": code_rate,
        "coded_bits_per_codeword": codeword_bits,
        "information_bits_per_codeword": (
            information_bits
        ),
        "buffer_bits": buffer_bits,
        "buffer_bytes": buffer_bytes,
        "buffer_kib": buffer_kib,
        "additional_latency_coded_bits": (
            additional_latency_coded_bits
        ),
        "example_bit_rate_bps": (
            EXAMPLE_CHANNEL_BIT_RATE_BPS
        ),
        "example_additional_latency_us": (
            example_latency_microseconds
        ),
        "codeword_fer_at_target_ber": fer_at_target,
        "fer_at_target_ber": fer_at_target,
        "bit_goodput_per_channel_bit_at_target": (
            bit_goodput_per_channel_bit
        ),
        "runtime_per_frame_at_target_ms": (
            runtime_at_target_ms
        ),
        "runtime_per_1000_information_bits_ms": (
            runtime_per_1000_information_bits_ms
        ),
        "software_information_throughput_kbps": (
            software_information_throughput_kbps
        ),
        "monotonicity_violations": violations,
        "within_0_5_db_of_best": False,
        "pareto_optimal_within_coherence": False,
    }


# =============================================================================
# PARETO ANALYSIS
# =============================================================================

def dominates(row_a, row_b):
    """
    Return True if A is no worse in all selected engineering objectives
    and strictly better in at least one.

    Minimise three cross-code comparable quantities:
    - required Eb/N0
    - interleaver buffer bits
    - measured runtime per 1000 information bits

    Codeword FER is deliberately excluded because RS and LDPC codewords
    contain different numbers of information bits.
    """

    minimise_a = [
        row_a["required_ebn0_db"],
        row_a["buffer_bits"],
        row_a["runtime_per_1000_information_bits_ms"],
    ]

    minimise_b = [
        row_b["required_ebn0_db"],
        row_b["buffer_bits"],
        row_b["runtime_per_1000_information_bits_ms"],
    ]

    no_worse = all(
        a <= b
        for a, b in zip(
            minimise_a,
            minimise_b,
        )
    )

    strictly_better = any(
        a < b
        for a, b in zip(
            minimise_a,
            minimise_b,
        )
    )

    return no_worse and strictly_better


def mark_pareto_rows(summary_rows):
    for coherence_bits in [64, 256]:
        subset = [
            row for row in summary_rows
            if row["coherence_bits"] == coherence_bits
        ]

        for candidate in subset:
            dominated = any(
                dominates(other, candidate)
                for other in subset
                if other is not candidate
            )

            candidate[
                "pareto_optimal_within_coherence"
            ] = not dominated


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_summary_csv(rows):
    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# FIGURE 1: BER OVERLAY
# =============================================================================

def plot_ber_overlay(all_rows):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharey=True,
    )

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

    for axis, coherence_bits in zip(
        axes,
        [64, 256],
    ):
        for depth in DEPTHS:
            colour = colours[depth]
            marker = markers[depth]

            rs_curve = select_curve(
                all_rows,
                "RS",
                coherence_bits,
                depth,
            )

            ldpc_curve = select_curve(
                all_rows,
                "LDPC",
                coherence_bits,
                depth,
            )

            rs_x = np.array(
                [row["ebn0_db"] for row in rs_curve]
            )

            rs_y = np.array(
                [row["ber"] for row in rs_curve]
            )

            ldpc_x = np.array(
                [row["ebn0_db"] for row in ldpc_curve]
            )

            ldpc_y = np.array(
                [row["ber"] for row in ldpc_curve]
            )

            ldpc_lower = np.array(
                [
                    row["ber_ci_lower"]
                    for row in ldpc_curve
                ]
            )

            ldpc_upper = np.array(
                [
                    row["ber_ci_upper"]
                    for row in ldpc_curve
                ]
            )

            axis.semilogy(
                rs_x,
                rs_y,
                color=colour,
                linestyle="--",
                marker=marker,
                markersize=4,
                linewidth=1.6,
                alpha=0.8,
                label=f"RS D={depth}",
            )

            axis.semilogy(
                ldpc_x,
                ldpc_y,
                color=colour,
                linestyle="-",
                marker=marker,
                markersize=5,
                linewidth=2.2,
                label=f"LDPC D={depth}",
            )

            axis.fill_between(
                ldpc_x,
                np.maximum(ldpc_lower, 1e-7),
                np.maximum(ldpc_upper, 1e-7),
                color=colour,
                alpha=0.08,
            )

        axis.axhline(
            TARGET_BER,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label="Target BER=1e-3",
        )

        axis.set_title(
            f"Physical coherence length "
            f"Lc={coherence_bits} bits"
        )

        axis.set_xlabel(
            "Information-bit Eb/N0 (dB)"
        )

        axis.grid(
            True,
            which="both",
            alpha=0.32,
        )

        axis.set_xlim(18, 37)
        axis.set_ylim(1e-5, 3e-2)

    axes[0].set_ylabel(
        "Decoded information BER"
    )

    axes[1].legend(
        fontsize=8,
        ncol=2,
        loc="upper right",
    )

    fig.suptitle(
        "Task 08: Rate-Matched RS and LDPC over the Same "
        "Fixed-Coherence FSO Channel\n"
        "Dashed: hard-decision RS; solid and shaded: "
        "group-aware soft-decision LDPC",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.90)
    )

    fig.savefig(
        OVERLAY_PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        OVERLAY_PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# FIGURE 2: FINAL TRADE-OFF
# =============================================================================

def plot_final_comparison(summary_rows):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
    )

    code_styles = {
        "RS": {
            "marker": "o",
            "linestyle": "--",
        },
        "LDPC": {
            "marker": "s",
            "linestyle": "-",
        },
    }

    coherence_colours = {
        64: "royalblue",
        256: "firebrick",
    }

    # -------------------------------------------------------------------------
    # Required Eb/N0 for Lc=64
    # -------------------------------------------------------------------------

    for panel_index, coherence_bits in enumerate(
        [64, 256]
    ):
        axis = axes[0, panel_index]

        for code in ["RS", "LDPC"]:
            selected = [
                row for row in summary_rows
                if row["code"] == code
                and row["coherence_bits"]
                == coherence_bits
            ]

            selected.sort(
                key=lambda row:
                row["interleaver_depth"]
            )

            depths = np.array(
                [
                    row["interleaver_depth"]
                    for row in selected
                ]
            )

            thresholds = np.array(
                [
                    row["required_ebn0_db"]
                    for row in selected
                ]
            )

            code_colour = (
                "tab:blue" if code == "RS" else "tab:orange"
            )

            axis.plot(
                depths,
                thresholds,
                color=code_colour,
                marker=code_styles[code]["marker"],
                linestyle=(
                    code_styles[code]["linestyle"]
                ),
                linewidth=2,
                markersize=7,
                label=code,
            )

            if code == "LDPC":
                optimistic = np.array(
                    [
                        row[
                            "required_ebn0_optimistic_db"
                        ]
                        for row in selected
                    ]
                )

                pessimistic = np.array(
                    [
                        row[
                            "required_ebn0_pessimistic_db"
                        ]
                        for row in selected
                    ]
                )

                finite = (
                    np.isfinite(optimistic)
                    & np.isfinite(pessimistic)
                )

                if np.any(finite):
                    axis.fill_between(
                        depths[finite],
                        optimistic[finite],
                        pessimistic[finite],
                        color=code_colour,
                        alpha=0.16,
                    )

        axis.set_xscale("log", base=2)

        axis.set_xticks(DEPTHS)
        axis.set_xticklabels(
            [str(depth) for depth in DEPTHS]
        )

        axis.set_xlabel("Interleaver depth D")
        axis.set_ylabel(
            "Required Eb/N0 at BER=1e-3 (dB)"
        )

        axis.set_title(
            f"(a{panel_index + 1}) "
            f"Lc={coherence_bits} transmitted bits"
        )

        axis.grid(True, alpha=0.35)
        axis.legend()

    # -------------------------------------------------------------------------
    # Interleaving gain versus buffer
    # -------------------------------------------------------------------------

    gain_axis = axes[1, 0]

    for row in summary_rows:
        colour = coherence_colours[
            row["coherence_bits"]
        ]

        marker = (
            "o" if row["code"] == "RS" else "s"
        )

        gain_axis.scatter(
            row["buffer_kib"],
            row["interleaving_gain_db"],
            color=colour,
            marker=marker,
            s=65,
            edgecolor="black",
            linewidth=0.5,
        )

        gain_axis.annotate(
            f"{row['code']} D={row['interleaver_depth']}",
            (
                row["buffer_kib"],
                row["interleaving_gain_db"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    gain_axis.set_xscale("log", base=2)
    gain_axis.set_xlabel(
        "Total interleaver buffer proxy (KiB)"
    )

    gain_axis.set_ylabel(
        "Interleaving gain relative to D=1 (dB)"
    )

    gain_axis.set_title(
        "(b) Reliability gain versus memory"
    )

    gain_axis.grid(True, alpha=0.35)

    gain_axis.text(
        0.02,
        0.97,
        "Blue: Lc=64 bits\nRed: Lc=256 bits\n"
        "Circle: RS\nSquare: LDPC",
        transform=gain_axis.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "grey",
        },
    )

    # -------------------------------------------------------------------------
    # Reliability/runtime trade-off using a common information-bit unit
    # -------------------------------------------------------------------------

    trade_axis = axes[1, 1]

    for row in summary_rows:
        colour = coherence_colours[
            row["coherence_bits"]
        ]

        marker = (
            "o" if row["code"] == "RS" else "s"
        )

        size = (
            45
            + 15
            * np.log2(
                1.0 + row["buffer_kib"]
            )
        )

        trade_axis.scatter(
            row["runtime_per_1000_information_bits_ms"],
            row["required_ebn0_db"],
            color=colour,
            marker=marker,
            s=size,
            edgecolor=(
                "gold"
                if row[
                    "pareto_optimal_within_coherence"
                ]
                else "black"
            ),
            linewidth=(
                2.0
                if row[
                    "pareto_optimal_within_coherence"
                ]
                else 0.5
            ),
        )

        annotation_offsets = {
            ("RS", 1, 64): (5, 7),
            ("RS", 8, 64): (5, 10),
            ("RS", 32, 64): (5, -13),
            ("RS", 128, 64): (30, -2),
            ("RS", 1, 256): (5, 7),
            ("RS", 8, 256): (5, 10),
            ("RS", 32, 256): (5, -13),
            ("RS", 128, 256): (30, -2),
            ("LDPC", 1, 64): (5, 7),
            ("LDPC", 8, 64): (5, 7),
            ("LDPC", 32, 64): (5, -13),
            ("LDPC", 128, 64): (5, 7),
            ("LDPC", 1, 256): (5, 7),
            ("LDPC", 8, 256): (5, 7),
            ("LDPC", 32, 256): (5, 7),
            ("LDPC", 128, 256): (5, -13),
        }

        annotation_offset = annotation_offsets.get(
            (
                row["code"],
                row["interleaver_depth"],
                row["coherence_bits"],
            ),
            (4, 4),
        )

        trade_axis.annotate(
            f"{row['code']} D={row['interleaver_depth']}",
            (
                row[
                    "runtime_per_1000_information_bits_ms"
                ],
                row["required_ebn0_db"],
            ),
            xytext=annotation_offset,
            textcoords="offset points",
            fontsize=7,
        )

    trade_axis.set_xscale("log")

    trade_axis.set_xlabel(
        "Measured runtime per 1000 information bits "
        "near target BER (ms)"
    )

    trade_axis.set_ylabel(
        "Required Eb/N0 at BER=1e-3 (dB)"
    )

    trade_axis.set_title(
        "(c) Reliability versus normalised software cost"
    )

    trade_axis.grid(
        True,
        which="both",
        alpha=0.35,
    )

    trade_axis.text(
        0.02,
        0.97,
        "Gold edge: empirical Pareto point\n"
        "Marker size increases with buffer",
        transform=trade_axis.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "grey",
        },
    )

    fig.suptitle(
        "Task 08: RS-LDPC Reliability and "
        "Implementation Trade-off\n"
        "Moderate Gamma-Gamma turbulence, severe normalised "
        "pointing error, perfect CSI",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    fig.savefig(
        COMPARISON_PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        COMPARISON_PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# FINDINGS REPORT
# =============================================================================

def recommended_depth(rows, code, coherence_bits):
    selected = [
        row for row in rows
        if row["code"] == code
        and row["coherence_bits"] == coherence_bits
        and row["within_0_5_db_of_best"]
    ]

    selected.sort(
        key=lambda row:
        row["interleaver_depth"]
    )

    if not selected:
        return None

    return selected[0]


def write_findings(summary_rows):
    lines = []

    lines.append(
        "TASK 08: FINAL RS-LDPC COMPARISON"
    )

    lines.append("=" * 72)
    lines.append("")

    lines.append(
        f"Target BER: {TARGET_BER:.1e}"
    )

    lines.append(
        f"RS rate: {RS_RATE:.6f}"
    )

    lines.append(
        f"LDPC rate: {LDPC_RATE:.6f}"
    )

    lines.append(
        f"Rate difference: "
        f"{LDPC_RATE - RS_RATE:+.6f}"
    )

    lines.append("")

    for coherence_bits in [64, 256]:
        lines.append("-" * 72)

        lines.append(
            f"Physical coherence length "
            f"Lc={coherence_bits} transmitted bits"
        )

        lines.append("-" * 72)

        for code in ["RS", "LDPC"]:
            lines.append(f"\n{code}")

            selected = [
                row for row in summary_rows
                if row["code"] == code
                and row["coherence_bits"]
                == coherence_bits
            ]

            selected.sort(
                key=lambda row:
                row["interleaver_depth"]
            )

            for row in selected:
                lines.append(
                    f"  D={row['interleaver_depth']:3d} | "
                    f"required Eb/N0="
                    f"{row['required_ebn0_db']:7.3f} dB | "
                    f"gain="
                    f"{row['interleaving_gain_db']:6.3f} dB | "
                    f"codeword FER@target="
                    f"{row['codeword_fer_at_target_ber']:.3e} | "
                    f"buffer="
                    f"{row['buffer_kib']:.3f} KiB | "
                    f"normalised runtime="
                    f"{row['runtime_per_1000_information_bits_ms']:.3f} "
                    f"ms/1000 info bits"
                )

            recommendation = recommended_depth(
                summary_rows,
                code,
                coherence_bits,
            )

            if recommendation is not None:
                lines.append(
                    f"  Recommended smallest depth within "
                    f"{BEST_DEPTH_TOLERANCE_DB:.1f} dB of best: "
                    f"D={recommendation['interleaver_depth']}"
                )

        lines.append("")

    lines.append("=" * 72)

    lines.append("INTERPRETATION LIMITS")

    lines.append("=" * 72)

    lines.append(
        "1. RS and LDPC rates are closely matched, but codeword "
        "lengths differ."
    )

    lines.append(
        "2. RS uses hard-decision algebraic decoding; LDPC uses "
        "soft-decision iterative decoding."
    )

    lines.append(
        "3. LDPC confidence intervals are group-aware. The existing "
        "RS dataset does not contain equivalent group-aware intervals."
    )

    lines.append(
        "4. Runtime is implementation- and library-dependent. It is "
        "normalised per 1000 information bits, but remains an empirical "
        "software cost rather than an operation-count proof."
    )

    lines.append(
        "5. Buffer and latency values are engineering proxies. Actual "
        "hardware scheduling may change absolute latency."
    )

    lines.append(
        "6. RS and LDPC codeword FER values are retained only as "
        "within-code diagnostics. They are not directly compared because "
        "the information-block lengths differ (1784 versus 449 bits)."
    )

    lines.append(
        "7. Bit-goodput is computed as R*(1-target BER), using a common "
        "bit-level definition rather than codeword FER."
    )

    lines.append(
        "8. Uncoded coding gain is not reported because the existing "
        "severe-pointing uncoded sweep does not bracket BER=1e-3."
    )

    with FINDINGS_PATH.open(
        "w",
        encoding="utf-8",
    ) as findings_file:
        findings_file.write(
            "\n".join(lines)
        )


# =============================================================================
# TERMINAL SUMMARY
# =============================================================================

def print_summary(rows):
    print("\n" + "=" * 120)

    print(
        "FINAL REQUIRED Eb/N0 AT BER=1e-3"
    )

    print("=" * 120)

    header = (
        f"{'Code':<6}"
        f"{'Lc(bits)':>10}"
        f"{'D':>6}"
        f"{'Eb/N0(dB)':>13}"
        f"{'Gain(dB)':>12}"
        f"{'CW-FER':>13}"
        f"{'Buffer(KiB)':>15}"
        f"{'ms/1000bits':>16}"
        f"{'Near best':>12}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['code']:<6}"
            f"{row['coherence_bits']:>10d}"
            f"{row['interleaver_depth']:>6d}"
            f"{row['required_ebn0_db']:>13.3f}"
            f"{row['interleaving_gain_db']:>12.3f}"
            f"{row['codeword_fer_at_target_ber']:>13.3e}"
            f"{row['buffer_kib']:>15.3f}"
            f"{row['runtime_per_1000_information_bits_ms']:>16.3f}"
            f"{str(row['within_0_5_db_of_best']):>12}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 88)
    print("TASK 08: FINAL RS-LDPC COMPARISON")
    print("=" * 88)

    print("\n[1] Checking required input files...")

    for path in [
        RS_LC64_PATH,
        RS_LC256_PATH,
        LDPC_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        print(f"Found: {path.name}")

    print("\n[2] Loading and standardising data...")

    all_rows = []

    all_rows.extend(
        load_rs_file(
            RS_LC64_PATH,
            coherence_bits=64,
        )
    )

    all_rows.extend(
        load_rs_file(
            RS_LC256_PATH,
            coherence_bits=256,
        )
    )

    all_rows.extend(
        load_ldpc_file(LDPC_PATH)
    )

    print(
        f"Loaded {len(all_rows)} standardised "
        f"curve points."
    )

    print("\n[3] Analysing target-BER crossings...")

    summary_rows = []

    for coherence_bits in [64, 256]:
        for code in ["RS", "LDPC"]:
            for depth in DEPTHS:
                row = analyse_curve(
                    all_rows,
                    code,
                    coherence_bits,
                    depth,
                )

                summary_rows.append(row)

    # Interleaving gain relative to the same code and Lc at D=1.
    for coherence_bits in [64, 256]:
        for code in ["RS", "LDPC"]:
            subset = [
                row for row in summary_rows
                if row["code"] == code
                and row["coherence_bits"]
                == coherence_bits
            ]

            baseline = next(
                row for row in subset
                if row["interleaver_depth"] == 1
            )

            best_required_ebn0 = min(
                row["required_ebn0_db"]
                for row in subset
            )

            for row in subset:
                row["interleaving_gain_db"] = (
                    baseline["required_ebn0_db"]
                    - row["required_ebn0_db"]
                )

                row["within_0_5_db_of_best"] = bool(
                    row["required_ebn0_db"]
                    <= best_required_ebn0
                    + BEST_DEPTH_TOLERANCE_DB
                )

    mark_pareto_rows(summary_rows)

    summary_rows.sort(
        key=lambda row: (
            row["coherence_bits"],
            row["code"],
            row["interleaver_depth"],
        )
    )

    # Acceptance checks.
    total_violations = sum(
        row["monotonicity_violations"]
        for row in summary_rows
    )

    if total_violations > 0:
        print(
            f"WARNING: {total_violations} total "
            f"monotonicity violations detected."
        )
    else:
        print(
            "All formal BER curves are monotonic."
        )

    print("\n[4] Saving summary and figures...")

    save_summary_csv(summary_rows)
    plot_ber_overlay(all_rows)
    plot_final_comparison(summary_rows)
    write_findings(summary_rows)

    print_summary(summary_rows)

    print("\n" + "=" * 88)
    print("TASK 08 FINISHED")
    print("=" * 88)

    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"BER overlay PNG: {OVERLAY_PNG_PATH}")
    print(f"BER overlay PDF: {OVERLAY_PDF_PATH}")
    print(
        f"Final comparison PNG: "
        f"{COMPARISON_PNG_PATH}"
    )
    print(
        f"Final comparison PDF: "
        f"{COMPARISON_PDF_PATH}"
    )
    print(f"Findings text: {FINDINGS_PATH}")

    print(
        "\nImportant: Task 08 reports interleaving gain "
        "relative to D=1."
    )

    print(
        "It does not report uncoded coding gain because "
        "the current uncoded severe-pointing sweep does "
        "not reach BER=1e-3."
    )


if __name__ == "__main__":
    main()
