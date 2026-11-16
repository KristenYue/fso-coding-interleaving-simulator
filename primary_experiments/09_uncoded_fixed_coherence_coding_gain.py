"""
TASK 09
Group-aware uncoded fixed-coherence FSO baseline and coding gain.

Purpose
-------
1. Extend the uncoded severe-pointing BER curve beyond 30 dB.
2. Ensure BER=1e-3 is bracketed.
3. Use the same physical coherence lengths as the coded experiments.
4. Compute true coding gain:

    coding gain =
    required uncoded Eb/N0 - required coded Eb/N0

Channel
-------
- IM/DD OOK
- Moderate Gamma-Gamma turbulence
- Severe normalised pointing error
- Perfect CSI
- Information-bit Eb/N0
- Fixed coherence lengths: 64 and 256 transmitted bits
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

MASTER_SEED = 20260621
SHOW_FIGURE = True

# The 12 formal Monte Carlo points already exist. Leave this True to reuse
# them and only redo the corrected pooled analysis. Set False only when a
# completely new uncoded simulation is intentionally required.
REUSE_EXISTING_UNCODED_CSV = True

TARGET_BER = 1e-3

COHERENCE_LENGTHS_BITS = [64, 256]
DEPTHS = [1, 8, 32, 128]

EBN0_DB_LIST = np.array(
    [24, 27, 30, 33, 36, 39],
    dtype=float,
)

# Independent statistical unit = one coherence block.
MIN_INDEPENDENT_BLOCKS = 5000
MIN_BIT_ERRORS = 400
MAX_INDEPENDENT_BLOCKS = 50000

BATCH_BLOCKS = 500

# Validated moderate Gamma-Gamma parameters
ALPHA_MODERATE = 4.868481548059242
BETA_MODERATE = 3.164867988481172

# Validated severe normalised pointing-error parameter
XI_SEVERE = 1.522040256461071


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TASK08_SUMMARY_PATH = (
    RESULTS_DIR
    / "08_rs_ldpc_final_summary.csv"
)

UNCODED_CSV_PATH = (
    RESULTS_DIR
    / "09_uncoded_fixed_coherence.csv"
)

POOLED_UNCODED_CSV_PATH = (
    RESULTS_DIR
    / "09_uncoded_pooled_group_aware.csv"
)

UNCODED_PNG_PATH = (
    RESULTS_DIR
    / "09_uncoded_fixed_coherence.png"
)

UNCODED_PDF_PATH = (
    RESULTS_DIR
    / "09_uncoded_fixed_coherence.pdf"
)

CODING_GAIN_CSV_PATH = (
    RESULTS_DIR
    / "09_coding_gain_summary.csv"
)

CODING_GAIN_PNG_PATH = (
    RESULTS_DIR
    / "09_coding_gain_summary.png"
)

CODING_GAIN_PDF_PATH = (
    RESULTS_DIR
    / "09_coding_gain_summary.pdf"
)


# =============================================================================
# CHANNEL
# =============================================================================

def sample_gamma_gamma(
    alpha,
    beta,
    number_samples,
    rng,
):
    large_scale = rng.gamma(
        shape=alpha,
        scale=1.0 / alpha,
        size=number_samples,
    )

    small_scale = rng.gamma(
        shape=beta,
        scale=1.0 / beta,
        size=number_samples,
    )

    return large_scale * small_scale


def sample_normalised_pointing_gain(
    xi,
    number_samples,
    rng,
):
    uniform_samples = rng.random(number_samples)

    return uniform_samples ** (
        1.0 / xi ** 2
    )


def sample_fso_block_gain(number_blocks, rng):
    turbulence = sample_gamma_gamma(
        ALPHA_MODERATE,
        BETA_MODERATE,
        number_blocks,
        rng,
    )

    pointing = sample_normalised_pointing_gain(
        XI_SEVERE,
        number_blocks,
        rng,
    )

    return turbulence * pointing


# =============================================================================
# GROUP-AWARE CONFIDENCE INTERVAL
# =============================================================================

def group_confidence_interval(group_values):
    """
    Calculate a 95% confidence interval using independent coherence blocks.

    Bits inside one coherence block are correlated through their common
    channel gain. Different blocks are independently generated.
    """

    values = np.asarray(
        group_values,
        dtype=float,
    )

    number_groups = values.size

    if number_groups == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

    if number_groups == 1:
        return mean_value, 0.0, 1.0

    standard_error = float(
        np.std(values, ddof=1)
        / np.sqrt(number_groups)
    )

    half_width = 1.96 * standard_error

    lower = max(
        0.0,
        mean_value - half_width,
    )

    upper = min(
        1.0,
        mean_value + half_width,
    )

    return mean_value, lower, upper


# =============================================================================
# SIMULATE ONE CONDITION
# =============================================================================

def simulate_condition(
    coherence_length_bits,
    ebn0_db,
    rng,
):
    """
    Simulate uncoded OOK under fixed-coherence block fading.

    For uncoded OOK, R=1:

        sigma^2 = 1 / (4 Eb/N0)

    Perfect-CSI hard threshold:

        threshold = h/2
    """

    ebn0_linear = 10.0 ** (
        ebn0_db / 10.0
    )

    noise_variance = 1.0 / (
        4.0 * ebn0_linear
    )

    noise_standard_deviation = np.sqrt(
        noise_variance
    )

    total_blocks = 0
    total_bits = 0
    total_bit_errors = 0
    total_error_blocks = 0

    block_ber_values = []
    block_error_indicators = []

    while total_blocks < MAX_INDEPENDENT_BLOCKS:
        remaining_blocks = (
            MAX_INDEPENDENT_BLOCKS
            - total_blocks
        )

        current_batch_blocks = min(
            BATCH_BLOCKS,
            remaining_blocks,
        )

        bits = rng.integers(
            0,
            2,
            size=(
                current_batch_blocks,
                coherence_length_bits,
            ),
            dtype=np.int8,
        )

        block_gain = sample_fso_block_gain(
            current_batch_blocks,
            rng,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=bits.shape,
        )

        received = (
            block_gain[:, None] * bits
            + noise
        )

        detected_bits = (
            received
            >= 0.5 * block_gain[:, None]
        ).astype(np.int8)

        error_matrix = (
            detected_bits != bits
        )

        errors_per_block = np.count_nonzero(
            error_matrix,
            axis=1,
        )

        batch_bit_errors = int(
            np.sum(errors_per_block)
        )

        batch_error_blocks = int(
            np.count_nonzero(
                errors_per_block > 0
            )
        )

        block_ber_values.extend(
            (
                errors_per_block
                / coherence_length_bits
            ).tolist()
        )

        block_error_indicators.extend(
            (
                errors_per_block > 0
            ).astype(float).tolist()
        )

        total_blocks += current_batch_blocks

        total_bits += (
            current_batch_blocks
            * coherence_length_bits
        )

        total_bit_errors += batch_bit_errors
        total_error_blocks += batch_error_blocks

        enough_blocks = (
            total_blocks
            >= MIN_INDEPENDENT_BLOCKS
        )

        enough_errors = (
            total_bit_errors
            >= MIN_BIT_ERRORS
        )

        if enough_blocks and enough_errors:
            break

    (
        ber,
        ber_ci_lower,
        ber_ci_upper,
    ) = group_confidence_interval(
        block_ber_values
    )

    (
        block_error_probability,
        block_error_ci_lower,
        block_error_ci_upper,
    ) = group_confidence_interval(
        block_error_indicators
    )

    return {
        "coherence_length_bits": (
            coherence_length_bits
        ),
        "ebn0_db": float(ebn0_db),
        "independent_blocks": total_blocks,
        "total_bits": total_bits,
        "bit_errors": total_bit_errors,
        "ber": ber,
        "ber_ci_lower": ber_ci_lower,
        "ber_ci_upper": ber_ci_upper,
        "error_blocks": total_error_blocks,
        "coherence_block_error_probability": (
            block_error_probability
        ),
        "block_error_ci_lower": (
            block_error_ci_lower
        ),
        "block_error_ci_upper": (
            block_error_ci_upper
        ),
    }


# =============================================================================
# INTERPOLATION
# =============================================================================

def log_crossing(x_values, y_values, target):
    """
    Find target crossing using linear interpolation in log10(y).
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
        if (
            y[index] >= target
            and y[index + 1] <= target
        ):
            log_left = np.log10(y[index])
            log_right = np.log10(
                y[index + 1]
            )

            log_target = np.log10(target)

            if log_left == log_right:
                return float(
                    0.5
                    * (
                        x[index]
                        + x[index + 1]
                    )
                )

            fraction = (
                (log_target - log_left)
                / (log_right - log_left)
            )

            return float(
                x[index]
                + fraction
                * (
                    x[index + 1]
                    - x[index]
                )
            )

    return np.nan


# =============================================================================
# CSV FUNCTIONS
# =============================================================================

def save_rows(path, rows):
    if not rows:
        return

    with path.open(
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


def read_uncoded_rows(path):
    """Read the already completed formal uncoded Monte Carlo data."""

    if not path.exists():
        raise FileNotFoundError(
            f"Existing uncoded CSV was not found:\n{path}"
        )

    integer_fields = {
        "coherence_length_bits",
        "independent_blocks",
        "total_bits",
        "bit_errors",
        "error_blocks",
    }

    rows = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for raw in reader:
            converted = {}

            for key, value in raw.items():
                if key in integer_fields:
                    converted[key] = int(float(value))
                else:
                    converted[key] = float(value)

            rows.append(converted)

    return rows


def read_task08_summary():
    if not TASK08_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Task 08 summary was not found:\n"
            f"{TASK08_SUMMARY_PATH}"
        )

    rows = []

    with TASK08_SUMMARY_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for raw in reader:
            rows.append({
                "code": raw["code"],
                "coherence_bits": int(
                    float(raw["coherence_bits"])
                ),
                "interleaver_depth": int(
                    float(raw["interleaver_depth"])
                ),
                "coded_required_ebn0_db": float(
                    raw["required_ebn0_db"]
                ),
                "buffer_kib": float(
                    raw["buffer_kib"]
                ),
                "runtime_per_1000_information_bits_ms": float(
                    raw[
                        "runtime_per_1000_information_bits_ms"
                    ]
                ),
            })

    return rows


# =============================================================================
# ANALYSE UNCODED THRESHOLDS
# =============================================================================

def calculate_pooled_uncoded_curve(rows):
    """
    Pool the Lc=64 and Lc=256 BER estimates at each Eb/N0.

    Coherence length changes error correlation and block-error probability,
    but it does not change the marginal BER expectation. Each individual
    curve is a noisy estimate of the same marginal BER.

    The two group-aware estimates are combined using fixed-effect
    inverse-variance weighting. The standard error of each estimate is
    reconstructed from its saved 95% confidence interval.
    """

    pooled_rows = []

    for ebn0_db in sorted({row["ebn0_db"] for row in rows}):
        selected = [
            row for row in rows
            if row["ebn0_db"] == ebn0_db
        ]

        estimates = []
        variances = []

        for row in selected:
            half_width = (
                row["ber_ci_upper"]
                - row["ber_ci_lower"]
            ) / 2.0

            standard_error = half_width / 1.96

            if standard_error <= 0 or not np.isfinite(standard_error):
                raise RuntimeError(
                    f"Invalid BER confidence interval at {ebn0_db} dB."
                )

            estimates.append(row["ber"])
            variances.append(standard_error ** 2)

        weights = 1.0 / np.asarray(variances, dtype=float)
        estimates = np.asarray(estimates, dtype=float)

        pooled_ber = float(
            np.sum(weights * estimates) / np.sum(weights)
        )

        pooled_standard_error = float(
            np.sqrt(1.0 / np.sum(weights))
        )

        pooled_lower = max(
            0.0,
            pooled_ber - 1.96 * pooled_standard_error,
        )

        pooled_upper = min(
            1.0,
            pooled_ber + 1.96 * pooled_standard_error,
        )

        pooled_rows.append({
            "ebn0_db": float(ebn0_db),
            "number_of_coherence_estimates": len(selected),
            "pooled_ber": pooled_ber,
            "pooled_ber_ci_lower": pooled_lower,
            "pooled_ber_ci_upper": pooled_upper,
            "total_independent_blocks": int(sum(
                row["independent_blocks"] for row in selected
            )),
            "total_bits": int(sum(
                row["total_bits"] for row in selected
            )),
            "total_bit_errors": int(sum(
                row["bit_errors"] for row in selected
            )),
        })

    return pooled_rows


def calculate_common_uncoded_threshold(pooled_rows):
    ebn0 = np.array(
        [row["ebn0_db"] for row in pooled_rows]
    )

    ber = np.array(
        [row["pooled_ber"] for row in pooled_rows]
    )

    lower = np.array(
        [row["pooled_ber_ci_lower"] for row in pooled_rows]
    )

    upper = np.array(
        [row["pooled_ber_ci_upper"] for row in pooled_rows]
    )

    required_ebn0 = log_crossing(ebn0, ber, TARGET_BER)
    optimistic_ebn0 = log_crossing(ebn0, lower, TARGET_BER)
    pessimistic_ebn0 = log_crossing(ebn0, upper, TARGET_BER)

    if not np.isfinite(required_ebn0):
        raise RuntimeError(
            "The pooled uncoded BER curve does not bracket the target."
        )

    return {
        "target_ber": TARGET_BER,
        "common_uncoded_required_ebn0_db": required_ebn0,
        "common_uncoded_required_ebn0_optimistic_db": optimistic_ebn0,
        "common_uncoded_required_ebn0_pessimistic_db": pessimistic_ebn0,
        "pooling_method": "fixed_effect_inverse_variance_group_aware",
    }


# =============================================================================
# CODING GAIN
# =============================================================================

def calculate_coding_gain(
    task08_rows,
    common_uncoded_threshold,
):
    output_rows = []

    uncoded_threshold = common_uncoded_threshold[
        "common_uncoded_required_ebn0_db"
    ]

    for coded in task08_rows:
        coherence = coded["coherence_bits"]

        coding_gain = (
            uncoded_threshold
            - coded["coded_required_ebn0_db"]
        )

        output_rows.append({
            "code": coded["code"],
            "coherence_bits": coherence,
            "interleaver_depth": (
                coded["interleaver_depth"]
            ),
            "target_ber": TARGET_BER,
            "uncoded_required_ebn0_db": (
                uncoded_threshold
            ),
            "uncoded_required_ebn0_optimistic_db": (
                common_uncoded_threshold[
                    "common_uncoded_required_ebn0_optimistic_db"
                ]
            ),
            "uncoded_required_ebn0_pessimistic_db": (
                common_uncoded_threshold[
                    "common_uncoded_required_ebn0_pessimistic_db"
                ]
            ),
            "coded_required_ebn0_db": (
                coded["coded_required_ebn0_db"]
            ),
            "true_coding_gain_db": coding_gain,
            "buffer_kib": coded["buffer_kib"],
            "runtime_per_1000_information_bits_ms": (
                coded[
                    "runtime_per_1000_information_bits_ms"
                ]
            ),
        })

    output_rows.sort(
        key=lambda row: (
            row["coherence_bits"],
            row["code"],
            row["interleaver_depth"],
        )
    )

    return output_rows


# =============================================================================
# FIGURE 1: UNCODED BASELINE
# =============================================================================

def plot_uncoded_results(
    rows,
    pooled_rows,
    common_threshold,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
    )

    colours = {
        64: "royalblue",
        256: "firebrick",
    }

    markers = {
        64: "o",
        256: "s",
    }

    # BER panel
    ber_axis = axes[0]

    for coherence in COHERENCE_LENGTHS_BITS:
        selected = [
            row for row in rows
            if row["coherence_length_bits"]
            == coherence
        ]

        selected.sort(
            key=lambda row: row["ebn0_db"]
        )

        ebn0 = np.array(
            [row["ebn0_db"] for row in selected]
        )

        ber = np.array(
            [row["ber"] for row in selected]
        )

        lower = np.array(
            [
                row["ber_ci_lower"]
                for row in selected
            ]
        )

        upper = np.array(
            [
                row["ber_ci_upper"]
                for row in selected
            ]
        )

        ber_axis.semilogy(
            ebn0,
            ber,
            color=colours[coherence],
            marker=markers[coherence],
            linewidth=1.5,
            alpha=0.65,
            label=f"Lc={coherence} bits",
        )

        ber_axis.fill_between(
            ebn0,
            np.maximum(lower, 1e-7),
            np.maximum(upper, 1e-7),
            color=colours[coherence],
            alpha=0.08,
        )

    pooled_ebn0 = np.array(
        [row["ebn0_db"] for row in pooled_rows]
    )

    pooled_ber = np.array(
        [row["pooled_ber"] for row in pooled_rows]
    )

    pooled_lower = np.array(
        [row["pooled_ber_ci_lower"] for row in pooled_rows]
    )

    pooled_upper = np.array(
        [row["pooled_ber_ci_upper"] for row in pooled_rows]
    )

    ber_axis.semilogy(
        pooled_ebn0,
        pooled_ber,
        color="black",
        marker="D",
        linewidth=2.5,
        markersize=5,
        label="Pooled marginal BER",
    )

    ber_axis.fill_between(
        pooled_ebn0,
        np.maximum(pooled_lower, 1e-7),
        np.maximum(pooled_upper, 1e-7),
        color="black",
        alpha=0.12,
    )

    ber_axis.axhline(
        TARGET_BER,
        color="grey",
        linestyle="--",
        linewidth=1.4,
        label="Target BER=1e-3",
    )

    ber_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    ber_axis.set_ylabel("Uncoded BER")

    ber_axis.set_title(
        "(a) Group-aware uncoded BER"
    )

    ber_axis.set_ylim(1e-4, 2e-2)
    ber_axis.grid(
        True,
        which="both",
        alpha=0.35,
    )

    ber_axis.legend()

    # Coherence-block error probability
    block_axis = axes[1]

    for coherence in COHERENCE_LENGTHS_BITS:
        selected = [
            row for row in rows
            if row["coherence_length_bits"]
            == coherence
        ]

        selected.sort(
            key=lambda row: row["ebn0_db"]
        )

        ebn0 = np.array(
            [row["ebn0_db"] for row in selected]
        )

        probability = np.array(
            [
                row[
                    "coherence_block_error_probability"
                ]
                for row in selected
            ]
        )

        lower = np.array(
            [
                row["block_error_ci_lower"]
                for row in selected
            ]
        )

        upper = np.array(
            [
                row["block_error_ci_upper"]
                for row in selected
            ]
        )

        block_axis.semilogy(
            ebn0,
            probability,
            color=colours[coherence],
            marker=markers[coherence],
            linewidth=2,
            label=f"Lc={coherence} bits",
        )

        block_axis.fill_between(
            ebn0,
            np.maximum(lower, 1e-7),
            np.maximum(upper, 1e-7),
            color=colours[coherence],
            alpha=0.14,
        )

    block_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    block_axis.set_ylabel(
        "Probability of ≥1 error in a coherence block"
    )

    block_axis.set_title(
        "(b) Error clustering within coherence blocks"
    )

    block_axis.grid(
        True,
        which="both",
        alpha=0.35,
    )

    block_axis.legend()

    threshold_text = (
        f"Common pooled threshold: "
        f"{common_threshold['common_uncoded_required_ebn0_db']:.3f} dB\n"
        f"95% crossing range: ["
        f"{common_threshold['common_uncoded_required_ebn0_optimistic_db']:.3f}, "
        f"{common_threshold['common_uncoded_required_ebn0_pessimistic_db']:.3f}] dB"
    )

    ber_axis.text(
        0.03,
        0.05,
        "Required Eb/N0 at BER=1e-3\n" + threshold_text,
        transform=ber_axis.transAxes,
        fontsize=9,
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "grey",
        },
    )

    fig.suptitle(
        "Task 09: Uncoded Fixed-Coherence FSO Baseline\n"
        "Moderate Gamma-Gamma turbulence, severe normalised "
        "pointing error, perfect CSI",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.89)
    )

    fig.savefig(
        UNCODED_PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        UNCODED_PDF_PATH,
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# FIGURE 2: TRUE CODING GAIN
# =============================================================================

def plot_coding_gain(rows):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
        sharey=True,
    )

    code_colours = {
        "RS": "royalblue",
        "LDPC": "darkorange",
    }

    for axis, coherence in zip(
        axes,
        COHERENCE_LENGTHS_BITS,
    ):
        x_positions = np.arange(
            len(DEPTHS)
        )

        width = 0.34

        for code_index, code in enumerate(
            ["RS", "LDPC"]
        ):
            selected = [
                row for row in rows
                if row["coherence_bits"]
                == coherence
                and row["code"] == code
            ]

            selected.sort(
                key=lambda row:
                row["interleaver_depth"]
            )

            gains = [
                row["true_coding_gain_db"]
                for row in selected
            ]

            offsets = (
                x_positions
                + (
                    code_index - 0.5
                ) * width
            )

            bars = axis.bar(
                offsets,
                gains,
                width=width,
                color=code_colours[code],
                label=code,
                alpha=0.86,
            )

            for bar, gain in zip(
                bars,
                gains,
            ):
                axis.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{gain:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.axhline(
            0,
            color="black",
            linewidth=0.8,
        )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [str(depth) for depth in DEPTHS]
        )

        axis.set_xlabel(
            "Interleaver depth D"
        )

        axis.set_title(
            f"Lc={coherence} transmitted bits"
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        axis.legend()

    axes[0].set_ylabel(
        "True coding gain at BER=1e-3 (dB)"
    )

    fig.suptitle(
        "Task 09: True Coding Gain Relative to the "
        "Matched Uncoded FSO Baseline",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.91)
    )

    fig.savefig(
        CODING_GAIN_PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        CODING_GAIN_PDF_PATH,
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 88)

    print(
        "TASK 09: GROUP-AWARE UNCODED BASELINE "
        "AND TRUE CODING GAIN"
    )

    print("=" * 88)

    print(
        f"Target BER: {TARGET_BER:.1e}"
    )

    print(
        f"Eb/N0 values: {EBN0_DB_LIST.tolist()}"
    )

    print(
        f"Minimum independent blocks: "
        f"{MIN_INDEPENDENT_BLOCKS}"
    )

    print(
        f"Maximum independent blocks: "
        f"{MAX_INDEPENDENT_BLOCKS}"
    )

    if REUSE_EXISTING_UNCODED_CSV and UNCODED_CSV_PATH.exists():
        print(
            "\n[1] Reusing the completed formal uncoded CSV..."
        )

        uncoded_rows = read_uncoded_rows(
            UNCODED_CSV_PATH
        )

        expected_conditions = (
            len(COHERENCE_LENGTHS_BITS)
            * len(EBN0_DB_LIST)
        )

        if len(uncoded_rows) != expected_conditions:
            raise RuntimeError(
                f"Expected {expected_conditions} formal uncoded rows, "
                f"but found {len(uncoded_rows)}."
            )

        print(
            f"Loaded {len(uncoded_rows)} completed conditions from "
            f"{UNCODED_CSV_PATH.name}."
        )

    else:
        print(
            "\n[1] Running group-aware uncoded sweep..."
        )

        uncoded_rows = []

        total_conditions = (
            len(COHERENCE_LENGTHS_BITS)
            * len(EBN0_DB_LIST)
        )

        condition_number = 0

        for coherence_length in COHERENCE_LENGTHS_BITS:
            print("\n" + "-" * 88)
            print(f"Coherence length Lc={coherence_length} bits")
            print("-" * 88)

            for ebn0_db in EBN0_DB_LIST:
                condition_number += 1

                seed_sequence = np.random.SeedSequence([
                    MASTER_SEED,
                    coherence_length,
                    int(round(10 * ebn0_db)),
                ])

                rng = np.random.default_rng(seed_sequence)

                row = simulate_condition(
                    coherence_length,
                    float(ebn0_db),
                    rng,
                )

                uncoded_rows.append(row)
                save_rows(UNCODED_CSV_PATH, uncoded_rows)

                print(
                    f"[{condition_number:2d}/{total_conditions}] "
                    f"Lc={coherence_length:3d} | "
                    f"Eb/N0={ebn0_db:4.1f} dB | "
                    f"BER={row['ber']:.3e} | "
                    f"95% CI=[{row['ber_ci_lower']:.3e}, "
                    f"{row['ber_ci_upper']:.3e}] | "
                    f"blocks={row['independent_blocks']:5d} | "
                    f"errors={row['bit_errors']:6d}"
                )

    print(
        "\n[2] Pooling the common marginal uncoded BER and "
        "calculating one target crossing..."
    )

    pooled_rows = calculate_pooled_uncoded_curve(
        uncoded_rows
    )

    save_rows(POOLED_UNCODED_CSV_PATH, pooled_rows)

    common_threshold = calculate_common_uncoded_threshold(
        pooled_rows
    )

    print(
        f"Common pooled uncoded required Eb/N0="
        f"{common_threshold['common_uncoded_required_ebn0_db']:.3f} dB | "
        f"95% crossing range=["
        f"{common_threshold['common_uncoded_required_ebn0_optimistic_db']:.3f}, "
        f"{common_threshold['common_uncoded_required_ebn0_pessimistic_db']:.3f}] dB"
    )

    print(
        "\n[3] Reading Task 08 and calculating "
        "true coding gain..."
    )

    task08_rows = read_task08_summary()

    coding_gain_rows = calculate_coding_gain(
        task08_rows,
        common_threshold,
    )

    save_rows(
        CODING_GAIN_CSV_PATH,
        coding_gain_rows,
    )

    print("\n" + "=" * 104)

    print(
        "TRUE CODING GAIN AT BER=1e-3"
    )

    print("=" * 104)

    header = (
        f"{'Code':<6}"
        f"{'Lc(bits)':>10}"
        f"{'D':>6}"
        f"{'Uncoded(dB)':>15}"
        f"{'Coded(dB)':>13}"
        f"{'Gain(dB)':>12}"
        f"{'Buffer(KiB)':>15}"
    )

    print(header)
    print("-" * len(header))

    for row in coding_gain_rows:
        print(
            f"{row['code']:<6}"
            f"{row['coherence_bits']:>10d}"
            f"{row['interleaver_depth']:>6d}"
            f"{row['uncoded_required_ebn0_db']:>15.3f}"
            f"{row['coded_required_ebn0_db']:>13.3f}"
            f"{row['true_coding_gain_db']:>12.3f}"
            f"{row['buffer_kib']:>15.3f}"
        )

    print("\n[4] Creating figures...")

    plot_uncoded_results(
        uncoded_rows,
        pooled_rows,
        common_threshold,
    )

    plot_coding_gain(
        coding_gain_rows
    )

    print("\n" + "=" * 88)
    print("TASK 09 FINISHED")
    print("=" * 88)

    print(
        f"Uncoded CSV: "
        f"{UNCODED_CSV_PATH}"
    )

    print(
        f"Uncoded figure: "
        f"{UNCODED_PNG_PATH}"
    )

    print(
        f"Pooled uncoded CSV: "
        f"{POOLED_UNCODED_CSV_PATH}"
    )

    print(
        f"Coding-gain CSV: "
        f"{CODING_GAIN_CSV_PATH}"
    )

    print(
        f"Coding-gain figure: "
        f"{CODING_GAIN_PNG_PATH}"
    )

    print(
        "\nThe coding gain is now measured relative to one common, "
        "group-aware pooled uncoded marginal-BER baseline."
    )


if __name__ == "__main__":
    main()
