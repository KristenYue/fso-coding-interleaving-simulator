"""
TASK 07B
Group-aware refinement of LDPC-coded FSO results.

Purpose
-------
The previous simulation counted frames, but frames inside one interleaver
group share fading samples and are therefore correlated.

This version:
1. Counts independent interleaver groups.
2. Uses at least 40 independent groups near the target BER.
3. Uses batch LDPC decoding for speed.
4. Calculates confidence intervals from independent group statistics.
5. Supports resuming after interruption.
"""

from pathlib import Path
import csv
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from pyldpc import make_ldpc, decode, get_message
from pyldpc.utils import binaryproduct


# =============================================================================
# SETTINGS
# =============================================================================

MASTER_SEED = 20260620
SHOW_FIGURE = True

LDPC_N = 512
LDPC_DV = 2
LDPC_DC = 16
LDPC_SEED = 7
MAX_DECODER_ITERATIONS = 100

RS_REFERENCE_RATE = 223 / 255

COHERENCE_LENGTHS_BITS = [64, 256]
INTERLEAVER_DEPTHS = [1, 8, 32, 128]

# Refinement region around BER=1e-3 crossings
EBN0_DB_LIST = np.array(
    [21, 24, 27, 30, 33, 36],
    dtype=float,
)

# Statistical stopping requirements
MIN_INDEPENDENT_GROUPS = 40
MIN_INFORMATION_BIT_ERRORS = 200
MIN_FRAME_ERRORS = 80

# Divisible by 1, 8, 32 and 128.
# D=128 therefore receives at most 50 independent groups.
MAX_FRAMES = 6400

ALPHA_MODERATE = 4.868481548059242
BETA_MODERATE = 3.164867988481172
XI_SEVERE = 1.522040256461071


# =============================================================================
# OUTPUT
# =============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_STEM = "07b_ldpc_fso_group_aware_refinement"

CSV_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.csv"
PNG_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.png"
PDF_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.pdf"


# =============================================================================
# CONFIDENCE INTERVAL BASED ON INDEPENDENT GROUPS
# =============================================================================

def group_confidence_interval(group_values):
    """
    Calculate an approximate 95% confidence interval using independent
    interleaver-group measurements.

    Frames inside a group may be correlated, but different groups are
    independently generated.
    """

    values = np.asarray(group_values, dtype=float)
    number_groups = values.size

    if number_groups == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

    if number_groups == 1:
        return mean_value, 0.0, 1.0

    standard_error = float(
        np.std(values, ddof=1) / np.sqrt(number_groups)
    )

    half_width = 1.96 * standard_error

    lower = max(0.0, mean_value - half_width)
    upper = min(1.0, mean_value + half_width)

    return mean_value, lower, upper


# =============================================================================
# CHANNEL
# =============================================================================

def sample_gamma_gamma(alpha, beta, number_samples, rng):
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


def sample_normalised_pointing_gain(xi, number_samples, rng):
    uniform_samples = rng.random(number_samples)

    return uniform_samples ** (1.0 / xi ** 2)


def generate_fixed_coherence_channel(
    number_bits,
    coherence_length_bits,
    rng,
):
    number_blocks = int(
        np.ceil(number_bits / coherence_length_bits)
    )

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

    block_gain = turbulence * pointing

    return np.repeat(
        block_gain,
        coherence_length_bits,
    )[:number_bits]


# =============================================================================
# LDPC
# =============================================================================

def construct_ldpc():
    parity_check_matrix, generator_matrix = make_ldpc(
        LDPC_N,
        LDPC_DV,
        LDPC_DC,
        systematic=True,
        sparse=True,
        seed=LDPC_SEED,
    )

    information_length = generator_matrix.shape[1]
    code_rate = information_length / LDPC_N

    return (
        parity_check_matrix,
        generator_matrix,
        information_length,
        code_rate,
    )


def encode_information_bits(generator_matrix, information_bits):
    codeword = binaryproduct(
        generator_matrix,
        information_bits.astype(np.int64),
    )

    return (
        np.asarray(codeword)
        .astype(np.int64)
        .reshape(-1)
        % 2
    )


def decode_llr_batch(
    parity_check_matrix,
    generator_matrix,
    llr_matrix,
):
    """
    Batch-decode one complete interleaver group.

    Input shape:
        number_frames x LDPC_N

    Output:
        decoded information matrix
        parity-check pass vector
    """

    number_frames = llr_matrix.shape[0]

    clipped_llr = np.clip(
        llr_matrix,
        -80.0,
        80.0,
    )

    # pyldpc expects n x number_messages.
    equivalent_observation = (
        -0.5 * clipped_llr
    ).T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        decoded = decode(
            parity_check_matrix,
            equivalent_observation,
            snr=0.0,
            maxiter=MAX_DECODER_ITERATIONS,
        )

    decoded = np.asarray(decoded)

    if decoded.ndim == 1:
        decoded = decoded.reshape(-1, 1)

    decoded = (
        np.rint(decoded)
        .astype(np.int64)
        % 2
    )

    syndrome = np.asarray(
        binaryproduct(
            parity_check_matrix,
            decoded,
        )
    ).astype(np.int64) % 2

    if syndrome.ndim == 1:
        syndrome = syndrome.reshape(-1, 1)

    parity_passed = np.all(
        syndrome == 0,
        axis=0,
    )

    decoded_information_matrix = np.empty(
        (
            number_frames,
            generator_matrix.shape[1],
        ),
        dtype=np.int64,
    )

    for frame_index in range(number_frames):
        codeword = decoded[:, frame_index].astype(
            np.int64
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            information = get_message(
                generator_matrix,
                codeword,
            )

        decoded_information_matrix[frame_index] = (
            np.rint(np.asarray(information))
            .astype(np.int64)
            .reshape(-1)
            % 2
        )

    return decoded_information_matrix, parity_passed


# =============================================================================
# SELF-TEST
# =============================================================================

def run_self_tests(
    parity_check_matrix,
    generator_matrix,
    information_length,
):
    rng = np.random.default_rng(123456)

    number_test_frames = 8

    information_matrix = rng.integers(
        0,
        2,
        size=(number_test_frames, information_length),
        dtype=np.int64,
    )

    codeword_matrix = np.empty(
        (number_test_frames, LDPC_N),
        dtype=np.int64,
    )

    for frame_index in range(number_test_frames):
        codeword_matrix[frame_index] = (
            encode_information_bits(
                generator_matrix,
                information_matrix[frame_index],
            )
        )

    syndrome = np.asarray(
        binaryproduct(
            parity_check_matrix,
            codeword_matrix.T,
        )
    ).astype(np.int64) % 2

    if not np.all(syndrome == 0):
        raise RuntimeError(
            "LDPC encoding parity-check test failed."
        )

    noiseless_llr = np.where(
        codeword_matrix == 1,
        40.0,
        -40.0,
    )

    decoded_information, parity_passed = (
        decode_llr_batch(
            parity_check_matrix,
            generator_matrix,
            noiseless_llr,
        )
    )

    if not np.all(parity_passed):
        raise RuntimeError(
            "Noiseless batch decoding parity test failed."
        )

    if not np.array_equal(
        decoded_information,
        information_matrix,
    ):
        raise RuntimeError(
            "Noiseless batch information recovery failed."
        )


# =============================================================================
# SIMULATE ONE CONDITION
# =============================================================================

def simulate_condition(
    parity_check_matrix,
    generator_matrix,
    information_length,
    code_rate,
    ebn0_db,
    coherence_length_bits,
    interleaver_depth,
    rng,
):
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    noise_variance = 1.0 / (
        4.0 * code_rate * ebn0_linear
    )

    noise_standard_deviation = np.sqrt(
        noise_variance
    )

    number_groups = 0
    total_frames = 0

    total_information_bits = 0
    total_coded_bits = 0

    information_bit_errors = 0
    raw_bit_errors = 0
    frame_errors = 0

    decoder_failures = 0
    undetected_codeword_errors = 0

    group_information_ber = []
    group_raw_ber = []
    group_fer = []

    start_time = time.perf_counter()

    while total_frames < MAX_FRAMES:
        group_size = interleaver_depth

        information_matrix = rng.integers(
            0,
            2,
            size=(group_size, information_length),
            dtype=np.int64,
        )

        codeword_matrix = np.empty(
            (group_size, LDPC_N),
            dtype=np.int64,
        )

        for frame_index in range(group_size):
            codeword_matrix[frame_index] = (
                encode_information_bits(
                    generator_matrix,
                    information_matrix[frame_index],
                )
            )

        # Column-wise bit interleaving.
        transmitted_bits = codeword_matrix.T.reshape(-1)

        channel_gain = generate_fixed_coherence_channel(
            transmitted_bits.size,
            coherence_length_bits,
            rng,
        )

        noise = rng.normal(
            0.0,
            noise_standard_deviation,
            transmitted_bits.size,
        )

        received = (
            channel_gain * transmitted_bits
            + noise
        )

        hard_decisions = (
            received >= 0.5 * channel_gain
        ).astype(np.int64)

        group_raw_errors = int(
            np.count_nonzero(
                hard_decisions != transmitted_bits
            )
        )

        llr_transmitted_order = (
            2.0 * channel_gain * received
            - channel_gain ** 2
        ) / (2.0 * noise_variance)

        llr_matrix = llr_transmitted_order.reshape(
            LDPC_N,
            group_size,
        ).T

        (
            decoded_information_matrix,
            parity_passed,
        ) = decode_llr_batch(
            parity_check_matrix,
            generator_matrix,
            llr_matrix,
        )

        information_error_matrix = (
            decoded_information_matrix
            != information_matrix
        )

        frame_error_vector = np.any(
            information_error_matrix,
            axis=1,
        )

        group_information_errors = int(
            np.count_nonzero(
                information_error_matrix
            )
        )

        group_frame_errors = int(
            np.count_nonzero(frame_error_vector)
        )

        group_decoder_failures = int(
            np.count_nonzero(~parity_passed)
        )

        group_undetected_errors = int(
            np.count_nonzero(
                frame_error_vector & parity_passed
            )
        )

        group_information_bits = (
            group_size * information_length
        )

        group_coded_bits = group_size * LDPC_N

        group_information_ber.append(
            group_information_errors
            / group_information_bits
        )

        group_raw_ber.append(
            group_raw_errors / group_coded_bits
        )

        group_fer.append(
            group_frame_errors / group_size
        )

        number_groups += 1
        total_frames += group_size

        total_information_bits += (
            group_information_bits
        )

        total_coded_bits += group_coded_bits

        information_bit_errors += (
            group_information_errors
        )

        raw_bit_errors += group_raw_errors
        frame_errors += group_frame_errors

        decoder_failures += (
            group_decoder_failures
        )

        undetected_codeword_errors += (
            group_undetected_errors
        )

        enough_groups = (
            number_groups
            >= MIN_INDEPENDENT_GROUPS
        )

        enough_bit_errors = (
            information_bit_errors
            >= MIN_INFORMATION_BIT_ERRORS
        )

        enough_frame_errors = (
            frame_errors >= MIN_FRAME_ERRORS
        )

        if (
            enough_groups
            and enough_bit_errors
            and enough_frame_errors
        ):
            break

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    (
        information_ber,
        information_ber_ci_lower,
        information_ber_ci_upper,
    ) = group_confidence_interval(
        group_information_ber
    )

    (
        raw_ber,
        raw_ber_ci_lower,
        raw_ber_ci_upper,
    ) = group_confidence_interval(
        group_raw_ber
    )

    (
        fer,
        fer_ci_lower,
        fer_ci_upper,
    ) = group_confidence_interval(
        group_fer
    )

    return {
        "ebn0_db": float(ebn0_db),
        "coherence_length_bits": (
            coherence_length_bits
        ),
        "interleaver_depth": interleaver_depth,
        "ldpc_n": LDPC_N,
        "ldpc_k": information_length,
        "code_rate": code_rate,
        "independent_groups": number_groups,
        "frames": total_frames,
        "information_bits": total_information_bits,
        "coded_bits": total_coded_bits,
        "raw_bit_errors": raw_bit_errors,
        "raw_ber": raw_ber,
        "raw_ber_ci_lower": raw_ber_ci_lower,
        "raw_ber_ci_upper": raw_ber_ci_upper,
        "information_bit_errors": (
            information_bit_errors
        ),
        "information_ber": information_ber,
        "information_ber_ci_lower": (
            information_ber_ci_lower
        ),
        "information_ber_ci_upper": (
            information_ber_ci_upper
        ),
        "frame_errors": frame_errors,
        "fer": fer,
        "fer_ci_lower": fer_ci_lower,
        "fer_ci_upper": fer_ci_upper,
        "decoder_failures": decoder_failures,
        "undetected_codeword_errors": (
            undetected_codeword_errors
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_per_frame_ms": (
            1000.0 * runtime_seconds
            / total_frames
        ),
    }


# =============================================================================
# SAVE AND RESUME
# =============================================================================

def save_csv(rows):
    if not rows:
        return

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["coherence_length_bits"]),
            int(row["interleaver_depth"]),
            float(row["ebn0_db"]),
        ),
    )

    with CSV_PATH.open(
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


def load_existing_results():
    if not CSV_PATH.exists():
        return []

    rows = []

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            converted = {}

            integer_fields = {
                "coherence_length_bits",
                "interleaver_depth",
                "ldpc_n",
                "ldpc_k",
                "independent_groups",
                "frames",
                "information_bits",
                "coded_bits",
                "raw_bit_errors",
                "information_bit_errors",
                "frame_errors",
                "decoder_failures",
                "undetected_codeword_errors",
            }

            for key, value in row.items():
                if key in integer_fields:
                    converted[key] = int(float(value))
                else:
                    converted[key] = float(value)

            rows.append(converted)

    return rows


def condition_key(row):
    return (
        int(row["coherence_length_bits"]),
        int(row["interleaver_depth"]),
        float(row["ebn0_db"]),
    )


# =============================================================================
# PLOT
# =============================================================================

def plot_results(rows):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        sharex="col",
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

    for column, coherence_length in enumerate(
        COHERENCE_LENGTHS_BITS
    ):
        ber_axis = axes[0, column]
        fer_axis = axes[1, column]

        for depth in INTERLEAVER_DEPTHS:
            selected = [
                row for row in rows
                if int(row["coherence_length_bits"])
                == coherence_length
                and int(row["interleaver_depth"])
                == depth
            ]

            selected.sort(
                key=lambda row: float(row["ebn0_db"])
            )

            ebn0 = np.array(
                [row["ebn0_db"] for row in selected]
            )

            ber = np.array(
                [
                    row["information_ber"]
                    for row in selected
                ]
            )

            ber_lower = np.array(
                [
                    row["information_ber_ci_lower"]
                    for row in selected
                ]
            )

            ber_upper = np.array(
                [
                    row["information_ber_ci_upper"]
                    for row in selected
                ]
            )

            fer = np.array(
                [row["fer"] for row in selected]
            )

            fer_lower = np.array(
                [
                    row["fer_ci_lower"]
                    for row in selected
                ]
            )

            fer_upper = np.array(
                [
                    row["fer_ci_upper"]
                    for row in selected
                ]
            )

            colour = colours[depth]

            ber_axis.semilogy(
                ebn0,
                np.maximum(ber, 1e-7),
                color=colour,
                marker=markers[depth],
                linewidth=2,
                markersize=5,
                label=f"D={depth}",
            )

            ber_axis.fill_between(
                ebn0,
                np.maximum(ber_lower, 1e-7),
                np.maximum(ber_upper, 1e-7),
                color=colour,
                alpha=0.10,
            )

            fer_axis.semilogy(
                ebn0,
                np.maximum(fer, 1e-7),
                color=colour,
                marker=markers[depth],
                linewidth=2,
                markersize=5,
                label=f"D={depth}",
            )

            fer_axis.fill_between(
                ebn0,
                np.maximum(fer_lower, 1e-7),
                np.maximum(fer_upper, 1e-7),
                color=colour,
                alpha=0.10,
            )

        ber_axis.axhline(
            1e-3,
            color="grey",
            linestyle="--",
            linewidth=1.4,
            label="Target BER=1e-3",
        )

        ber_axis.set_title(
            f"Group-aware BER, "
            f"Lc={coherence_length} bits"
        )

        fer_axis.set_title(
            f"Group-aware FER, "
            f"Lc={coherence_length} bits"
        )

        ber_axis.set_ylabel(
            "Decoded information BER"
        )

        fer_axis.set_ylabel(
            "Frame error rate"
        )

        fer_axis.set_xlabel(
            "Information-bit Eb/N0 (dB)"
        )

        for axis in (ber_axis, fer_axis):
            axis.grid(
                True,
                which="both",
                alpha=0.35,
            )

            axis.set_ylim(1e-6, 1.0)
            axis.legend(fontsize=9)

    fig.suptitle(
        "Task 07B: Group-Aware LDPC-Coded FSO Refinement\n"
        "Shaded regions: 95% confidence intervals across "
        "independent interleaver groups",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.93)
    )

    fig.savefig(
        PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF_PATH,
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
        "TASK 07B: GROUP-AWARE LDPC FSO REFINEMENT"
    )
    print("=" * 88)

    (
        parity_check_matrix,
        generator_matrix,
        information_length,
        code_rate,
    ) = construct_ldpc()

    print(
        f"LDPC: n={LDPC_N}, "
        f"k={information_length}, "
        f"R={code_rate:.6f}"
    )

    print(
        f"RS reference rate: "
        f"{RS_REFERENCE_RATE:.6f}"
    )

    print(
        f"Eb/N0: {EBN0_DB_LIST.tolist()}"
    )

    print(
        f"Minimum independent groups: "
        f"{MIN_INDEPENDENT_GROUPS}"
    )

    print(
        f"Maximum frames per condition: "
        f"{MAX_FRAMES}"
    )

    print("\n[1] Running batch LDPC self-tests...")

    run_self_tests(
        parity_check_matrix,
        generator_matrix,
        information_length,
    )

    print("Batch LDPC self-tests passed.")

    existing_rows = load_existing_results()
    completed_keys = {
        condition_key(row)
        for row in existing_rows
    }

    if existing_rows:
        print(
            f"\nResuming from {len(existing_rows)} "
            f"completed conditions."
        )

    rows = list(existing_rows)

    all_conditions = [
        (lc, depth, float(ebn0))
        for lc in COHERENCE_LENGTHS_BITS
        for depth in INTERLEAVER_DEPTHS
        for ebn0 in EBN0_DB_LIST
    ]

    total_conditions = len(all_conditions)

    print(
        f"\n[2] Running {total_conditions} "
        f"group-aware conditions..."
    )

    for condition_number, (
        coherence_length,
        depth,
        ebn0_db,
    ) in enumerate(all_conditions, start=1):

        key = (
            coherence_length,
            depth,
            ebn0_db,
        )

        if key in completed_keys:
            print(
                f"[{condition_number:2d}/"
                f"{total_conditions}] "
                f"Lc={coherence_length:3d}, "
                f"D={depth:3d}, "
                f"Eb/N0={ebn0_db:4.1f} dB "
                f"| already completed"
            )

            continue

        seed_sequence = np.random.SeedSequence(
            [
                MASTER_SEED,
                coherence_length,
                depth,
                int(round(10 * ebn0_db)),
            ]
        )

        rng = np.random.default_rng(
            seed_sequence
        )

        row = simulate_condition(
            parity_check_matrix,
            generator_matrix,
            information_length,
            code_rate,
            ebn0_db,
            coherence_length,
            depth,
            rng,
        )

        rows.append(row)
        completed_keys.add(key)

        save_csv(rows)

        print(
            f"[{condition_number:2d}/"
            f"{total_conditions}] "
            f"Lc={coherence_length:3d} | "
            f"D={depth:3d} | "
            f"Eb/N0={ebn0_db:4.1f} dB | "
            f"BER={row['information_ber']:.3e} | "
            f"95% CI=["
            f"{row['information_ber_ci_lower']:.3e}, "
            f"{row['information_ber_ci_upper']:.3e}] | "
            f"FER={row['fer']:.3e} | "
            f"groups={row['independent_groups']:3d} | "
            f"frames={row['frames']:5d} | "
            f"time={row['runtime_seconds']:.1f}s"
        )

    print("\n[3] Creating group-aware figures...")

    save_csv(rows)
    plot_results(rows)

    print("\n" + "=" * 88)
    print("TASK 07B FINISHED")
    print("=" * 88)

    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")

    print(
        "\nThis CSV should be used for the final "
        "RS-LDPC comparison."
    )


if __name__ == "__main__":
    main()