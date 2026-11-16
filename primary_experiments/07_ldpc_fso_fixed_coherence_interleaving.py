"""
TASK 07
Rate-matched LDPC over a fixed-coherence FSO channel.

Channel:
- IM/DD OOK
- Moderate Gamma-Gamma turbulence
- Severe normalised pointing error
- Perfect receiver CSI
- Fixed coherence length independent of interleaver depth

Coherence correspondence:
- 64 transmitted bits  = 8 RS symbols
- 256 transmitted bits = 32 RS symbols
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
# FORMAL SIMULATION SETTINGS
# =============================================================================

MASTER_SEED = 20260619
SHOW_FIGURE = True

# Rate-matched LDPC
LDPC_N = 512
LDPC_DV = 2
LDPC_DC = 16
LDPC_SEED = 7
MAX_DECODER_ITERATIONS = 100

RS_REFERENCE_RATE = 223 / 255

# Fixed coherence lengths in transmitted bits
COHERENCE_LENGTHS_BITS = [64, 256]

INTERLEAVER_DEPTHS = [1, 8, 32, 128]

# Formal Eb/N0 sweep
EBN0_DB_LIST = np.arange(6, 37, 3, dtype=float)

# Formal Monte Carlo stopping requirements
MIN_INFORMATION_BIT_ERRORS = 200
MIN_FRAME_ERRORS = 80
MAX_FRAMES = 2048

# Validated moderate Gamma-Gamma parameters
ALPHA_MODERATE = 4.868481548059242
BETA_MODERATE = 3.164867988481172

# Validated severe normalised pointing-error parameter
XI_SEVERE = 1.522040256461071


# =============================================================================
# OUTPUT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_STEM = "07_ldpc_fso_fixed_coherence_interleaving"

CSV_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.csv"
PNG_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.png"
PDF_PATH = RESULTS_DIR / f"{OUTPUT_STEM}.pdf"


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================

def wilson_interval(number_errors, number_trials, z=1.95996398454):
    """Return the 95% Wilson confidence interval."""

    if number_trials <= 0:
        return np.nan, np.nan

    p_hat = number_errors / number_trials
    z_squared = z ** 2

    denominator = 1.0 + z_squared / number_trials

    centre = (
        p_hat + z_squared / (2.0 * number_trials)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            p_hat * (1.0 - p_hat) / number_trials
            + z_squared / (4.0 * number_trials ** 2)
        )
        / denominator
    )

    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)

    return lower, upper


# =============================================================================
# FSO CHANNEL
# =============================================================================

def sample_gamma_gamma(alpha, beta, number_samples, rng):
    """
    Generate unit-mean Gamma-Gamma irradiance.

    I = X * Y
    X ~ Gamma(alpha, 1/alpha)
    Y ~ Gamma(beta, 1/beta)
    """

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
    """
    Generate the normalised pointing gain hp/A0.

    If U is uniform on [0,1]:

        hp/A0 = U^(1/xi^2)
    """

    uniform_samples = rng.random(number_samples)

    return uniform_samples ** (1.0 / xi ** 2)


def generate_fixed_coherence_channel(
    number_bits,
    coherence_length_bits,
    rng,
):
    """
    Generate a block-fading FSO channel.

    Channel gain remains fixed for coherence_length_bits transmitted
    bits. Coherence length is independent of interleaver depth.
    """

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

    channel_gain = np.repeat(
        block_gain,
        coherence_length_bits,
    )

    return channel_gain[:number_bits]


# =============================================================================
# LDPC
# =============================================================================

def construct_ldpc():
    """Construct the deterministic rate-matched LDPC code."""

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
    """Encode one LDPC information block without adding noise."""

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


def decode_from_ook_llr(
    parity_check_matrix,
    generator_matrix,
    llr_bit_one,
):
    """
    Decode from OOK soft information.

    Input LLR definition:

        LLR = log[P(y|bit=1) / P(y|bit=0)]

    pyldpc uses the opposite BPSK sign convention. Therefore:

        y_equivalent = -LLR / 2
    """

    clipped_llr = np.clip(llr_bit_one, -80.0, 80.0)
    equivalent_observation = -0.5 * clipped_llr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        decoded_codeword = decode(
            parity_check_matrix,
            equivalent_observation,
            snr=0.0,
            maxiter=MAX_DECODER_ITERATIONS,
        )

    # Keep this signed. uint8 causes overflow inside pyldpc.
    decoded_codeword = (
        np.rint(np.asarray(decoded_codeword))
        .astype(np.int64)
        .reshape(-1)
        % 2
    )

    syndrome = (
        np.asarray(
            binaryproduct(
                parity_check_matrix,
                decoded_codeword,
            )
        )
        .astype(np.int64)
        .reshape(-1)
        % 2
    )

    parity_check_passed = bool(np.all(syndrome == 0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        decoded_information = get_message(
            generator_matrix,
            decoded_codeword,
        )

    decoded_information = (
        np.rint(np.asarray(decoded_information))
        .astype(np.int64)
        .reshape(-1)
        % 2
    )

    return (
        decoded_information,
        decoded_codeword,
        parity_check_passed,
    )


# =============================================================================
# LDPC SELF-TESTS
# =============================================================================

def run_ldpc_self_tests(
    parity_check_matrix,
    generator_matrix,
    information_length,
):
    """Run parity-check and noiseless decoding tests."""

    rng = np.random.default_rng(123456)

    for test_index in range(5):
        information = rng.integers(
            0,
            2,
            information_length,
            dtype=np.int64,
        )

        codeword = encode_information_bits(
            generator_matrix,
            information,
        )

        syndrome = (
            np.asarray(
                binaryproduct(
                    parity_check_matrix,
                    codeword,
                )
            )
            .astype(np.int64)
            .reshape(-1)
            % 2
        )

        if not np.all(syndrome == 0):
            raise RuntimeError(
                f"Parity-check self-test failed at test {test_index}."
            )

        noiseless_llr = np.where(
            codeword == 1,
            40.0,
            -40.0,
        )

        (
            decoded_information,
            _,
            parity_passed,
        ) = decode_from_ook_llr(
            parity_check_matrix,
            generator_matrix,
            noiseless_llr,
        )

        if not parity_passed:
            raise RuntimeError(
                f"Noiseless parity test failed at test {test_index}."
            )

        if not np.array_equal(
            decoded_information,
            information,
        ):
            raise RuntimeError(
                f"Noiseless information recovery failed "
                f"at test {test_index}."
            )


# =============================================================================
# ONE MONTE CARLO CONDITION
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
    """Simulate one Eb/N0, coherence length and depth combination."""

    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    # Rate-aware information-bit Eb/N0:
    #
    # sigma² = 1 / (4 R Eb/N0)
    noise_variance = 1.0 / (
        4.0 * code_rate * ebn0_linear
    )

    noise_standard_deviation = np.sqrt(noise_variance)

    total_frames = 0
    total_information_bits = 0
    total_coded_bits = 0

    raw_bit_errors = 0
    information_bit_errors = 0
    frame_errors = 0

    decoder_failures = 0
    undetected_codeword_errors = 0

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

        # Column-wise bit interleaving:
        # cw0_bit0, cw1_bit0, ..., cwD_bit0, cw0_bit1, ...
        transmitted_bits = codeword_matrix.T.reshape(-1)

        channel_gain = generate_fixed_coherence_channel(
            number_bits=transmitted_bits.size,
            coherence_length_bits=coherence_length_bits,
            rng=rng,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=transmitted_bits.size,
        )

        received = (
            channel_gain * transmitted_bits
            + noise
        )

        # Perfect-CSI OOK hard decision
        hard_decisions = (
            received >= 0.5 * channel_gain
        ).astype(np.int64)

        raw_bit_errors += int(
            np.count_nonzero(
                hard_decisions != transmitted_bits
            )
        )

        # Exact OOK LLR with known instantaneous channel gain:
        #
        # log[P(y|1,h)/P(y|0,h)]
        # = (2hy - h²)/(2 sigma²)
        transmitted_llr = (
            2.0 * channel_gain * received
            - channel_gain ** 2
        ) / (2.0 * noise_variance)

        # Deinterleave the soft information
        llr_matrix = transmitted_llr.reshape(
            LDPC_N,
            group_size,
        ).T

        for frame_index in range(group_size):
            transmitted_information = (
                information_matrix[frame_index]
            )

            (
                decoded_information,
                _,
                parity_passed,
            ) = decode_from_ook_llr(
                parity_check_matrix,
                generator_matrix,
                llr_matrix[frame_index],
            )

            wrong_information_bits = int(
                np.count_nonzero(
                    decoded_information
                    != transmitted_information
                )
            )

            information_bit_errors += wrong_information_bits

            if wrong_information_bits > 0:
                frame_errors += 1

                if parity_passed:
                    undetected_codeword_errors += 1

            if not parity_passed:
                decoder_failures += 1

        total_frames += group_size
        total_information_bits += (
            group_size * information_length
        )
        total_coded_bits += group_size * LDPC_N

        enough_bit_errors = (
            information_bit_errors
            >= MIN_INFORMATION_BIT_ERRORS
        )

        enough_frame_errors = (
            frame_errors >= MIN_FRAME_ERRORS
        )

        if enough_bit_errors and enough_frame_errors:
            break

    runtime_seconds = time.perf_counter() - start_time

    raw_ber = raw_bit_errors / total_coded_bits

    information_ber = (
        information_bit_errors / total_information_bits
    )

    fer = frame_errors / total_frames

    raw_ci_lower, raw_ci_upper = wilson_interval(
        raw_bit_errors,
        total_coded_bits,
    )

    ber_ci_lower, ber_ci_upper = wilson_interval(
        information_bit_errors,
        total_information_bits,
    )

    fer_ci_lower, fer_ci_upper = wilson_interval(
        frame_errors,
        total_frames,
    )

    # Zero-error points are displayed using their 95% upper bounds.
    information_ber_plot = (
        information_ber
        if information_bit_errors > 0
        else ber_ci_upper
    )

    fer_plot = (
        fer
        if frame_errors > 0
        else fer_ci_upper
    )

    return {
        "ebn0_db": float(ebn0_db),
        "coherence_length_bits": coherence_length_bits,
        "interleaver_depth": interleaver_depth,
        "ldpc_n": LDPC_N,
        "ldpc_k": information_length,
        "code_rate": code_rate,
        "frames": total_frames,
        "information_bits": total_information_bits,
        "coded_bits": total_coded_bits,
        "raw_bit_errors": raw_bit_errors,
        "raw_ber": raw_ber,
        "raw_ber_ci_lower": raw_ci_lower,
        "raw_ber_ci_upper": raw_ci_upper,
        "information_bit_errors": information_bit_errors,
        "information_ber": information_ber,
        "information_ber_plot": information_ber_plot,
        "information_ber_ci_lower": ber_ci_lower,
        "information_ber_ci_upper": ber_ci_upper,
        "frame_errors": frame_errors,
        "fer": fer,
        "fer_plot": fer_plot,
        "fer_ci_lower": fer_ci_lower,
        "fer_ci_upper": fer_ci_upper,
        "decoder_failures": decoder_failures,
        "undetected_codeword_errors": (
            undetected_codeword_errors
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_per_frame_ms": (
            1000.0 * runtime_seconds / total_frames
        ),
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_csv(rows):
    """Save results progressively."""

    if not rows:
        return

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


# =============================================================================
# FIGURE
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
                if row["coherence_length_bits"]
                == coherence_length
                and row["interleaver_depth"] == depth
            ]

            selected.sort(
                key=lambda row: row["ebn0_db"]
            )

            ebn0 = np.array(
                [row["ebn0_db"] for row in selected]
            )

            ber = np.array(
                [
                    row["information_ber_plot"]
                    for row in selected
                ]
            )

            fer = np.array(
                [row["fer_plot"] for row in selected]
            )

            ber_axis.semilogy(
                ebn0,
                ber,
                color=colours[depth],
                marker=markers[depth],
                linewidth=2,
                markersize=5,
                label=f"D={depth}",
            )

            fer_axis.semilogy(
                ebn0,
                fer,
                color=colours[depth],
                marker=markers[depth],
                linewidth=2,
                markersize=5,
                label=f"D={depth}",
            )

        ber_axis.axhline(
            1e-3,
            color="grey",
            linestyle="--",
            linewidth=1.4,
            label="Target BER=1e-3",
        )

        ber_axis.set_title(
            f"Information BER, coherence length="
            f"{coherence_length} bits"
        )

        fer_axis.set_title(
            f"Frame error rate, coherence length="
            f"{coherence_length} bits"
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
        "Task 07: Rate-Matched LDPC-Coded FSO Performance\n"
        "Moderate Gamma-Gamma turbulence, severe normalised "
        "pointing error, perfect CSI",
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
    print("=" * 84)
    print(
        "TASK 07: FORMAL RATE-MATCHED LDPC "
        "FIXED-COHERENCE FSO SIMULATION"
    )
    print("=" * 84)

    (
        parity_check_matrix,
        generator_matrix,
        information_length,
        code_rate,
    ) = construct_ldpc()

    print(
        f"Generated LDPC: n={LDPC_N}, "
        f"k={information_length}"
    )

    print(f"LDPC rate: {code_rate:.6f}")
    print(
        f"RS reference rate: "
        f"{RS_REFERENCE_RATE:.6f}"
    )

    print(
        f"Rate difference: "
        f"{code_rate - RS_REFERENCE_RATE:+.6f}"
    )

    print(
        f"Eb/N0 values: {EBN0_DB_LIST.tolist()}"
    )

    print(
        f"Coherence lengths: "
        f"{COHERENCE_LENGTHS_BITS} bits"
    )

    print(
        f"Interleaver depths: "
        f"{INTERLEAVER_DEPTHS}"
    )

    print(
        f"Stopping rules: bit errors >= "
        f"{MIN_INFORMATION_BIT_ERRORS}, "
        f"frame errors >= {MIN_FRAME_ERRORS}, "
        f"maximum frames = {MAX_FRAMES}"
    )

    print("\n[1] Running LDPC self-tests...")

    run_ldpc_self_tests(
        parity_check_matrix,
        generator_matrix,
        information_length,
    )

    print(
        "LDPC parity-check and noiseless "
        "soft-decoding tests passed."
    )

    print("\n[2] Running formal FSO Monte Carlo sweep...")

    all_rows = []

    total_conditions = (
        len(COHERENCE_LENGTHS_BITS)
        * len(INTERLEAVER_DEPTHS)
        * len(EBN0_DB_LIST)
    )

    condition_number = 0

    for coherence_length in COHERENCE_LENGTHS_BITS:
        print("\n" + "-" * 84)

        print(
            f"Fixed coherence length: "
            f"Lc={coherence_length} transmitted bits"
        )

        print("-" * 84)

        for depth in INTERLEAVER_DEPTHS:
            print(
                f"\nInterleaver depth D={depth}"
            )

            for ebn0_db in EBN0_DB_LIST:
                condition_number += 1

                seed_sequence = np.random.SeedSequence(
                    [
                        MASTER_SEED,
                        int(coherence_length),
                        int(depth),
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
                    float(ebn0_db),
                    coherence_length,
                    depth,
                    rng,
                )

                all_rows.append(row)

                if row["information_bit_errors"] > 0:
                    ber_text = (
                        f"{row['information_ber']:.3e}"
                    )
                else:
                    ber_text = (
                        "upper "
                        f"{row['information_ber_ci_upper']:.3e}"
                    )

                if row["frame_errors"] > 0:
                    fer_text = f"{row['fer']:.3e}"
                else:
                    fer_text = (
                        "upper "
                        f"{row['fer_ci_upper']:.3e}"
                    )

                print(
                    f"[{condition_number:3d}/"
                    f"{total_conditions}] "
                    f"Eb/N0={ebn0_db:5.1f} dB | "
                    f"raw={row['raw_ber']:.3e} | "
                    f"BER={ber_text} | "
                    f"FER={fer_text} | "
                    f"frames={row['frames']:4d} | "
                    f"failures="
                    f"{row['decoder_failures']:4d} | "
                    f"undetected="
                    f"{row['undetected_codeword_errors']:4d} | "
                    f"time={row['runtime_seconds']:.1f}s"
                )

                # Protect data during a long simulation.
                save_csv(all_rows)

    print("\n[3] Creating formal figures...")

    save_csv(all_rows)
    plot_results(all_rows)

    print("\n" + "=" * 84)
    print("TASK 07 FORMAL SIMULATION FINISHED")
    print("=" * 84)

    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")

    print(
        "\nSend me the formal PNG, CSV and the final "
        "terminal output for Task 08."
    )


if __name__ == "__main__":
    main()