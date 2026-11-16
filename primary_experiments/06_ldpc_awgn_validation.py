"""
Task 06: Rate-matched LDPC validation over AWGN IM/DD OOK.

LDPC:
    n = 512
    generated k = 449
    rate approximately 0.87695

Reference RS:
    RS(255,223), rate approximately 0.87451

Channel:
    OOK: bit 0 -> 0, bit 1 -> 1
    AWGN
    information-bit Eb/N0
    sigma^2 = 1 / (4 * R * Eb/N0)

Decoder:
    Soft-decision belief propagation through pyldpc.

Outputs:
    results/06_ldpc_awgn_validation.csv
    results/06_ldpc_awgn_validation.png
    results/06_ldpc_awgn_validation.pdf
    results/06_ldpc_matrix_seed7.npz
"""

from pathlib import Path
from math import erfc, sqrt
import csv
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyldpc
from pyldpc.utils import binaryproduct


# ============================================================
# 1. Experiment settings
# ============================================================

MASTER_SEED = 20260622

LDPC_N = 512
LDPC_DV = 2
LDPC_DC = 16
LDPC_SEED = 7

MAX_DECODER_ITERATIONS = 100

REFERENCE_RS_RATE = 223 / 255

EBN0_DB = np.arange(0, 11, 1)

MINIMUM_BIT_ERRORS = 100
MINIMUM_FRAME_ERRORS = 50
MAXIMUM_FRAMES = 3_000

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "06_ldpc_awgn_validation.csv"
PNG_PATH = RESULTS_DIR / "06_ldpc_awgn_validation.png"
PDF_PATH = RESULTS_DIR / "06_ldpc_awgn_validation.pdf"
MATRIX_PATH = RESULTS_DIR / "06_ldpc_matrix_seed7.npz"


# ============================================================
# 2. Generate reproducible LDPC matrices
# ============================================================

H_MATRIX, G_MATRIX = pyldpc.make_ldpc(
    LDPC_N,
    LDPC_DV,
    LDPC_DC,
    systematic=True,
    sparse=True,
    seed=LDPC_SEED,
)

LDPC_K = int(G_MATRIX.shape[1])
LDPC_RATE = LDPC_K / LDPC_N

np.savez_compressed(
    MATRIX_PATH,
    H=np.asarray(H_MATRIX),
    G=np.asarray(G_MATRIX),
    n=LDPC_N,
    k=LDPC_K,
    d_v=LDPC_DV,
    d_c=LDPC_DC,
    seed=LDPC_SEED,
)


# ============================================================
# 3. LDPC encoder and decoder
# ============================================================

def ldpc_encode(message_bits):
    """Encode one LDPC information message."""
    message_bits = np.asarray(
        message_bits,
        dtype=np.uint8,
    ).reshape(-1)

    if message_bits.size != LDPC_K:
        raise ValueError(
            f"Expected {LDPC_K} information bits, "
            f"received {message_bits.size}."
        )

    codeword = binaryproduct(
        G_MATRIX,
        message_bits,
    )

    codeword = np.asarray(
        codeword,
        dtype=np.uint8,
    ).reshape(-1)

    if codeword.size != LDPC_N:
        raise RuntimeError(
            f"Encoded codeword has length {codeword.size}, "
            f"expected {LDPC_N}."
        )

    return codeword


def ldpc_decode_from_ook_llr(llr_bit_one):
    """
    Decode external OOK LLR values using pyldpc.

    Input:
        llr_bit_one = log(P(y|bit=1) / P(y|bit=0))

    pyldpc internally expects an LLR positive for bit 0.

    At snr=0 dB, pyldpc uses:

        internal LLR = 2 * y_equivalent

    Therefore:

        y_equivalent = -llr_bit_one / 2

    Important:
        decoded_codeword must be converted to signed int64,
        not uint8. pyldpc.get_message() performs subtraction
        internally, and uint8 produces overflow warnings.
    """
    llr_bit_one = np.asarray(
        llr_bit_one,
        dtype=np.float64,
    ).reshape(-1)

    if llr_bit_one.size != LDPC_N:
        raise ValueError(
            f"Expected {LDPC_N} LLR values, "
            f"received {llr_bit_one.size}."
        )

    equivalent_bpsk_observation = (
        -llr_bit_one / 2.0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        decoded_codeword = pyldpc.decode(
            H_MATRIX,
            equivalent_bpsk_observation,
            snr=0.0,
            maxiter=MAX_DECODER_ITERATIONS,
        )

    # Signed integer is essential for pyldpc.get_message().
    decoded_codeword = np.asarray(
        decoded_codeword,
        dtype=np.int64,
    ).reshape(-1)

    syndrome = binaryproduct(
        H_MATRIX,
        decoded_codeword,
    )

    syndrome = np.asarray(
        syndrome,
        dtype=np.int64,
    ).reshape(-1)

    converged = not np.any(
        syndrome % 2
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        decoded_message = pyldpc.get_message(
            G_MATRIX,
            decoded_codeword,
        )

    decoded_message = np.asarray(
        decoded_message,
        dtype=np.int64,
    ).reshape(-1)

    decoded_message = (
        decoded_message % 2
    ).astype(np.uint8)

    if decoded_message.size != LDPC_K:
        raise RuntimeError(
            "Decoded LDPC message has incorrect length: "
            f"{decoded_message.size}, expected {LDPC_K}."
        )

    return converged, decoded_message


# ============================================================
# 4. OOK and Eb/N0
# ============================================================

def noise_sigma_from_information_ebn0(ebn0_db):
    """
    Rate-aware noise standard deviation:

        Es/N0 = R * Eb/N0
        sigma^2 = 1 / (4 * R * Eb/N0)
    """
    ebn0_linear = 10.0 ** (
        float(ebn0_db) / 10.0
    )

    noise_variance = (
        1.0
        / (
            4.0
            * LDPC_RATE
            * ebn0_linear
        )
    )

    return sqrt(noise_variance)


def ook_llr_bit_one(received, noise_sigma):
    """
    OOK soft information:

        LLR = log(P(y|1) / P(y|0))
            = (2*y - 1) / (2*sigma^2)
    """
    received = np.asarray(
        received,
        dtype=np.float64,
    )

    return (
        2.0 * received - 1.0
    ) / (
        2.0 * noise_sigma ** 2
    )


def theoretical_raw_ber(ebn0_db):
    """
    Raw hard-decision OOK BER:

        BER = Q(sqrt(R * Eb/N0))
    """
    ebn0_linear = 10.0 ** (
        np.asarray(ebn0_db, dtype=float)
        / 10.0
    )

    argument = np.sqrt(
        LDPC_RATE * ebn0_linear
    )

    vectorised_erfc = np.vectorize(
        erfc,
        otypes=[float],
    )

    return (
        0.5
        * vectorised_erfc(
            argument / sqrt(2.0)
        )
    )


# ============================================================
# 5. Statistics
# ============================================================

def wilson_interval(events, trials):
    """95% Wilson confidence interval."""
    if trials <= 0:
        raise ValueError(
            "Trials must be positive."
        )

    z = 1.959963984540054
    estimate = events / trials

    denominator = (
        1.0 + z ** 2 / trials
    )

    centre = (
        estimate
        + z ** 2 / (2.0 * trials)
    ) / denominator

    half_width = (
        z
        * sqrt(
            estimate
            * (1.0 - estimate)
            / trials
            + z ** 2
            / (4.0 * trials ** 2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


# ============================================================
# 6. LDPC self-tests
# ============================================================

def run_self_tests(rng):
    print("\n[1] Running LDPC self-tests...")

    number_of_tests = 20

    for _ in range(number_of_tests):
        message = rng.integers(
            0,
            2,
            size=LDPC_K,
            dtype=np.uint8,
        )

        codeword = ldpc_encode(message)

        syndrome = binaryproduct(
            H_MATRIX,
            codeword,
        )

        syndrome = np.asarray(
            syndrome,
            dtype=np.int64,
        )

        if np.any(syndrome % 2):
            raise RuntimeError(
                "Encoded codeword failed parity check."
            )

        # Strong noiseless LLR:
        # positive for transmitted bit 1.
        strong_llr_bit_one = np.where(
            codeword == 1,
            50.0,
            -50.0,
        )

        converged, decoded_message = (
            ldpc_decode_from_ook_llr(
                strong_llr_bit_one
            )
        )

        if not converged:
            raise RuntimeError(
                "Noiseless LDPC decoder did not converge."
            )

        if not np.array_equal(
            message,
            decoded_message,
        ):
            raise RuntimeError(
                "Noiseless LDPC message mismatch."
            )

    print(
        "LDPC parity-check and noiseless "
        "soft-decoding tests passed."
    )


# ============================================================
# 7. One AWGN Monte Carlo point
# ============================================================

def simulate_ldpc_awgn_point(ebn0_db, rng):
    noise_sigma = (
        noise_sigma_from_information_ebn0(
            ebn0_db
        )
    )

    frames = 0

    information_bit_errors = 0
    frame_errors = 0

    raw_bit_errors = 0
    raw_bits = 0

    decoder_failures = 0
    decoder_miscorrections = 0

    start_time = time.perf_counter()

    while (
        frames < MAXIMUM_FRAMES
        and (
            information_bit_errors
            < MINIMUM_BIT_ERRORS
            or frame_errors
            < MINIMUM_FRAME_ERRORS
        )
    ):
        message = rng.integers(
            0,
            2,
            size=LDPC_K,
            dtype=np.uint8,
        )

        codeword = ldpc_encode(message)

        noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=LDPC_N,
        )

        received = (
            codeword.astype(np.float64)
            + noise
        )

        hard_detected = (
            received >= 0.5
        ).astype(np.uint8)

        raw_bit_errors += int(
            np.count_nonzero(
                hard_detected != codeword
            )
        )

        raw_bits += LDPC_N

        llr_bit_one = ook_llr_bit_one(
            received,
            noise_sigma,
        )

        converged, decoded_message = (
            ldpc_decode_from_ook_llr(
                llr_bit_one
            )
        )

        errors_this_frame = int(
            np.count_nonzero(
                decoded_message != message
            )
        )

        information_bit_errors += (
            errors_this_frame
        )

        if errors_this_frame > 0:
            frame_errors += 1

        if not converged:
            decoder_failures += 1

        if (
            converged
            and errors_this_frame > 0
        ):
            decoder_miscorrections += 1

        frames += 1

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    information_bits = (
        frames * LDPC_K
    )

    decoded_ber = (
        information_bit_errors
        / information_bits
    )

    fer = frame_errors / frames

    raw_ber = raw_bit_errors / raw_bits

    decoded_lower, decoded_upper = (
        wilson_interval(
            information_bit_errors,
            information_bits,
        )
    )

    fer_lower, fer_upper = (
        wilson_interval(
            frame_errors,
            frames,
        )
    )

    raw_lower, raw_upper = (
        wilson_interval(
            raw_bit_errors,
            raw_bits,
        )
    )

    raw_theory = float(
        theoretical_raw_ber(
            [ebn0_db]
        )[0]
    )

    return {
        "ebn0_db": float(ebn0_db),
        "ldpc_n": LDPC_N,
        "ldpc_k": LDPC_K,
        "code_rate": LDPC_RATE,
        "frames": frames,
        "information_bits": information_bits,
        "information_bit_errors": (
            information_bit_errors
        ),
        "decoded_ber": decoded_ber,
        "decoded_ber_lower_95": (
            decoded_lower
        ),
        "decoded_ber_upper_95": (
            decoded_upper
        ),
        "zero_decoded_ber_upper_bound": (
            information_bit_errors == 0
        ),
        "frame_errors": frame_errors,
        "fer": fer,
        "fer_lower_95": fer_lower,
        "fer_upper_95": fer_upper,
        "zero_fer_upper_bound": (
            frame_errors == 0
        ),
        "raw_bits": raw_bits,
        "raw_bit_errors": raw_bit_errors,
        "raw_ber": raw_ber,
        "raw_ber_lower_95": raw_lower,
        "raw_ber_upper_95": raw_upper,
        "raw_theory": raw_theory,
        "raw_theory_inside_95": (
            raw_lower
            <= raw_theory
            <= raw_upper
        ),
        "decoder_failures": (
            decoder_failures
        ),
        "decoder_miscorrections": (
            decoder_miscorrections
        ),
        "runtime_seconds": runtime_seconds,
    }


# ============================================================
# 8. Plotting
# ============================================================

def plot_with_upper_bounds(
    axis,
    x,
    measured,
    lower,
    upper,
    zero_mask,
    colour,
    marker,
    label,
):
    displayed = np.where(
        zero_mask,
        upper,
        measured,
    )

    axis.semilogy(
        x,
        displayed,
        color=colour,
        marker=marker,
        linewidth=1.7,
        markersize=4,
        label=label,
    )

    axis.fill_between(
        x,
        np.maximum(lower, 1e-9),
        upper,
        color=colour,
        alpha=0.12,
    )

    if np.any(zero_mask):
        axis.scatter(
            x[zero_mask],
            displayed[zero_mask],
            marker="v",
            facecolors="none",
            edgecolors=colour,
            s=55,
            zorder=5,
            label=f"{label} zero-error upper bound",
        )


def draw_figure(results):
    ebn0 = np.array([
        row["ebn0_db"]
        for row in results
    ])

    raw_ber = np.array([
        row["raw_ber"]
        for row in results
    ])

    raw_lower = np.array([
        row["raw_ber_lower_95"]
        for row in results
    ])

    raw_upper = np.array([
        row["raw_ber_upper_95"]
        for row in results
    ])

    decoded_ber = np.array([
        row["decoded_ber"]
        for row in results
    ])

    decoded_lower = np.array([
        row["decoded_ber_lower_95"]
        for row in results
    ])

    decoded_upper = np.array([
        row["decoded_ber_upper_95"]
        for row in results
    ])

    decoded_zero = np.array([
        row[
            "zero_decoded_ber_upper_bound"
        ]
        for row in results
    ])

    fer = np.array([
        row["fer"]
        for row in results
    ])

    fer_lower = np.array([
        row["fer_lower_95"]
        for row in results
    ])

    fer_upper = np.array([
        row["fer_upper_95"]
        for row in results
    ])

    fer_zero = np.array([
        row["zero_fer_upper_bound"]
        for row in results
    ])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
    )

    # Raw channel panel
    raw_axis = axes[0]

    theory_x = np.linspace(
        EBN0_DB.min(),
        EBN0_DB.max(),
        300,
    )

    raw_axis.semilogy(
        theory_x,
        theoretical_raw_ber(
            theory_x
        ),
        "k-",
        linewidth=2,
        label="Rate-aware OOK theory",
    )

    raw_axis.semilogy(
        ebn0,
        raw_ber,
        color="royalblue",
        marker="o",
        linewidth=1.6,
        label="Measured raw coded-bit BER",
    )

    raw_axis.fill_between(
        ebn0,
        np.maximum(raw_lower, 1e-9),
        raw_upper,
        color="royalblue",
        alpha=0.12,
    )

    raw_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    raw_axis.set_ylabel(
        "Raw coded-bit BER"
    )

    raw_axis.set_title(
        "(a) Rate-aware OOK validation"
    )

    raw_axis.set_ylim(1e-7, 1)

    raw_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    raw_axis.legend(fontsize=8)

    # LDPC decoder panel
    decoder_axis = axes[1]

    plot_with_upper_bounds(
        decoder_axis,
        ebn0,
        decoded_ber,
        decoded_lower,
        decoded_upper,
        decoded_zero,
        colour="forestgreen",
        marker="s",
        label="Decoded information BER",
    )

    plot_with_upper_bounds(
        decoder_axis,
        ebn0,
        fer,
        fer_lower,
        fer_upper,
        fer_zero,
        colour="firebrick",
        marker="^",
        label="Frame error rate",
    )

    decoder_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    decoder_axis.set_ylabel(
        "Error probability"
    )

    decoder_axis.set_title(
        f"(b) LDPC({LDPC_N},{LDPC_K}) "
        "soft decoding"
    )

    decoder_axis.set_ylim(1e-7, 1)

    decoder_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    decoder_axis.legend(fontsize=7)

    figure.suptitle(
        f"Task 06: Rate-Matched "
        f"LDPC({LDPC_N},{LDPC_K}) AWGN Validation\n"
        f"LDPC rate={LDPC_RATE:.5f}, "
        f"RS reference rate={REFERENCE_RS_RATE:.5f}",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# 9. Main program
# ============================================================

def main():
    print("=" * 76)
    print(
        "TASK 06: RATE-MATCHED LDPC "
        "AWGN VALIDATION"
    )
    print("=" * 76)

    print(
        f"Generated LDPC: "
        f"n={LDPC_N}, k={LDPC_K}"
    )

    print(
        f"LDPC rate: {LDPC_RATE:.6f}"
    )

    print(
        f"RS reference rate: "
        f"{REFERENCE_RS_RATE:.6f}"
    )

    print(
        f"Rate difference: "
        f"{LDPC_RATE - REFERENCE_RS_RATE:+.6f}"
    )

    print(
        f"Maximum decoder iterations: "
        f"{MAX_DECODER_ITERATIONS}"
    )

    self_test_rng = np.random.default_rng(
        MASTER_SEED
    )

    run_self_tests(
        self_test_rng
    )

    print("\n[2] Running AWGN sweep...")

    point_seeds = np.random.SeedSequence(
        MASTER_SEED + 1
    ).spawn(
        len(EBN0_DB)
    )

    results = []

    for ebn0_db, point_seed in zip(
        EBN0_DB,
        point_seeds,
    ):
        result = simulate_ldpc_awgn_point(
            ebn0_db,
            np.random.default_rng(
                point_seed
            ),
        )

        results.append(result)

        if result[
            "zero_decoded_ber_upper_bound"
        ]:
            ber_text = (
                "BER upper="
                f"{result['decoded_ber_upper_95']:.3e}"
            )
        else:
            ber_text = (
                "BER="
                f"{result['decoded_ber']:.3e}"
            )

        if result[
            "zero_fer_upper_bound"
        ]:
            fer_text = (
                "FER upper="
                f"{result['fer_upper_95']:.3e}"
            )
        else:
            fer_text = (
                f"FER={result['fer']:.3e}"
            )

        print(
            f"Eb/N0={ebn0_db:2.0f} dB | "
            f"raw={result['raw_ber']:.3e} | "
            f"raw theory pass="
            f"{result['raw_theory_inside_95']} | "
            f"{ber_text} | "
            f"{fer_text} | "
            f"frames={result['frames']:,} | "
            f"failures={result['decoder_failures']} | "
            f"miscorrections="
            f"{result['decoder_miscorrections']} | "
            f"time={result['runtime_seconds']:.2f}s"
        )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    draw_figure(results)

    print("\n" + "=" * 76)
    print("TASK 06 FINISHED")
    print("=" * 76)

    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"Matrix: {MATRIX_PATH}")

    print("\nRequired checks:")

    print(
        "1. LDPC noiseless tests must pass."
    )

    print(
        "2. Raw theory pass should normally be True."
    )

    print(
        "3. BER and FER should decrease with Eb/N0."
    )

    print(
        "4. Decoder failures must be reported."
    )

    print(
        "5. No uint8 overflow warning should appear."
    )


if __name__ == "__main__":
    main()