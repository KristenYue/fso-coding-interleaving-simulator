"""
Task 03: RS(255,223) validation over an AWGN OOK channel.

Purpose:
1. Validate the same reedsolo RS implementation used later in FSO experiments.
2. Verify noiseless decoding and correction of up to 16 erroneous RS symbols.
3. Verify rate-aware Eb/N0 noise scaling.
4. Measure raw coded BER, decoded information BER and FER.
5. Report 95% confidence intervals and honest zero-error upper bounds.

RS parameters:
    RS(255,223) over GF(256)
    32 parity symbols
    t = 16 correctable erroneous symbols
    rate = 223/255

OOK convention:
    bit 0 -> 0
    bit 1 -> 1
    threshold = 0.5
    sigma^2 = 1 / (4 * R * Eb/N0)
"""

from pathlib import Path
from math import erfc, sqrt
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import reedsolo


# ============================================================
# 1. Settings
# ============================================================

MASTER_SEED = 20260621

RS_N = 255
RS_K = 223
RS_PARITY_SYMBOLS = RS_N - RS_K
RS_T = RS_PARITY_SYMBOLS // 2
RS_RATE = RS_K / RS_N

INFORMATION_BITS_PER_FRAME = RS_K * 8
CODED_BITS_PER_FRAME = RS_N * 8

EBN0_DB = np.arange(0, 13, 1)

MINIMUM_BIT_ERRORS = 100
MINIMUM_FRAME_ERRORS = 50
MAXIMUM_FRAMES = 10_000

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "03_rs_awgn_validation.csv"
PNG_PATH = RESULTS_DIR / "03_rs_awgn_validation.png"
PDF_PATH = RESULTS_DIR / "03_rs_awgn_validation.pdf"

RS_CODEC = reedsolo.RSCodec(RS_PARITY_SYMBOLS)


# ============================================================
# 2. RS encoder and decoder
# ============================================================

def rs_encode(message_bytes):
    """Encode one 223-byte information message."""
    message_bytes = np.asarray(
        message_bytes,
        dtype=np.uint8,
    )

    if message_bytes.size != RS_K:
        raise ValueError(
            f"Expected {RS_K} information bytes."
        )

    encoded = RS_CODEC.encode(
        bytes(message_bytes)
    )

    codeword = np.frombuffer(
        encoded,
        dtype=np.uint8,
    ).copy()

    if codeword.size != RS_N:
        raise RuntimeError(
            f"Expected {RS_N} coded bytes, "
            f"received {codeword.size}."
        )

    return codeword


def rs_decode(received_codeword):
    """
    Decode one RS codeword.

    Returns:
        success, decoded_message
    """
    received_codeword = np.asarray(
        received_codeword,
        dtype=np.uint8,
    )

    if received_codeword.size != RS_N:
        raise ValueError(
            f"Expected {RS_N} coded bytes."
        )

    try:
        result = RS_CODEC.decode(
            bytes(received_codeword)
        )

        if isinstance(result, tuple):
            decoded_bytes = result[0]
        else:
            decoded_bytes = result

        decoded_message = np.frombuffer(
            bytes(decoded_bytes),
            dtype=np.uint8,
        ).copy()

        decoded_message = decoded_message[:RS_K]

        if decoded_message.size != RS_K:
            raise ValueError(
                "Decoded message has incorrect length."
            )

        return True, decoded_message

    except (
        reedsolo.ReedSolomonError,
        ValueError,
    ):
        # On decoder failure, use the systematic message
        # portion so that residual information errors can
        # still be counted honestly.
        fallback = received_codeword[:RS_K].copy()

        return False, fallback


# ============================================================
# 3. Bit and byte conversion
# ============================================================

def bytes_to_bits(values):
    return np.unpackbits(
        np.asarray(values, dtype=np.uint8),
        bitorder="big",
    )


def bits_to_bytes(bits):
    return np.packbits(
        np.asarray(bits, dtype=np.uint8),
        bitorder="big",
    )


def count_information_bit_errors(
    original_message,
    decoded_message,
):
    original_bits = bytes_to_bits(
        original_message
    )

    decoded_bits = bytes_to_bits(
        decoded_message
    )

    return int(
        np.count_nonzero(
            original_bits != decoded_bits
        )
    )


# ============================================================
# 4. OOK and Eb/N0
# ============================================================

def noise_sigma_from_information_ebn0(
    ebn0_db,
):
    """
    Rate-aware information-bit Eb/N0:

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
            * RS_RATE
            * ebn0_linear
        )
    )

    return sqrt(noise_variance)


def theoretical_raw_coded_ber(ebn0_db):
    """
    Raw hard-decision BER before RS decoding:

        Q(sqrt(R * Eb/N0))
    """
    ebn0_linear = 10.0 ** (
        np.asarray(ebn0_db, dtype=float)
        / 10.0
    )

    argument = np.sqrt(
        RS_RATE * ebn0_linear
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
    """95% Wilson binomial confidence interval."""
    if trials <= 0:
        raise ValueError(
            "Number of trials must be positive."
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


def displayed_value(
    measured_value,
    upper_bound,
    zero_events,
):
    if zero_events:
        return upper_bound

    return measured_value


# ============================================================
# 6. Validation tests
# ============================================================

def run_rs_self_tests(rng):
    """
    Check:
    - noiseless encode/decode;
    - correction of 1 to 16 random erroneous bytes.
    """
    print("\n[1] Running RS self-tests...")

    number_of_tests = 100

    for test_index in range(
        number_of_tests
    ):
        message = rng.integers(
            0,
            256,
            size=RS_K,
            dtype=np.uint8,
        )

        codeword = rs_encode(message)

        success, decoded = rs_decode(
            codeword
        )

        if not success:
            raise RuntimeError(
                "Noiseless RS decoding failed."
            )

        if not np.array_equal(
            message,
            decoded,
        ):
            raise RuntimeError(
                "Noiseless RS message mismatch."
            )

        number_of_errors = (
            test_index % RS_T
        ) + 1

        corrupted = codeword.copy()

        error_positions = rng.choice(
            RS_N,
            size=number_of_errors,
            replace=False,
        )

        error_values = rng.integers(
            1,
            256,
            size=number_of_errors,
            dtype=np.uint8,
        )

        corrupted[error_positions] ^= (
            error_values
        )

        success, decoded = rs_decode(
            corrupted
        )

        if not success:
            raise RuntimeError(
                "RS failed to correct "
                f"{number_of_errors} symbol errors."
            )

        if not np.array_equal(
            message,
            decoded,
        ):
            raise RuntimeError(
                "RS correction produced "
                "an incorrect message."
            )

    print(
        "RS noiseless and random-error "
        "correction tests passed."
    )


# ============================================================
# 7. Monte Carlo simulation
# ============================================================

def simulate_rs_awgn_point(
    ebn0_db,
    rng,
):
    noise_sigma = (
        noise_sigma_from_information_ebn0(
            ebn0_db
        )
    )

    frames = 0

    information_bit_errors = 0
    frame_errors = 0

    raw_coded_bit_errors = 0
    raw_coded_bits = 0

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
            256,
            size=RS_K,
            dtype=np.uint8,
        )

        codeword = rs_encode(message)

        transmitted_bits = bytes_to_bits(
            codeword
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=CODED_BITS_PER_FRAME,
        )

        received_samples = (
            transmitted_bits.astype(float)
            + noise
        )

        detected_bits = (
            received_samples >= 0.5
        ).astype(np.uint8)

        raw_errors_this_frame = int(
            np.count_nonzero(
                detected_bits
                != transmitted_bits
            )
        )

        raw_coded_bit_errors += (
            raw_errors_this_frame
        )

        raw_coded_bits += (
            CODED_BITS_PER_FRAME
        )

        received_codeword = bits_to_bytes(
            detected_bits
        )

        success, decoded_message = (
            rs_decode(
                received_codeword
            )
        )

        information_errors_this_frame = (
            count_information_bit_errors(
                message,
                decoded_message,
            )
        )

        information_bit_errors += (
            information_errors_this_frame
        )

        if information_errors_this_frame > 0:
            frame_errors += 1

        if not success:
            decoder_failures += 1

        if (
            success
            and information_errors_this_frame > 0
        ):
            decoder_miscorrections += 1

        frames += 1

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    information_bits = (
        frames
        * INFORMATION_BITS_PER_FRAME
    )

    decoded_ber = (
        information_bit_errors
        / information_bits
    )

    fer = frame_errors / frames

    raw_coded_ber = (
        raw_coded_bit_errors
        / raw_coded_bits
    )

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
            raw_coded_bit_errors,
            raw_coded_bits,
        )
    )

    raw_theory = float(
        theoretical_raw_coded_ber(
            [ebn0_db]
        )[0]
    )

    return {
        "ebn0_db": float(ebn0_db),
        "frames": frames,
        "information_bits": information_bits,
        "information_bit_errors": (
            information_bit_errors
        ),
        "decoded_ber": decoded_ber,
        "decoded_ber_lower_95": decoded_lower,
        "decoded_ber_upper_95": decoded_upper,
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
        "raw_coded_bits": raw_coded_bits,
        "raw_coded_bit_errors": (
            raw_coded_bit_errors
        ),
        "raw_coded_ber": raw_coded_ber,
        "raw_coded_ber_lower_95": raw_lower,
        "raw_coded_ber_upper_95": raw_upper,
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

def plot_curve_with_upper_bounds(
    axis,
    x,
    measured,
    lower,
    upper,
    zero_mask,
    label,
    marker,
    colour,
):
    displayed = np.array([
        displayed_value(
            value,
            upper_value,
            is_zero,
        )
        for value, upper_value, is_zero
        in zip(
            measured,
            upper,
            zero_mask,
        )
    ])

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
        np.maximum(lower, 1e-10),
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
        row["raw_coded_ber"]
        for row in results
    ])

    raw_lower = np.array([
        row["raw_coded_ber_lower_95"]
        for row in results
    ])

    raw_upper = np.array([
        row["raw_coded_ber_upper_95"]
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

    # Raw channel validation
    raw_axis = axes[0]

    theory_x = np.linspace(
        EBN0_DB.min(),
        EBN0_DB.max(),
        300,
    )

    raw_axis.semilogy(
        theory_x,
        theoretical_raw_coded_ber(
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
        np.maximum(raw_lower, 1e-10),
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
        "(a) Rate-aware OOK channel validation"
    )

    raw_axis.set_ylim(1e-7, 1)
    raw_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )
    raw_axis.legend(fontsize=8)

    # Decoder result
    decoded_axis = axes[1]

    plot_curve_with_upper_bounds(
        decoded_axis,
        ebn0,
        decoded_ber,
        decoded_lower,
        decoded_upper,
        decoded_zero,
        label="Decoded information BER",
        marker="s",
        colour="forestgreen",
    )

    plot_curve_with_upper_bounds(
        decoded_axis,
        ebn0,
        fer,
        fer_lower,
        fer_upper,
        fer_zero,
        label="Frame error rate",
        marker="^",
        colour="firebrick",
    )

    decoded_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    decoded_axis.set_ylabel(
        "Error probability"
    )

    decoded_axis.set_title(
        "(b) RS(255,223) decoding performance"
    )

    decoded_axis.set_ylim(1e-7, 1)
    decoded_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )
    decoded_axis.legend(fontsize=7)

    figure.suptitle(
        "Task 03: RS(255,223) AWGN Validation\n"
        "Hard-decision OOK, rate-aware Eb/N0",
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
    print("=" * 72)
    print("TASK 03: RS(255,223) AWGN VALIDATION")
    print("=" * 72)

    print(
        f"RS parameters: n={RS_N}, "
        f"k={RS_K}, parity={RS_PARITY_SYMBOLS}, "
        f"t={RS_T}"
    )

    print(
        f"Code rate: R={RS_RATE:.6f}"
    )

    print(
        "Noise scaling: "
        "sigma^2 = 1/(4*R*Eb/N0)"
    )

    self_test_rng = np.random.default_rng(
        MASTER_SEED
    )

    run_rs_self_tests(
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
        result = simulate_rs_awgn_point(
            ebn0_db,
            np.random.default_rng(
                point_seed
            ),
        )

        results.append(result)

        if result[
            "zero_decoded_ber_upper_bound"
        ]:
            decoded_text = (
                "decoded BER upper="
                f"{result['decoded_ber_upper_95']:.3e}"
            )
        else:
            decoded_text = (
                "decoded BER="
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
            f"raw={result['raw_coded_ber']:.3e} | "
            f"raw theory pass="
            f"{result['raw_theory_inside_95']} | "
            f"{decoded_text} | "
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

    print("\n" + "=" * 72)
    print("TASK 03 FINISHED")
    print("=" * 72)

    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")

    print(
        "\nRequired checks:"
    )

    print(
        "1. RS self-tests must pass."
    )

    print(
        "2. Raw theory pass should normally be True."
    )

    print(
        "3. Decoded BER and FER must decrease with Eb/N0."
    )

    print(
        "4. Decoder miscorrections should be zero or "
        "explicitly investigated."
    )


if __name__ == "__main__":
    main()