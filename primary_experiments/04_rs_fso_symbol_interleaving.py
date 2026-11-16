"""
Task 04B: RS(255,223) over slow-fading FSO with GF(256) symbol-level interleaving.

Purpose:
- Use the validated RS(255,223) codec in the validated FSO fading channel.
- Test RS interleaving at the correct RS symbol level, i.e. byte level.
- Quantify BER and FER versus information-bit Eb/N0 for interleaver depths D.
- Use moderate Gamma-Gamma turbulence + severe pointing error.
- Use perfect receiver CSI.
- Store raw data, Wilson confidence intervals, decoder failures,
  miscorrections, runtime and zero-event upper-bound flags.

Why symbol-level interleaving:
- RS(255,223) is a GF(256) code.
- Its correction capability is 16 erroneous RS symbols, i.e. 16 bytes.
- Therefore the interleaver should distribute fade-induced burst errors
  across RS byte symbols, not only across individual bits.

Channel:
- IM/DD OOK.
- Gamma-Gamma turbulence, Rytov variance = 0.75.
- Severe pointing jitter sigma = 0.10 m.
- Pointing gain is normalised as h_p/A0.
- Fixed aperture loss A0 is not included in the main fading gain.
- Perfect CSI threshold h/2.
- Slow fading over one interleaved byte column:
      one fading block spans D RS symbols = D bytes = 8D OOK bits.

Interleaver:
- D RS codewords are written row-wise as bytes.
- They are transmitted column-wise as RS symbols:
      cw0_byte0, cw1_byte0, ..., cwD-1_byte0,
      cw0_byte1, cw1_byte1, ..., cwD-1_byte1,
      ...
- Each transmitted byte is then mapped to 8 OOK bits.
- One fading gain is applied to each interleaved byte column.

Outputs:
- results/04_rs_fso_symbol_interleaving.csv
- results/04_rs_fso_symbol_interleaving.png
- results/04_rs_fso_symbol_interleaving.pdf
"""

from pathlib import Path
from math import erf, exp, pi, sqrt
import csv
import time

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Experiment settings
# ============================================================

MASTER_SEED = 20260623

RS_N = 255
RS_K = 223
RS_NSYM = RS_N - RS_K
RS_T = RS_NSYM // 2
RS_RATE = RS_K / RS_N

INFORMATION_BITS_PER_FRAME = 8 * RS_K
CODED_BITS_PER_FRAME = 8 * RS_N

# Formal RS-FSO symbol-interleaving sweep.
EBN0_DB = np.arange(0, 37, 3)

INTERLEAVER_DEPTHS = [1, 8, 32, 128]

# Larger formal budget.
# This will take time, but the user said time/cost is acceptable.
MINIMUM_BIT_ERRORS = 200
MINIMUM_FRAME_ERRORS = 100
MAXIMUM_FRAMES = 20_000

# FSO channel: moderate turbulence + severe pointing.
RYTOV_VARIANCE = 0.75
JITTER_SIGMA_M = 0.10

BEAM_RADIUS_M = 0.30
APERTURE_RADIUS_M = 0.05

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "results":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Gamma-Gamma turbulence and pointing error
# ============================================================

def rytov_to_alpha_beta(rytov_variance):
    """
    Plane-wave Gamma-Gamma parameter mapping.

    Input:
        rytov_variance = sigma_R^2
    """
    if rytov_variance <= 0:
        raise ValueError("Rytov variance must be positive.")

    s = float(rytov_variance)

    # sigma_R^(12/5) = (sigma_R^2)^(6/5)
    common = s ** (6.0 / 5.0)

    alpha = 1.0 / (
        exp(
            0.49 * s
            / (1.0 + 1.11 * common) ** (7.0 / 6.0)
        ) - 1.0
    )

    beta = 1.0 / (
        exp(
            0.51 * s
            / (1.0 + 0.69 * common) ** (5.0 / 6.0)
        ) - 1.0
    )

    return alpha, beta


def generate_gamma_gamma(alpha, beta, size, rng):
    """
    Unit-mean Gamma-Gamma irradiance:

        h_a = X * Y

    X ~ Gamma(alpha, scale=1/alpha)
    Y ~ Gamma(beta, scale=1/beta)
    """
    large_scale = rng.gamma(
        shape=alpha,
        scale=1.0 / alpha,
        size=size,
    )

    small_scale = rng.gamma(
        shape=beta,
        scale=1.0 / beta,
        size=size,
    )

    return large_scale * small_scale


def calculate_pointing_parameters(
    beam_radius_m,
    aperture_radius_m,
    jitter_sigma_m,
):
    """
    Farid-Hranilovic zero-boresight pointing model.
    """
    v = (
        sqrt(pi)
        * aperture_radius_m
        / (sqrt(2.0) * beam_radius_m)
    )

    a0 = erf(v) ** 2

    equivalent_beam_radius_squared = (
        beam_radius_m ** 2
        * sqrt(pi)
        * erf(v)
        / (2.0 * v * exp(-(v ** 2)))
    )

    equivalent_beam_radius = sqrt(
        equivalent_beam_radius_squared
    )

    xi = equivalent_beam_radius / (2.0 * jitter_sigma_m)

    return {
        "v": v,
        "A0": a0,
        "w_eq": equivalent_beam_radius,
        "xi": xi,
    }


def generate_normalised_pointing_gain(
    parameters,
    jitter_sigma_m,
    size,
    rng,
):
    """
    Generate normalised pointing gain h_p/A0.

    This excludes fixed aperture loss A0 and retains only random
    pointing fading.
    """
    radial_displacement = rng.rayleigh(
        scale=jitter_sigma_m,
        size=size,
    )

    return np.exp(
        -2.0
        * radial_displacement ** 2
        / parameters["w_eq"] ** 2
    )


def make_combined_fso_gain_generator(
    rytov_variance,
    jitter_sigma_m,
):
    """
    Gamma-Gamma turbulence + normalised pointing fading.
    """
    alpha, beta = rytov_to_alpha_beta(rytov_variance)

    pointing_parameters = calculate_pointing_parameters(
        BEAM_RADIUS_M,
        APERTURE_RADIUS_M,
        jitter_sigma_m,
    )

    def generator(size, rng):
        turbulence_gain = generate_gamma_gamma(
            alpha,
            beta,
            size,
            rng,
        )

        pointing_gain = generate_normalised_pointing_gain(
            pointing_parameters,
            jitter_sigma_m,
            size,
            rng,
        )

        return turbulence_gain * pointing_gain

    return generator, alpha, beta, pointing_parameters


# ============================================================
# 3. GF(256) arithmetic for RS(255,223)
# ============================================================

PRIMITIVE_POLY = 0x11D
FIELD_ORDER = 255

GF_EXP = [0] * 512
GF_LOG = [0] * 256

x = 1
for i in range(FIELD_ORDER):
    GF_EXP[i] = x
    GF_LOG[x] = i

    x <<= 1

    if x & 0x100:
        x ^= PRIMITIVE_POLY

for i in range(FIELD_ORDER, len(GF_EXP)):
    GF_EXP[i] = GF_EXP[i - FIELD_ORDER]


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0

    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("GF division by zero.")

    if a == 0:
        return 0

    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % FIELD_ORDER]


def gf_pow(a, power):
    if power == 0:
        return 1

    if a == 0:
        return 0

    return GF_EXP[(GF_LOG[a] * power) % FIELD_ORDER]


def gf_poly_mul(p, q):
    result = [0] * (len(p) + len(q) - 1)

    for i, p_value in enumerate(p):
        if p_value == 0:
            continue

        for j, q_value in enumerate(q):
            if q_value != 0:
                result[i + j] ^= gf_mul(p_value, q_value)

    return result


def gf_poly_eval_high_to_low(poly, value):
    result = poly[0]

    for coefficient in poly[1:]:
        result = gf_mul(result, value) ^ coefficient

    return result


# ============================================================
# 4. RS encoder and decoder
# ============================================================

def rs_generator_poly(nsym):
    generator = [1]

    for i in range(nsym):
        generator = gf_poly_mul(
            generator,
            [1, gf_pow(2, i)],
        )

    return generator


RS_GENERATOR = rs_generator_poly(RS_NSYM)


def rs_encode_msg(message_bytes):
    """Systematic RS encoding: [223 message bytes | 32 parity bytes]."""
    if len(message_bytes) != RS_K:
        raise ValueError(f"RS message must contain {RS_K} bytes.")

    output = list(message_bytes) + [0] * RS_NSYM

    for i in range(RS_K):
        coefficient = output[i]

        if coefficient != 0:
            for j in range(1, len(RS_GENERATOR)):
                output[i + j] ^= gf_mul(
                    RS_GENERATOR[j],
                    coefficient,
                )

    output[:RS_K] = list(message_bytes)

    return np.asarray(output, dtype=np.uint8)


def rs_calc_syndromes(codeword_bytes):
    return [
        gf_poly_eval_high_to_low(
            list(codeword_bytes),
            gf_pow(2, i),
        )
        for i in range(RS_NSYM)
    ]


def rs_is_codeword_valid(codeword_bytes):
    return max(rs_calc_syndromes(codeword_bytes)) == 0


def berlekamp_massey(syndromes):
    """
    Return the error-locator polynomial in low-to-high order:

        1 + lambda_1 x + ... + lambda_L x^L
    """
    connection = [1] + [0] * RS_NSYM
    previous = [1] + [0] * RS_NSYM

    locator_degree = 0
    shift = 1
    previous_discrepancy = 1

    for n in range(RS_NSYM):
        discrepancy = syndromes[n]

        for i in range(1, locator_degree + 1):
            discrepancy ^= gf_mul(
                connection[i],
                syndromes[n - i],
            )

        if discrepancy == 0:
            shift += 1
            continue

        old_connection = connection.copy()
        scale = gf_div(discrepancy, previous_discrepancy)

        for j in range(shift, RS_NSYM + 1):
            connection[j] ^= gf_mul(
                scale,
                previous[j - shift],
            )

        if 2 * locator_degree <= n:
            locator_degree = n + 1 - locator_degree
            previous = old_connection
            previous_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

    return connection[:locator_degree + 1]


def find_error_positions(error_locator, codeword_length):
    """Locate byte-error positions using Chien search."""
    number_of_errors = len(error_locator) - 1
    positions = []

    locator_high_to_low = list(reversed(error_locator))

    for i in range(codeword_length):
        value = gf_pow(2, (-i) % FIELD_ORDER)

        if gf_poly_eval_high_to_low(locator_high_to_low, value) == 0:
            positions.append(codeword_length - 1 - i)

    if len(positions) != number_of_errors:
        raise ValueError(
            "Could not locate all RS byte errors: "
            f"found {len(positions)}, expected {number_of_errors}."
        )

    return positions


def solve_linear_system_gf256(matrix, vector):
    """Solve A x = b over GF(256) by Gauss-Jordan elimination."""
    size = len(vector)

    augmented = [
        list(matrix[row]) + [vector[row]]
        for row in range(size)
    ]

    pivot_row = 0

    for column in range(size):
        pivot = None

        for row in range(pivot_row, size):
            if augmented[row][column] != 0:
                pivot = row
                break

        if pivot is None:
            raise ValueError("Singular GF(256) system.")

        augmented[pivot_row], augmented[pivot] = (
            augmented[pivot],
            augmented[pivot_row],
        )

        inverse = gf_div(1, augmented[pivot_row][column])

        for col in range(column, size + 1):
            augmented[pivot_row][col] = gf_mul(
                augmented[pivot_row][col],
                inverse,
            )

        for row in range(size):
            if row == pivot_row:
                continue

            factor = augmented[row][column]

            if factor != 0:
                for col in range(column, size + 1):
                    augmented[row][col] ^= gf_mul(
                        factor,
                        augmented[pivot_row][col],
                    )

        pivot_row += 1

    return [augmented[row][size] for row in range(size)]


def correct_error_values(codeword_bytes, syndromes, error_positions):
    """
    Compute byte-error magnitudes from syndrome equations.

    For error position p in high-to-low codeword order:

        S_i = sum_j e_j * alpha^(i * (n - 1 - p_j))
    """
    number_of_errors = len(error_positions)

    if number_of_errors == 0:
        return list(codeword_bytes)

    matrix = []

    for i in range(number_of_errors):
        row = []

        for position in error_positions:
            exponent = (
                i * (len(codeword_bytes) - 1 - position)
            ) % FIELD_ORDER

            row.append(gf_pow(2, exponent))

        matrix.append(row)

    magnitudes = solve_linear_system_gf256(
        matrix,
        syndromes[:number_of_errors],
    )

    corrected = list(codeword_bytes)

    for position, magnitude in zip(error_positions, magnitudes):
        corrected[position] ^= magnitude

    return corrected


def rs_decode_msg(received_codeword_bytes):
    """
    Decode one RS(255,223) codeword.

    Returns:
        decoded_message_bytes, corrected_byte_errors

    Raises:
        ValueError on decoding failure.
    """
    if len(received_codeword_bytes) != RS_N:
        raise ValueError(f"RS codeword must contain {RS_N} bytes.")

    received = list(received_codeword_bytes)
    syndromes = rs_calc_syndromes(received)

    if max(syndromes) == 0:
        return np.asarray(received[:RS_K], dtype=np.uint8), 0

    error_locator = berlekamp_massey(syndromes)
    number_of_errors = len(error_locator) - 1

    if number_of_errors > RS_T:
        raise ValueError(
            f"RS decoder estimated {number_of_errors} byte errors; "
            f"correction capability is {RS_T}."
        )

    error_positions = find_error_positions(
        error_locator,
        RS_N,
    )

    corrected = correct_error_values(
        received,
        syndromes,
        error_positions,
    )

    if not rs_is_codeword_valid(corrected):
        raise ValueError("RS correction failed syndrome check.")

    return (
        np.asarray(corrected[:RS_K], dtype=np.uint8),
        number_of_errors,
    )


# ============================================================
# 5. Bit/byte conversion, noise and statistics
# ============================================================

def bytes_to_bits(byte_array):
    return np.unpackbits(
        np.asarray(byte_array, dtype=np.uint8),
        bitorder="big",
    )


def bits_to_bytes(bit_array):
    return np.packbits(
        np.asarray(bit_array, dtype=np.uint8),
        bitorder="big",
    )


def noise_sigma_from_information_ebn0(ebn0_db):
    """
    Information-bit Eb/N0 with code-rate correction:

        sigma^2 = 1 / (4 * R * Eb/N0)
    """
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)

    return sqrt(1.0 / (4.0 * RS_RATE * ebn0_linear))


def wilson_interval(errors, trials):
    """95% Wilson confidence interval."""
    z = 1.959963984540054

    if trials <= 0:
        raise ValueError("Number of trials must be positive.")

    estimate = errors / trials

    denominator = 1.0 + z ** 2 / trials

    centre = (
        estimate
        + z ** 2 / (2.0 * trials)
    ) / denominator

    half_width = (
        z
        * sqrt(
            estimate * (1.0 - estimate) / trials
            + z ** 2 / (4.0 * trials ** 2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def bit_errors_between(a_bytes, b_bytes):
    return int(
        np.count_nonzero(
            bytes_to_bits(a_bytes) != bytes_to_bits(b_bytes)
        )
    )


# ============================================================
# 6. RS symbol-level interleaver
# ============================================================

def interleave_codeword_symbols(codeword_matrix):
    """
    Row-wise write, column-wise transmit at RS symbol level.

    Input:
        codeword_matrix shape = (D, RS_N), dtype uint8

    Output:
        transmitted byte stream shape = (D * RS_N,), dtype uint8

    Transmission order:
        cw0_sym0, cw1_sym0, ..., cwD-1_sym0,
        cw0_sym1, cw1_sym1, ..., cwD-1_sym1,
        ...
    """
    if codeword_matrix.ndim != 2:
        raise ValueError("codeword_matrix must be 2-D.")

    if codeword_matrix.shape[1] != RS_N:
        raise ValueError(
            f"Each row must contain {RS_N} RS symbols."
        )

    return codeword_matrix.T.reshape(-1)


def deinterleave_codeword_symbols(received_symbols, depth):
    """
    Inverse operation of interleave_codeword_symbols.

    Input:
        received_symbols shape = (D * RS_N,)

    Output:
        received_codeword_matrix shape = (D, RS_N)
    """
    expected_size = depth * RS_N

    if received_symbols.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} received symbols, "
            f"got {received_symbols.size}."
        )

    return received_symbols.reshape(
        RS_N,
        depth,
    ).T


def run_interleaver_self_test(rng):
    for depth in INTERLEAVER_DEPTHS:
        test_matrix = rng.integers(
            0,
            256,
            size=(depth, RS_N),
            dtype=np.uint8,
        )

        interleaved = interleave_codeword_symbols(
            test_matrix,
        )

        restored = deinterleave_codeword_symbols(
            interleaved,
            depth,
        )

        if not np.array_equal(test_matrix, restored):
            raise RuntimeError(
                f"Symbol interleaver inverse test failed for D={depth}."
            )

    return True


def run_rs_self_test(rng, n_frames=200):
    for _ in range(n_frames):
        message = rng.integers(
            0,
            256,
            size=RS_K,
            dtype=np.uint8,
        )

        codeword = rs_encode_msg(message)
        decoded, corrected = rs_decode_msg(codeword)

        if corrected != 0 or not np.array_equal(decoded, message):
            raise RuntimeError("Noiseless RS self-test failed.")

        corrupted = codeword.copy()
        n_errors = int(rng.integers(0, RS_T + 1))

        if n_errors > 0:
            positions = rng.choice(
                RS_N,
                size=n_errors,
                replace=False,
            )

            for position in positions:
                error_value = int(rng.integers(1, 256))
                corrupted[position] ^= error_value

        decoded, _ = rs_decode_msg(corrupted)

        if not np.array_equal(decoded, message):
            raise RuntimeError("RS <=16-byte correction test failed.")

    return True


# ============================================================
# 7. One symbol-interleaved FSO group simulation
# ============================================================

def simulate_one_symbol_interleaved_group(
    depth,
    ebn0_db,
    gain_generator,
    rng,
):
    """
    Simulate one symbol-interleaved group containing D RS codewords.

    Symbol-level slow fading model:
    - D codewords are interleaved by RS byte symbols.
    - There are RS_N transmitted byte-columns per group.
    - One fading coefficient is generated per byte-column.
    - Each fade spans D bytes = 8D OOK bits.
    - Therefore a deep fade corrupts at most one RS symbol per
      codeword in that column, which is the intended RS interleaving model.
    """
    noise_sigma = noise_sigma_from_information_ebn0(
        ebn0_db,
    )

    messages = []
    codeword_rows = []

    for _ in range(depth):
        message = rng.integers(
            0,
            256,
            size=RS_K,
            dtype=np.uint8,
        )

        codeword = rs_encode_msg(message)

        messages.append(message)
        codeword_rows.append(codeword)

    codeword_matrix = np.asarray(
        codeword_rows,
        dtype=np.uint8,
    )

    transmitted_symbols = interleave_codeword_symbols(
        codeword_matrix,
    )

    transmitted_bits = bytes_to_bits(
        transmitted_symbols,
    )

    # One fading coefficient per interleaved RS-symbol column.
    # Each column contains D bytes = 8D OOK bits.
    column_gains = gain_generator(RS_N, rng)

    channel_gain = np.repeat(
        column_gains,
        depth * 8,
    )

    if channel_gain.size != transmitted_bits.size:
        raise RuntimeError(
            "Channel gain length does not match transmitted bit length."
        )

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=transmitted_bits.size,
    )

    received = (
        channel_gain
        * transmitted_bits.astype(float)
        + noise
    )

    # Perfect-CSI ML threshold for OOK.
    detected_bits = (
        received >= channel_gain / 2.0
    ).astype(np.uint8)

    received_symbols_interleaved = bits_to_bytes(
        detected_bits,
    )

    received_codeword_matrix = deinterleave_codeword_symbols(
        received_symbols_interleaved,
        depth,
    )

    group_information_bit_errors = 0
    group_frame_errors = 0
    group_decoder_failures = 0
    group_decoder_miscorrections = 0
    group_corrected_byte_errors = 0

    raw_coded_bit_errors = int(
        np.count_nonzero(
            detected_bits != transmitted_bits
        )
    )

    for frame_index in range(depth):
        received_codeword = received_codeword_matrix[
            frame_index
        ]

        original_message = messages[frame_index]

        decoder_failed_this_frame = False

        try:
            decoded_message, corrected_byte_errors = rs_decode_msg(
                received_codeword,
            )

            group_corrected_byte_errors += corrected_byte_errors

        except ValueError:
            decoder_failed_this_frame = True
            group_decoder_failures += 1

            # Treat failure as hard-decision undecoded payload.
            decoded_message = received_codeword[:RS_K]

        bit_errors = bit_errors_between(
            decoded_message,
            original_message,
        )

        if bit_errors > 0:
            group_frame_errors += 1
            group_information_bit_errors += bit_errors

            if not decoder_failed_this_frame:
                group_decoder_miscorrections += 1

    return {
        "frames": depth,
        "information_bits": depth * INFORMATION_BITS_PER_FRAME,
        "coded_bits": depth * CODED_BITS_PER_FRAME,
        "information_bit_errors": group_information_bit_errors,
        "frame_errors": group_frame_errors,
        "raw_coded_bit_errors": raw_coded_bit_errors,
        "decoder_failures": group_decoder_failures,
        "decoder_miscorrections": group_decoder_miscorrections,
        "corrected_byte_errors_total": group_corrected_byte_errors,
    }


# ============================================================
# 8. One BER/FER point
# ============================================================

def simulate_depth_point(
    depth,
    ebn0_db,
    gain_generator,
    rng,
):
    total_frames = 0
    total_information_bits = 0
    total_coded_bits = 0

    information_bit_errors = 0
    frame_errors = 0
    raw_coded_bit_errors = 0

    decoder_failures = 0
    decoder_miscorrections = 0
    corrected_byte_errors_total = 0

    start_time = time.perf_counter()

    while (
        (
            information_bit_errors < MINIMUM_BIT_ERRORS
            or frame_errors < MINIMUM_FRAME_ERRORS
        )
        and total_frames < MAXIMUM_FRAMES
    ):
        group_result = simulate_one_symbol_interleaved_group(
            depth,
            ebn0_db,
            gain_generator,
            rng,
        )

        total_frames += group_result["frames"]
        total_information_bits += group_result["information_bits"]
        total_coded_bits += group_result["coded_bits"]

        information_bit_errors += group_result[
            "information_bit_errors"
        ]

        frame_errors += group_result[
            "frame_errors"
        ]

        raw_coded_bit_errors += group_result[
            "raw_coded_bit_errors"
        ]

        decoder_failures += group_result[
            "decoder_failures"
        ]

        decoder_miscorrections += group_result[
            "decoder_miscorrections"
        ]

        corrected_byte_errors_total += group_result[
            "corrected_byte_errors_total"
        ]

    runtime_seconds = time.perf_counter() - start_time

    decoded_ber = (
        information_bit_errors / total_information_bits
    )

    fer = frame_errors / total_frames

    raw_coded_ber = raw_coded_bit_errors / total_coded_bits

    decoded_ber_lower, decoded_ber_upper = wilson_interval(
        information_bit_errors,
        total_information_bits,
    )

    fer_lower, fer_upper = wilson_interval(
        frame_errors,
        total_frames,
    )

    raw_coded_ber_lower, raw_coded_ber_upper = wilson_interval(
        raw_coded_bit_errors,
        total_coded_bits,
    )

    return {
        "scenario": "Moderate turbulence + severe pointing",
        "interleaver_type": "GF256 symbol-level",
        "depth": depth,
        "ebn0_db": float(ebn0_db),
        "rs_n": RS_N,
        "rs_k": RS_K,
        "rs_nsym": RS_NSYM,
        "rs_t": RS_T,
        "code_rate": RS_RATE,
        "rytov_variance": RYTOV_VARIANCE,
        "jitter_sigma_m": JITTER_SIGMA_M,
        "coherence_block_rs_symbols": depth,
        "coherence_block_bits": depth * 8,
        "fading_blocks_per_group": RS_N,
        "frames": total_frames,
        "information_bits": total_information_bits,
        "coded_bits": total_coded_bits,
        "information_bit_errors": information_bit_errors,
        "frame_errors": frame_errors,
        "raw_coded_bit_errors": raw_coded_bit_errors,
        "decoded_ber": decoded_ber,
        "decoded_ber_lower_95": decoded_ber_lower,
        "decoded_ber_upper_95": decoded_ber_upper,
        "fer": fer,
        "fer_lower_95": fer_lower,
        "fer_upper_95": fer_upper,
        "raw_coded_ber": raw_coded_ber,
        "raw_coded_ber_lower_95": raw_coded_ber_lower,
        "raw_coded_ber_upper_95": raw_coded_ber_upper,
        "decoder_failures": decoder_failures,
        "decoder_miscorrections": decoder_miscorrections,
        "corrected_byte_errors_total": corrected_byte_errors_total,
        "runtime_seconds": runtime_seconds,
        "zero_decoded_ber_upper_bound": information_bit_errors == 0,
        "zero_fer_upper_bound": frame_errors == 0,
    }


# ============================================================
# 9. Plotting
# ============================================================

def value_for_plot(value, upper, is_zero):
    return upper if is_zero else value


def plot_metric_by_depth(
    axis,
    all_results,
    metric_name,
    lower_name,
    upper_name,
    zero_name,
    ylabel,
    title,
):
    markers = {
        1: "o",
        8: "s",
        32: "^",
        128: "D",
    }

    for depth in INTERLEAVER_DEPTHS:
        rows = [
            row
            for row in all_results
            if row["depth"] == depth
        ]

        rows = sorted(
            rows,
            key=lambda item: item["ebn0_db"],
        )

        x = np.array([
            row["ebn0_db"]
            for row in rows
        ])

        y = np.array([
            row[metric_name]
            for row in rows
        ])

        lower = np.array([
            row[lower_name]
            for row in rows
        ])

        upper = np.array([
            row[upper_name]
            for row in rows
        ])

        zero = np.array([
            row[zero_name]
            for row in rows
        ])

        displayed = np.array([
            value_for_plot(value, upper_value, is_zero)
            for value, upper_value, is_zero in zip(
                y,
                upper,
                zero,
            )
        ])

        axis.semilogy(
            x,
            displayed,
            marker=markers.get(depth, "o"),
            linewidth=1.7,
            markersize=4,
            label=f"D={depth}",
        )

        axis.fill_between(
            x,
            np.maximum(lower, 1e-12),
            upper,
            alpha=0.10,
        )

        if np.any(zero):
            axis.scatter(
                x[zero],
                displayed[zero],
                marker="v",
                facecolors="none",
                s=45,
                zorder=5,
            )

    axis.set_xlabel("Information-bit Eb/N0 (dB)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(EBN0_DB.min(), EBN0_DB.max())
    axis.set_ylim(1e-8, 1)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Interleaver depth")


def draw_figure(all_results):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
    )

    plot_metric_by_depth(
        axis=axes[0],
        all_results=all_results,
        metric_name="decoded_ber",
        lower_name="decoded_ber_lower_95",
        upper_name="decoded_ber_upper_95",
        zero_name="zero_decoded_ber_upper_bound",
        ylabel="Decoded information-bit BER",
        title="(a) RS decoded BER",
    )

    plot_metric_by_depth(
        axis=axes[1],
        all_results=all_results,
        metric_name="fer",
        lower_name="fer_lower_95",
        upper_name="fer_upper_95",
        zero_name="zero_fer_upper_bound",
        ylabel="Frame error rate",
        title="(b) RS frame error rate",
    )

    figure.suptitle(
        "Task 04B: RS(255,223) over Slow-Fading FSO with GF(256) Symbol Interleaving\n"
        "Moderate Gamma-Gamma turbulence + severe normalised pointing, perfect CSI",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    png_path = RESULTS_DIR / "04_rs_fso_symbol_interleaving.png"
    pdf_path = RESULTS_DIR / "04_rs_fso_symbol_interleaving.pdf"

    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    return png_path, pdf_path


# ============================================================
# 10. Main program
# ============================================================

def main():
    print("=" * 86)
    print("TASK 04B: RS(255,223) OVER SLOW-FADING FSO WITH GF(256) SYMBOL INTERLEAVING")
    print("=" * 86)

    gain_generator, alpha, beta, pointing_parameters = (
        make_combined_fso_gain_generator(
            RYTOV_VARIANCE,
            JITTER_SIGMA_M,
        )
    )

    print(
        f"RS parameters: n={RS_N}, k={RS_K}, "
        f"nsym={RS_NSYM}, t={RS_T}"
    )
    print(f"Code rate: R={RS_RATE:.6f}")
    print("Noise scaling: sigma^2 = 1 / (4 * R * Eb/N0)")
    print(
        f"Channel: Rytov variance={RYTOV_VARIANCE}, "
        f"alpha={alpha:.4f}, beta={beta:.4f}"
    )
    print(
        f"Pointing: jitter sigma={JITTER_SIGMA_M:.3f} m, "
        f"A0={pointing_parameters['A0']:.6f}, "
        f"xi={pointing_parameters['xi']:.4f}"
    )
    print("Pointing gain used in simulation: h_p/A0")
    print("Interleaver type: GF(256) RS symbol-level")
    print(
        "Slow fading block: one interleaved RS-symbol column "
        "= D bytes = 8D coded bits"
    )
    print(f"Fading blocks per interleaved group: {RS_N}")
    print(f"Eb/N0 sweep: {EBN0_DB[0]} to {EBN0_DB[-1]} dB")
    print(f"Interleaver depths: {INTERLEAVER_DEPTHS}")
    print(f"Minimum bit errors per point: {MINIMUM_BIT_ERRORS}")
    print(f"Minimum frame errors per point: {MINIMUM_FRAME_ERRORS}")
    print(f"Maximum frames per point: {MAXIMUM_FRAMES:,}")

    master_seed = np.random.SeedSequence(MASTER_SEED)

    self_test_seed, experiment_seed = master_seed.spawn(2)

    self_test_rng = np.random.default_rng(self_test_seed)

    print("\n[1] Running RS and symbol-interleaver self-tests...")

    run_rs_self_test(self_test_rng)
    run_interleaver_self_test(self_test_rng)

    print("RS self-test passed.")
    print("Symbol interleaver inverse self-test passed.")

    print("\n[2] Running RS FSO symbol-interleaving experiment...")

    depth_seeds = experiment_seed.spawn(
        len(INTERLEAVER_DEPTHS)
    )

    all_results = []

    for depth, depth_seed in zip(
        INTERLEAVER_DEPTHS,
        depth_seeds,
    ):
        print("\n" + "-" * 86)
        print(f"Interleaver depth D={depth}")
        print("-" * 86)

        point_seeds = depth_seed.spawn(
            len(EBN0_DB)
        )

        for ebn0_db, point_seed in zip(
            EBN0_DB,
            point_seeds,
        ):
            point_rng = np.random.default_rng(
                point_seed,
            )

            result = simulate_depth_point(
                depth=depth,
                ebn0_db=ebn0_db,
                gain_generator=gain_generator,
                rng=point_rng,
            )

            all_results.append(result)

            if result["zero_decoded_ber_upper_bound"]:
                ber_note = (
                    f"BER upper="
                    f"{result['decoded_ber_upper_95']:.3e}"
                )
            else:
                ber_note = (
                    f"BER={result['decoded_ber']:.3e}"
                )

            if result["zero_fer_upper_bound"]:
                fer_note = (
                    f"FER upper="
                    f"{result['fer_upper_95']:.3e}"
                )
            else:
                fer_note = (
                    f"FER={result['fer']:.3e}"
                )

            print(
                f"D={depth:3d} | "
                f"Eb/N0={ebn0_db:2.0f} dB | "
                f"raw BER={result['raw_coded_ber']:.3e} | "
                f"{ber_note} | {fer_note} | "
                f"frames={result['frames']:,} | "
                f"failures={result['decoder_failures']:,} | "
                f"miscorr={result['decoder_miscorrections']:,} | "
                f"time={result['runtime_seconds']:.1f}s"
            )

    csv_path = RESULTS_DIR / "04_rs_fso_symbol_interleaving.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_results)

    png_path, pdf_path = draw_figure(all_results)

    print("\n" + "=" * 86)
    print("TASK 04B FINISHED")
    print("=" * 86)
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")
    print("Open triangle markers represent zero-event 95% upper bounds.")
    print(
        "This is the RS GF(256) symbol-level interleaving result. "
        "Use this version as the preferred RS interleaving model "
        "unless the thesis explicitly compares bit-level and symbol-level interleavers."
    )

    plt.show()


if __name__ == "__main__":
    main()