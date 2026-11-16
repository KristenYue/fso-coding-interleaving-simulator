"""
Task 02: Uncoded OOK BER over FSO channels.

This program compares:

Left panel:
- AWGN theoretical BER
- Weak Gamma-Gamma turbulence
- Moderate Gamma-Gamma turbulence
- Strong Gamma-Gamma turbulence

Right panel:
- Moderate turbulence + near-ideal pointing
- Moderate turbulence + moderate pointing
- Moderate turbulence + severe pointing

Important assumptions:
- Uncoded IM/DD OOK
- Perfect channel-state information at the receiver
- Maximum-likelihood threshold h/2
- Independent fading samples per bit
- Pointing gain is normalised by A0
- SNR axis is information-bit Eb/N0
- Zero-error results are plotted using a 95% upper bound
"""

from pathlib import Path
from math import erf, exp, pi, sqrt
import csv

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Experiment settings
# ============================================================

MASTER_SEED = 20260620

EBN0_DB = np.arange(0, 31, 3)

BATCH_BITS = 200_000
MINIMUM_BIT_ERRORS = 200
MAXIMUM_BITS = 5_000_000

TURBULENCE = {
    "Weak": 0.20,
    "Moderate": 0.75,
    "Strong": 2.00,
}

POINTING = {
    "Near-ideal": 0.001,
    "Moderate": 0.05,
    "Severe": 0.10,
}

BEAM_RADIUS_M = 0.30
APERTURE_RADIUS_M = 0.05

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Gamma-Gamma turbulence
# ============================================================

def rytov_to_alpha_beta(rytov_variance):
    """
    Plane-wave Gamma-Gamma mapping.

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
    Unit-mean Gamma-Gamma irradiance.

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


# ============================================================
# 3. Pointing-error model
# ============================================================

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

    xi = (
        equivalent_beam_radius
        / (2.0 * jitter_sigma_m)
    )

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
    Generate the normalised pointing gain h_p/A0.

    This isolates random pointing fading from the fixed
    aperture coupling loss A0.
    """
    radial_displacement = rng.rayleigh(
        scale=jitter_sigma_m,
        size=size,
    )

    normalised_gain = np.exp(
        -2.0
        * radial_displacement ** 2
        / parameters["w_eq"] ** 2
    )

    return normalised_gain


# ============================================================
# 4. OOK and statistics
# ============================================================

def theoretical_awgn_ber(ebn0_db):
    """
    Equal-probability OOK theoretical BER:

        BER = Q(sqrt(Eb/N0))
    """
    from scipy.special import erfc

    ebn0_linear = 10.0 ** (
        np.asarray(ebn0_db, dtype=float) / 10.0
    )

    return 0.5 * erfc(
        np.sqrt(ebn0_linear / 2.0)
    )


def noise_sigma_from_ebn0(ebn0_db):
    """
    For unit-amplitude equal-probability OOK:

        sigma^2 = 1 / (4 * Eb/N0)
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    return sqrt(
        1.0 / (4.0 * ebn0_linear)
    )


def wilson_interval(errors, total_bits):
    """95% Wilson confidence interval."""
    z = 1.959963984540054

    estimate = errors / total_bits

    denominator = (
        1.0 + z ** 2 / total_bits
    )

    centre = (
        estimate
        + z ** 2 / (2.0 * total_bits)
    ) / denominator

    half_width = (
        z
        * sqrt(
            estimate * (1.0 - estimate) / total_bits
            + z ** 2 / (4.0 * total_bits ** 2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def simulate_ber_point(
    ebn0_db,
    gain_generator,
    rng,
):
    """
    Run one Monte Carlo BER point.

    The simulation continues until:
    - at least MINIMUM_BIT_ERRORS are observed, or
    - MAXIMUM_BITS is reached.
    """
    noise_sigma = noise_sigma_from_ebn0(
        ebn0_db
    )

    total_bits = 0
    total_errors = 0

    while (
        total_errors < MINIMUM_BIT_ERRORS
        and total_bits < MAXIMUM_BITS
    ):
        current_batch = min(
            BATCH_BITS,
            MAXIMUM_BITS - total_bits,
        )

        information_bits = rng.integers(
            0,
            2,
            size=current_batch,
            dtype=np.uint8,
        )

        channel_gain = gain_generator(
            current_batch,
            rng,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=current_batch,
        )

        # Received IM/DD OOK signal
        received = (
            channel_gain
            * information_bits.astype(float)
            + noise
        )

        # Perfect-CSI maximum-likelihood threshold
        detected_bits = (
            received >= channel_gain / 2.0
        ).astype(np.uint8)

        total_errors += int(
            np.count_nonzero(
                detected_bits != information_bits
            )
        )

        total_bits += current_batch

    measured_ber = total_errors / total_bits

    lower_95, upper_95 = wilson_interval(
        total_errors,
        total_bits,
    )

    return {
        "ebn0_db": float(ebn0_db),
        "bits": total_bits,
        "errors": total_errors,
        "ber": measured_ber,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "zero_error_upper_bound": total_errors == 0,
    }


# ============================================================
# 5. Channel-generator construction
# ============================================================

def make_turbulence_generator(rytov_variance):
    alpha, beta = rytov_to_alpha_beta(
        rytov_variance
    )

    def generator(size, rng):
        return generate_gamma_gamma(
            alpha,
            beta,
            size,
            rng,
        )

    return generator


def make_combined_generator(
    rytov_variance,
    jitter_sigma_m,
):
    alpha, beta = rytov_to_alpha_beta(
        rytov_variance
    )

    pointing_parameters = (
        calculate_pointing_parameters(
            BEAM_RADIUS_M,
            APERTURE_RADIUS_M,
            jitter_sigma_m,
        )
    )

    def generator(size, rng):
        turbulence_gain = generate_gamma_gamma(
            alpha,
            beta,
            size,
            rng,
        )

        pointing_gain = (
            generate_normalised_pointing_gain(
                pointing_parameters,
                jitter_sigma_m,
                size,
                rng,
            )
        )

        return turbulence_gain * pointing_gain

    return generator


# ============================================================
# 6. Run complete BER experiment
# ============================================================

def run_scenario(
    scenario_name,
    panel_name,
    gain_generator,
    seed_sequence,
):
    """
    Run one complete BER curve.
    """
    print("\n" + "-" * 70)
    print(f"Running: {scenario_name}")
    print("-" * 70)

    point_seeds = seed_sequence.spawn(
        len(EBN0_DB)
    )

    scenario_results = []

    for ebn0_db, point_seed in zip(
        EBN0_DB,
        point_seeds,
    ):
        point_rng = np.random.default_rng(
            point_seed
        )

        result = simulate_ber_point(
            ebn0_db,
            gain_generator,
            point_rng,
        )

        result["panel"] = panel_name
        result["scenario"] = scenario_name

        scenario_results.append(result)

        if result["errors"] == 0:
            displayed_value = result["upper_95"]

            note = (
                f"95% upper bound={displayed_value:.3e}"
            )
        else:
            note = f"BER={result['ber']:.3e}"

        print(
            f"Eb/N0={ebn0_db:2d} dB | "
            f"{note} | "
            f"errors={result['errors']:4d} | "
            f"bits={result['bits']:,}"
        )

    return scenario_results


# ============================================================
# 7. Plotting helper
# ============================================================

def plot_ber_curve(
    axis,
    scenario_results,
    colour,
    marker,
    label,
):
    ebn0 = np.array([
        row["ebn0_db"]
        for row in scenario_results
    ])

    displayed_ber = np.array([
        (
            row["ber"]
            if row["errors"] > 0
            else row["upper_95"]
        )
        for row in scenario_results
    ])

    lower = np.array([
        row["lower_95"]
        for row in scenario_results
    ])

    upper = np.array([
        row["upper_95"]
        for row in scenario_results
    ])

    axis.semilogy(
        ebn0,
        displayed_ber,
        color=colour,
        marker=marker,
        markersize=4,
        linewidth=1.6,
        label=label,
    )

    # Confidence band
    safe_lower = np.maximum(
        lower,
        1e-9,
    )

    axis.fill_between(
        ebn0,
        safe_lower,
        upper,
        color=colour,
        alpha=0.10,
    )

    # Mark zero-error points separately
    zero_mask = np.array([
        row["errors"] == 0
        for row in scenario_results
    ])

    if np.any(zero_mask):
        axis.scatter(
            ebn0[zero_mask],
            displayed_ber[zero_mask],
            marker="v",
            facecolors="none",
            edgecolors=colour,
            s=45,
            zorder=5,
        )


# ============================================================
# 8. Main program
# ============================================================

def main():
    print("=" * 70)
    print("TASK 02: UNCODED OOK OVER FSO CHANNELS")
    print("=" * 70)

    master_seed = np.random.SeedSequence(
        MASTER_SEED
    )

    scenario_seeds = master_seed.spawn(6)

    all_results = []

    # --------------------------------------------------------
    # Left panel: turbulence only
    # --------------------------------------------------------

    turbulence_results = {}

    for index, (
        turbulence_name,
        rytov_variance,
    ) in enumerate(TURBULENCE.items()):
        scenario_name = (
            f"Gamma-Gamma {turbulence_name}"
        )

        results = run_scenario(
            scenario_name=scenario_name,
            panel_name="Turbulence only",
            gain_generator=(
                make_turbulence_generator(
                    rytov_variance
                )
            ),
            seed_sequence=scenario_seeds[index],
        )

        turbulence_results[
            turbulence_name
        ] = results

        all_results.extend(results)

    # --------------------------------------------------------
    # Right panel: moderate turbulence + pointing error
    # --------------------------------------------------------

    pointing_results = {}

    moderate_rytov = TURBULENCE["Moderate"]

    for index, (
        pointing_name,
        jitter_sigma,
    ) in enumerate(POINTING.items()):
        scenario_name = (
            "Moderate turbulence + "
            f"{pointing_name} pointing"
        )

        results = run_scenario(
            scenario_name=scenario_name,
            panel_name="Turbulence plus pointing",
            gain_generator=make_combined_generator(
                moderate_rytov,
                jitter_sigma,
            ),
            seed_sequence=scenario_seeds[
                index + 3
            ],
        )

        pointing_results[
            pointing_name
        ] = results

        all_results.extend(results)

    # --------------------------------------------------------
    # Save raw CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "02_uncoded_fso_ber.csv"
    )

    fieldnames = [
        "panel",
        "scenario",
        "ebn0_db",
        "bits",
        "errors",
        "ber",
        "lower_95",
        "upper_95",
        "zero_error_upper_bound",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_results)

    # --------------------------------------------------------
    # Draw figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
    )

    # Left panel
    left_axis = axes[0]

    awgn_x = np.linspace(
        EBN0_DB.min(),
        EBN0_DB.max(),
        300,
    )

    left_axis.semilogy(
        awgn_x,
        theoretical_awgn_ber(awgn_x),
        "k-",
        linewidth=2,
        label="AWGN theory",
    )

    turbulence_colours = {
        "Weak": "royalblue",
        "Moderate": "forestgreen",
        "Strong": "firebrick",
    }

    turbulence_markers = {
        "Weak": "o",
        "Moderate": "s",
        "Strong": "^",
    }

    for turbulence_name, results in (
        turbulence_results.items()
    ):
        plot_ber_curve(
            left_axis,
            results,
            colour=turbulence_colours[
                turbulence_name
            ],
            marker=turbulence_markers[
                turbulence_name
            ],
            label=(
                f"Gamma-Gamma "
                f"{turbulence_name}"
            ),
        )

    left_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    left_axis.set_ylabel("BER")

    left_axis.set_title(
        "(a) Effect of Gamma-Gamma turbulence"
    )

    left_axis.set_xlim(
        EBN0_DB.min(),
        EBN0_DB.max(),
    )

    left_axis.set_ylim(1e-7, 1)

    left_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    left_axis.legend(fontsize=8)

    # Right panel
    right_axis = axes[1]

    pointing_colours = {
        "Near-ideal": "royalblue",
        "Moderate": "darkorange",
        "Severe": "firebrick",
    }

    pointing_markers = {
        "Near-ideal": "o",
        "Moderate": "s",
        "Severe": "^",
    }

    for pointing_name, results in (
        pointing_results.items()
    ):
        plot_ber_curve(
            right_axis,
            results,
            colour=pointing_colours[
                pointing_name
            ],
            marker=pointing_markers[
                pointing_name
            ],
            label=pointing_name,
        )

    right_axis.set_xlabel(
        "Information-bit Eb/N0 (dB)"
    )

    right_axis.set_ylabel("BER")

    right_axis.set_title(
        "(b) Moderate turbulence + pointing error"
    )

    right_axis.set_xlim(
        EBN0_DB.min(),
        EBN0_DB.max(),
    )

    right_axis.set_ylim(1e-7, 1)

    right_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    right_axis.legend(
        title="Pointing severity",
        fontsize=8,
    )

    figure.suptitle(
        "Task 02: Uncoded IM/DD OOK FSO Channel Characterisation\n"
        "i.i.d. fading, perfect CSI, pointing gain normalised by A0",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    png_path = (
        RESULTS_DIR
        / "02_uncoded_fso_ber.png"
    )

    pdf_path = (
        RESULTS_DIR
        / "02_uncoded_fso_ber.pdf"
    )

    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("\n" + "=" * 70)
    print("TASK 02 FINISHED")
    print("=" * 70)

    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")

    print(
        "\nOpen triangle markers represent "
        "zero-error 95% upper bounds."
    )

    print(
        "This figure uses normalised pointing gain h_p/A0, "
        "so fixed aperture loss A0 is not included."
    )

    plt.show()


if __name__ == "__main__":
    main()