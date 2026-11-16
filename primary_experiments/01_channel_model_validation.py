"""
Task 01: Validate the basic FSO channel model.

This program checks:
1. OOK over AWGN against theoretical BER.
2. Gamma-Gamma random samples against theoretical moments/PDF.
3. Pointing-error samples against the Farid-Hranilovic model.

References:
- Al-Habash et al. (2001), DOI: 10.1117/1.1386641
- Farid and Hranilovic (2007), DOI: 10.1109/JLT.2007.899174
"""

from pathlib import Path
from math import erf, exp, pi, sqrt
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc, gamma as gamma_function, kv


# ============================================================
# 1. Experiment settings
# ============================================================

SEED = 20260619
RNG = np.random.default_rng(SEED)

EBN0_DB = np.arange(0, 13, 2)

AWGN_BATCH_BITS = 200_000
AWGN_MIN_ERRORS = 300
AWGN_MAX_BITS = 20_000_000

N_CHANNEL_SAMPLES = 500_000

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
    Plane-wave mapping from Rytov variance sigma_R^2
    to Gamma-Gamma parameters alpha and beta.
    """
    if rytov_variance <= 0:
        raise ValueError("Rytov variance must be positive.")

    s = float(rytov_variance)

    # Since s = sigma_R^2:
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
    A Gamma-Gamma variable is generated as the product of
    two independent unit-mean Gamma random variables.
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


def gamma_gamma_pdf(x, alpha, beta):
    """Theoretical Gamma-Gamma irradiance PDF."""
    coefficient = (
        2.0
        * (alpha * beta) ** ((alpha + beta) / 2.0)
        / (gamma_function(alpha) * gamma_function(beta))
    )

    return (
        coefficient
        * x ** ((alpha + beta) / 2.0 - 1.0)
        * kv(
            alpha - beta,
            2.0 * np.sqrt(alpha * beta * x),
        )
    )


def theoretical_scintillation_index(alpha, beta):
    return (
        1.0 / alpha
        + 1.0 / beta
        + 1.0 / (alpha * beta)
    )


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
    Generate h_p/A0.

    The absolute pointing gain is:
        h_p = A0 * exp(-2*r^2/w_eq^2)

    Here we validate the normalised random component h_p/A0.
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


# ============================================================
# 4. AWGN OOK validation
# ============================================================

def theoretical_ook_ber(ebn0_db):
    """
    BER = Q(sqrt(Eb/N0))
        = 0.5*erfc(sqrt(Eb/N0/2))
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    return 0.5 * erfc(
        np.sqrt(ebn0_linear / 2.0)
    )


def wilson_interval(errors, total_bits):
    """95% Wilson confidence interval."""
    z = 1.959963984540054

    estimate = errors / total_bits
    denominator = 1.0 + z ** 2 / total_bits

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


def simulate_awgn_ook(ebn0_db, rng):
    """
    Equal-probability OOK:
        bit 0 -> 0
        bit 1 -> 1

    Threshold = 0.5.

    Under the energy convention used here:
        sigma^2 = 1 / (4*Eb/N0)
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    noise_sigma = sqrt(1.0 / (4.0 * ebn0_linear))

    total_bits = 0
    total_errors = 0

    while (
        total_errors < AWGN_MIN_ERRORS
        and total_bits < AWGN_MAX_BITS
    ):
        current_batch = min(
            AWGN_BATCH_BITS,
            AWGN_MAX_BITS - total_bits,
        )

        bits = rng.integers(
            0,
            2,
            size=current_batch,
            dtype=np.uint8,
        )

        noise = rng.normal(
            0.0,
            noise_sigma,
            size=current_batch,
        )

        received = bits.astype(float) + noise
        detected = (received >= 0.5).astype(np.uint8)

        total_errors += int(
            np.count_nonzero(bits != detected)
        )
        total_bits += current_batch

    ber = total_errors / total_bits
    lower, upper = wilson_interval(
        total_errors,
        total_bits,
    )

    return {
        "ebn0_db": float(ebn0_db),
        "bits": total_bits,
        "errors": total_errors,
        "ber": ber,
        "lower_95": lower,
        "upper_95": upper,
        "theory": float(theoretical_ook_ber(ebn0_db)),
    }


# ============================================================
# 5. Run validation
# ============================================================

def main():
    print("=" * 60)
    print("TASK 01: FSO CHANNEL MODEL VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------

    print("\n[1] Running AWGN OOK validation...")

    awgn_results = []

    for ebn0_db in EBN0_DB:
        result = simulate_awgn_ook(
            ebn0_db,
            RNG,
        )

        awgn_results.append(result)

        inside_interval = (
            result["lower_95"]
            <= result["theory"]
            <= result["upper_95"]
        )

        print(
            f"Eb/N0={ebn0_db:2d} dB | "
            f"BER={result['ber']:.6e} | "
            f"Theory={result['theory']:.6e} | "
            f"Errors={result['errors']:5d} | "
            f"Pass={inside_interval}"
        )

    # --------------------------------------------------------
    # Gamma-Gamma
    # --------------------------------------------------------

    print("\n[2] Running Gamma-Gamma validation...")

    gamma_results = {}
    summary_rows = []

    for name, rytov_variance in TURBULENCE.items():
        alpha, beta = rytov_to_alpha_beta(
            rytov_variance
        )

        samples = generate_gamma_gamma(
            alpha,
            beta,
            N_CHANNEL_SAMPLES,
            RNG,
        )

        sample_mean = float(np.mean(samples))
        sample_si = float(
            np.var(samples) / sample_mean ** 2
        )

        theory_si = theoretical_scintillation_index(
            alpha,
            beta,
        )

        relative_error = abs(
            sample_si - theory_si
        ) / theory_si

        gamma_results[name] = {
            "alpha": alpha,
            "beta": beta,
            "samples": samples,
        }

        summary_rows.append({
            "model": "Gamma-Gamma",
            "scenario": name,
            "alpha": alpha,
            "beta": beta,
            "A0": "",
            "xi": "",
            "sample_mean": sample_mean,
            "theory_mean": 1.0,
            "validation_error": relative_error,
        })

        print(
            f"{name:8s} | "
            f"alpha={alpha:.4f} | "
            f"beta={beta:.4f} | "
            f"mean={sample_mean:.5f} | "
            f"SI error={100*relative_error:.3f}%"
        )

    # --------------------------------------------------------
    # Pointing error
    # --------------------------------------------------------

    print("\n[3] Running pointing-error validation...")

    pointing_results = {}

    for name, jitter_sigma in POINTING.items():
        parameters = calculate_pointing_parameters(
            BEAM_RADIUS_M,
            APERTURE_RADIUS_M,
            jitter_sigma,
        )

        samples = generate_normalised_pointing_gain(
            parameters,
            jitter_sigma,
            N_CHANNEL_SAMPLES,
            RNG,
        )

        theoretical_mean = (
            parameters["xi"] ** 2
            / (parameters["xi"] ** 2 + 1.0)
        )

        grid = np.linspace(1e-6, 1.0, 300)

        empirical_cdf = (
            np.searchsorted(
                np.sort(samples),
                grid,
                side="right",
            )
            / samples.size
        )

        theoretical_cdf = (
            grid ** parameters["xi"] ** 2
        )

        maximum_cdf_error = float(
            np.max(
                np.abs(
                    empirical_cdf - theoretical_cdf
                )
            )
        )

        pointing_results[name] = {
            "parameters": parameters,
            "grid": grid,
            "empirical_cdf": empirical_cdf,
            "theoretical_cdf": theoretical_cdf,
        }

        summary_rows.append({
            "model": "Pointing error",
            "scenario": name,
            "alpha": "",
            "beta": "",
            "A0": parameters["A0"],
            "xi": parameters["xi"],
            "sample_mean": float(np.mean(samples)),
            "theory_mean": theoretical_mean,
            "validation_error": maximum_cdf_error,
        })

        print(
            f"{name:10s} | "
            f"A0={parameters['A0']:.6f} | "
            f"xi={parameters['xi']:.4f} | "
            f"mean={np.mean(samples):.6f} | "
            f"max CDF error={maximum_cdf_error:.6f}"
        )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    with (
        RESULTS_DIR / "01_awgn_validation.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(awgn_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(awgn_results)

    with (
        RESULTS_DIR / "01_channel_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # Draw figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    # AWGN panel
    axis = axes[0]

    ebn0_values = np.array([
        row["ebn0_db"]
        for row in awgn_results
    ])

    measured_ber = np.array([
        row["ber"]
        for row in awgn_results
    ])

    theoretical_ber = theoretical_ook_ber(
        ebn0_values
    )

    lower = np.array([
        row["lower_95"]
        for row in awgn_results
    ])

    upper = np.array([
        row["upper_95"]
        for row in awgn_results
    ])

    axis.semilogy(
        ebn0_values,
        theoretical_ber,
        "k-",
        linewidth=2,
        label="Theory",
    )

    axis.errorbar(
        ebn0_values,
        measured_ber,
        yerr=[
            measured_ber - lower,
            upper - measured_ber,
        ],
        fmt="o",
        capsize=3,
        label="Monte Carlo 95% CI",
    )

    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel("BER")
    axis.set_title("(a) AWGN OOK validation")
    axis.set_ylim(1e-6, 1)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    # Gamma-Gamma panel
    axis = axes[1]

    x_values = np.linspace(
        0.001,
        4.0,
        500,
    )

    for name, result in gamma_results.items():
        axis.hist(
            result["samples"],
            bins=160,
            range=(0, 4),
            density=True,
            histtype="step",
            linewidth=1,
            label=f"{name} empirical",
        )

        axis.plot(
            x_values,
            gamma_gamma_pdf(
                x_values,
                result["alpha"],
                result["beta"],
            ),
            linewidth=2,
            label=f"{name} theory",
        )

    axis.set_xlabel("Normalised irradiance")
    axis.set_ylabel("PDF")
    axis.set_title("(b) Gamma-Gamma validation")
    axis.set_xlim(0, 4)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)

    # Pointing panel
    axis = axes[2]

    for name, result in pointing_results.items():
        axis.plot(
            result["grid"],
            result["theoretical_cdf"],
            linewidth=2,
            label=f"{name} theory",
        )

        axis.plot(
            result["grid"],
            result["empirical_cdf"],
            "--",
            linewidth=1.5,
            label=f"{name} empirical",
        )

    axis.set_xlabel("Normalised pointing gain hp/A0")
    axis.set_ylabel("CDF")
    axis.set_title("(c) Pointing-error validation")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)

    figure.suptitle(
        "Task 01: FSO Channel Model Validation",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "01_channel_validation.png",
        dpi=180,
        bbox_inches="tight",
    )

    figure.savefig(
        RESULTS_DIR / "01_channel_validation.pdf",
        bbox_inches="tight",
    )

    print("\n" + "=" * 60)
    print("TASK 01 FINISHED")
    print("=" * 60)
    print(f"Results saved in: {RESULTS_DIR}")
    print("Expected key value:")
    print("A0 should be approximately 0.053972.")
    print("\nIf the curves overlap and no error appears,")
    print("send me the figure and console output.")

    plt.show()


if __name__ == "__main__":
    main()