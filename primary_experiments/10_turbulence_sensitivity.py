# 10_turbulence_sensitivity_optimized.py
# ============================================================
# Optimized formal version for dissertation use.
# Changes from quick draft:
#   1. RUN_MODE defaults to "formal".
#   2. SHOW_FIGURE defaults to False for headless/long runs.
#   3. CSV resume is kept, so interrupted runs can be continued.
#   4. Outputs are written to results/10_turbulence_sensitivity_* files.
#
# Run from the same directory as:
#   04c_rs_fso_fixed_coherence_interleaving.py
#   07b_ldpc_fso_group_aware_refinement.py
# and with results/08_rs_ldpc_final_summary.csv already available.
# ============================================================

"""
TASK 10
Selected-configuration turbulence sensitivity study.

This secondary experiment tests whether the practical RS/LDPC designs
selected under Moderate turbulence remain meaningful under Weak and Strong
Gamma-Gamma turbulence.

Selected configurations from Task 08:
    Lc=64 bits:  RS D=8,   LDPC D=128
    Lc=256 bits: RS D=32,  LDPC D=128

Moderate-turbulence thresholds are reused from Task 08. Only Weak and Strong
curves are newly simulated.

Run workflow:
    1. RUN_MODE = "formal"  -> validate the workflow.
    2. RUN_MODE = "formal" -> create dissertation sensitivity data.

The formal CSV supports resume after interruption.
"""

from pathlib import Path
import csv
import importlib.util
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_MODE = "formal"
SHOW_FIGURE = False
MASTER_SEED = 20260622
TARGET_BER = 1e-3

TURBULENCE_PARAMETERS = {
    "Weak": {
        "alpha": 11.650964651134455,
        "beta": 10.121669213873462,
    },
    "Moderate": {
        "alpha": 4.868481548059242,
        "beta": 3.164867988481172,
    },
    "Strong": {
        "alpha": 3.992947239231877,
        "beta": 1.701757574069907,
    },
}

SELECTED_CONFIGURATIONS = [
    {"code": "RS", "coherence_bits": 64, "depth": 8},
    {"code": "RS", "coherence_bits": 256, "depth": 32},
    {"code": "LDPC", "coherence_bits": 64, "depth": 128},
    {"code": "LDPC", "coherence_bits": 256, "depth": 128},
]

if RUN_MODE.lower() == "quick":
    # Workflow-only smoke test: one new point per turbulence/configuration.
    # These data are deliberately kept separate from the formal CSV.
    SNR_OFFSETS_DB = [0.0]
    RS_INDEPENDENT_GROUPS = 2
    LDPC_MIN_GROUPS = 1
    LDPC_MAX_GROUPS = 1
    LDPC_MIN_BIT_ERRORS = 1
    LDPC_MIN_FRAME_ERRORS = 1

elif RUN_MODE.lower() == "formal":
    # Local sweep around each Moderate-turbulence operating point.
    SNR_OFFSETS_DB = [-9.0, -3.0, 0.0, 3.0, 9.0]

    # RS is inexpensive, so use more independent groups.
    RS_INDEPENDENT_GROUPS = 100

    # LDPC is expensive. Twenty independent D=128 groups already contain
    # 2560 codewords; allow up to 30 groups when error targets are unmet.
    LDPC_MIN_GROUPS = 20
    LDPC_MAX_GROUPS = 30
    LDPC_MIN_BIT_ERRORS = 100
    LDPC_MIN_FRAME_ERRORS = 40

else:
    raise ValueError("RUN_MODE must be 'quick' or 'formal'.")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RS_MODULE_PATH = ROOT / "04c_rs_fso_fixed_coherence_interleaving.py"
LDPC_MODULE_PATH = ROOT / "07b_ldpc_fso_group_aware_refinement.py"
TASK08_SUMMARY_PATH = RESULTS_DIR / "08_rs_ldpc_final_summary.csv"

if RUN_MODE.lower() == "formal":
    OUTPUT_STEM = "10_turbulence_sensitivity"
else:
    OUTPUT_STEM = "10_turbulence_sensitivity_quick"

RAW_CSV_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_raw.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_summary.csv"
CURVES_PNG_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_curves.png"
CURVES_PDF_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_curves.pdf"
SUMMARY_PNG_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_thresholds.png"
SUMMARY_PDF_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_thresholds.pdf"


# =============================================================================
# MODULE LOADING
# =============================================================================

def load_python_module(module_name, path):
    if not path.exists():
        raise FileNotFoundError(f"Required module not found:\n{path}")

    specification = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# =============================================================================
# STATISTICS
# =============================================================================

def group_confidence_interval(group_values, total_events, total_trials):
    """95% confidence interval across independent interleaver groups."""

    values = np.asarray(group_values, dtype=float)

    if values.size == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

    if values.size > 1:
        standard_error = float(
            np.std(values, ddof=1) / np.sqrt(values.size)
        )

        lower = max(0.0, mean_value - 1.96 * standard_error)
        upper = min(1.0, mean_value + 1.96 * standard_error)
    else:
        lower = 0.0
        upper = 1.0

    # A zero sample mean is not a zero true probability.
    if total_events == 0 and total_trials > 0:
        upper = max(upper, 3.0 / total_trials)

    return mean_value, lower, upper


def log_crossing(x_values, y_values, target):
    """Interpolate a decreasing BER curve in log10(BER)."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for index in range(len(x) - 1):
        if y[index] >= target and y[index + 1] <= target:
            log_left = np.log10(y[index])
            log_right = np.log10(y[index + 1])
            log_target = np.log10(target)

            if log_left == log_right:
                return float(0.5 * (x[index] + x[index + 1]))

            fraction = (
                (log_target - log_left)
                / (log_right - log_left)
            )

            return float(
                x[index]
                + fraction * (x[index + 1] - x[index])
            )

    return np.nan


# =============================================================================
# TASK 08 REFERENCE CONFIGURATIONS
# =============================================================================

def read_task08_selected_thresholds():
    if not TASK08_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Task 08 summary not found:\n{TASK08_SUMMARY_PATH}"
        )

    rows = []

    with TASK08_SUMMARY_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for raw in reader:
            rows.append({
                "code": raw["code"],
                "coherence_bits": int(float(raw["coherence_bits"])),
                "depth": int(float(raw["interleaver_depth"])),
                "required_ebn0_db": float(raw["required_ebn0_db"]),
            })

    references = {}

    for configuration in SELECTED_CONFIGURATIONS:
        key = (
            configuration["code"],
            configuration["coherence_bits"],
            configuration["depth"],
        )

        matches = [
            row for row in rows
            if (
                row["code"],
                row["coherence_bits"],
                row["depth"],
            ) == key
        ]

        if len(matches) != 1:
            raise RuntimeError(f"Task 08 reference missing for {key}.")

        references[key] = matches[0]["required_ebn0_db"]

    return references


# =============================================================================
# RS GROUP-AWARE SENSITIVITY POINT
# =============================================================================

def simulate_rs_condition(
    rs_module,
    turbulence_name,
    coherence_bits,
    depth,
    ebn0_db,
    seed,
):
    if coherence_bits % 8 != 0:
        raise ValueError("RS coherence length must be a multiple of 8 bits.")

    coherence_symbols = coherence_bits // 8
    parameters = TURBULENCE_PARAMETERS[turbulence_name]

    rs_module.ALPHA = parameters["alpha"]
    rs_module.BETA = parameters["beta"]
    rs_module.COHERENCE_SYMBOLS = coherence_symbols
    rs_module.USE_NORMALIZED_POINTING = True

    np.random.seed(seed)

    total_information_bits = 0
    total_coded_bits = 0
    total_information_errors = 0
    total_raw_errors = 0
    total_frame_errors = 0
    total_failures = 0
    total_miscorrections = 0

    group_ber_values = []
    group_fer_values = []

    start_time = time.perf_counter()

    for _ in range(RS_INDEPENDENT_GROUPS):
        message_matrix = np.random.randint(
            0,
            256,
            size=(depth, rs_module.RS_K),
            dtype=np.uint8,
        )

        codeword_matrix = np.empty(
            (depth, rs_module.RS_N),
            dtype=np.uint8,
        )

        for frame_index in range(depth):
            codeword_matrix[frame_index] = rs_module.rs_encode(
                message_matrix[frame_index]
            )

        transmitted_bytes = rs_module.symbol_interleave(codeword_matrix)

        channel_symbol_gain = rs_module.make_fixed_coherence_fading(
            len(transmitted_bytes),
            coherence_symbols,
        )

        transmitted_bits = rs_module.bytes_to_bits(transmitted_bytes)
        channel_bit_gain = np.repeat(channel_symbol_gain, 8)

        received_bits = rs_module.ook_awgn_perfect_csi(
            transmitted_bits,
            channel_bit_gain,
            ebn0_db,
        )

        group_raw_errors = int(
            np.count_nonzero(received_bits != transmitted_bits)
        )

        received_bytes = rs_module.bits_to_bytes(received_bits)
        received_codewords = rs_module.symbol_deinterleave(
            received_bytes,
            depth,
        )

        group_information_errors = 0
        group_frame_errors = 0

        for frame_index in range(depth):
            success, decoded_message = rs_module.rs_decode(
                received_codewords[frame_index]
            )

            bit_errors = rs_module.count_bit_errors(
                decoded_message,
                message_matrix[frame_index],
            )

            group_information_errors += bit_errors

            if bit_errors > 0:
                group_frame_errors += 1

            if not success:
                total_failures += 1

            if success and bit_errors > 0:
                total_miscorrections += 1

        group_information_bits = depth * rs_module.RS_K * 8
        group_coded_bits = depth * rs_module.RS_N * 8

        group_ber_values.append(
            group_information_errors / group_information_bits
        )

        group_fer_values.append(group_frame_errors / depth)

        total_information_bits += group_information_bits
        total_coded_bits += group_coded_bits
        total_information_errors += group_information_errors
        total_raw_errors += group_raw_errors
        total_frame_errors += group_frame_errors

    runtime_seconds = time.perf_counter() - start_time

    ber, ber_lower, ber_upper = group_confidence_interval(
        group_ber_values,
        total_information_errors,
        total_information_bits,
    )

    fer, fer_lower, fer_upper = group_confidence_interval(
        group_fer_values,
        total_frame_errors,
        RS_INDEPENDENT_GROUPS * depth,
    )

    return {
        "run_mode": RUN_MODE.lower(),
        "code": "RS",
        "turbulence": turbulence_name,
        "coherence_bits": coherence_bits,
        "depth": depth,
        "ebn0_db": float(ebn0_db),
        "ber": ber,
        "ber_ci_lower": ber_lower,
        "ber_ci_upper": ber_upper,
        "fer": fer,
        "fer_ci_lower": fer_lower,
        "fer_ci_upper": fer_upper,
        "independent_groups": RS_INDEPENDENT_GROUPS,
        "frames": RS_INDEPENDENT_GROUPS * depth,
        "information_bits": total_information_bits,
        "information_bit_errors": total_information_errors,
        "coded_bits": total_coded_bits,
        "raw_bit_errors": total_raw_errors,
        "decoder_failures": total_failures,
        "undetected_codeword_errors": total_miscorrections,
        "runtime_seconds": runtime_seconds,
        "runtime_per_frame_ms": (
            1000.0 * runtime_seconds
            / (RS_INDEPENDENT_GROUPS * depth)
        ),
    }


# =============================================================================
# LDPC GROUP-AWARE SENSITIVITY POINT
# =============================================================================

def configure_ldpc_module(ldpc_module, turbulence_name, depth):
    parameters = TURBULENCE_PARAMETERS[turbulence_name]

    ldpc_module.ALPHA_MODERATE = parameters["alpha"]
    ldpc_module.BETA_MODERATE = parameters["beta"]
    ldpc_module.MIN_INDEPENDENT_GROUPS = LDPC_MIN_GROUPS
    ldpc_module.MIN_INFORMATION_BIT_ERRORS = LDPC_MIN_BIT_ERRORS
    ldpc_module.MIN_FRAME_ERRORS = LDPC_MIN_FRAME_ERRORS
    ldpc_module.MAX_FRAMES = depth * LDPC_MAX_GROUPS


def simulate_ldpc_condition(
    ldpc_module,
    parity_check_matrix,
    generator_matrix,
    information_length,
    code_rate,
    turbulence_name,
    coherence_bits,
    depth,
    ebn0_db,
    seed,
):
    configure_ldpc_module(ldpc_module, turbulence_name, depth)

    rng = np.random.default_rng(seed)

    raw = ldpc_module.simulate_condition(
        parity_check_matrix,
        generator_matrix,
        information_length,
        code_rate,
        float(ebn0_db),
        coherence_bits,
        depth,
        rng,
    )

    return {
        "run_mode": RUN_MODE.lower(),
        "code": "LDPC",
        "turbulence": turbulence_name,
        "coherence_bits": coherence_bits,
        "depth": depth,
        "ebn0_db": float(ebn0_db),
        "ber": raw["information_ber"],
        "ber_ci_lower": raw["information_ber_ci_lower"],
        "ber_ci_upper": raw["information_ber_ci_upper"],
        "fer": raw["fer"],
        "fer_ci_lower": raw["fer_ci_lower"],
        "fer_ci_upper": raw["fer_ci_upper"],
        "independent_groups": raw["independent_groups"],
        "frames": raw["frames"],
        "information_bits": raw["information_bits"],
        "information_bit_errors": raw["information_bit_errors"],
        "coded_bits": raw["coded_bits"],
        "raw_bit_errors": raw["raw_bit_errors"],
        "decoder_failures": raw["decoder_failures"],
        "undetected_codeword_errors": raw[
            "undetected_codeword_errors"
        ],
        "runtime_seconds": raw["runtime_seconds"],
        "runtime_per_frame_ms": raw["runtime_per_frame_ms"],
    }


# =============================================================================
# CSV RESUME
# =============================================================================

INTEGER_FIELDS = {
    "coherence_bits",
    "depth",
    "independent_groups",
    "frames",
    "information_bits",
    "information_bit_errors",
    "coded_bits",
    "raw_bit_errors",
    "decoder_failures",
    "undetected_codeword_errors",
}


def save_csv(path, rows):
    if not rows:
        return

    rows = sorted(
        rows,
        key=lambda row: (
            row.get("code", ""),
            row.get("coherence_bits", 0),
            row.get("depth", 0),
            row.get("turbulence", ""),
            row.get("ebn0_db", row.get("required_ebn0_db", 0.0)),
        ),
    )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def load_existing_csv(path):
    if not path.exists():
        return []

    rows = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for raw in reader:
            converted = {}

            for key, value in raw.items():
                if key in {"run_mode", "code", "turbulence"}:
                    converted[key] = value
                elif key in INTEGER_FIELDS:
                    converted[key] = int(float(value))
                else:
                    converted[key] = float(value)

            rows.append(converted)

    return rows


def row_key(row):
    return (
        row["code"],
        row["turbulence"],
        int(row["coherence_bits"]),
        int(row["depth"]),
        round(float(row["ebn0_db"]), 6),
    )


# =============================================================================
# SUMMARY THRESHOLDS
# =============================================================================

def calculate_threshold_summary(raw_rows, moderate_references):
    summary = []

    for configuration in SELECTED_CONFIGURATIONS:
        code = configuration["code"]
        coherence_bits = configuration["coherence_bits"]
        depth = configuration["depth"]

        reference_key = (code, coherence_bits, depth)
        moderate_threshold = moderate_references[reference_key]

        for turbulence_name in ["Weak", "Moderate", "Strong"]:
            if turbulence_name == "Moderate":
                threshold = moderate_threshold
                optimistic = np.nan
                pessimistic = np.nan
            else:
                selected = [
                    row for row in raw_rows
                    if row["code"] == code
                    and row["turbulence"] == turbulence_name
                    and row["coherence_bits"] == coherence_bits
                    and row["depth"] == depth
                ]

                selected.sort(key=lambda row: row["ebn0_db"])

                x = [row["ebn0_db"] for row in selected]
                y = [row["ber"] for row in selected]
                lower = [row["ber_ci_lower"] for row in selected]
                upper = [row["ber_ci_upper"] for row in selected]

                threshold = log_crossing(x, y, TARGET_BER)
                optimistic = log_crossing(x, lower, TARGET_BER)
                pessimistic = log_crossing(x, upper, TARGET_BER)

            summary.append({
                "run_mode": RUN_MODE.lower(),
                "code": code,
                "turbulence": turbulence_name,
                "coherence_bits": coherence_bits,
                "depth": depth,
                "target_ber": TARGET_BER,
                "required_ebn0_db": threshold,
                "required_ebn0_optimistic_db": optimistic,
                "required_ebn0_pessimistic_db": pessimistic,
                "moderate_reference_ebn0_db": moderate_threshold,
                "sensitivity_shift_from_moderate_db": (
                    threshold - moderate_threshold
                    if np.isfinite(threshold)
                    else np.nan
                ),
                "target_bracketed": bool(np.isfinite(threshold)),
            })

    return summary


# =============================================================================
# PLOTS
# =============================================================================

def plot_curves(raw_rows, moderate_references):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)

    colours = {"Weak": "forestgreen", "Strong": "firebrick"}
    markers = {"Weak": "o", "Strong": "s"}

    for axis, configuration in zip(axes.flat, SELECTED_CONFIGURATIONS):
        code = configuration["code"]
        coherence_bits = configuration["coherence_bits"]
        depth = configuration["depth"]
        reference = moderate_references[(code, coherence_bits, depth)]

        for turbulence_name in ["Weak", "Strong"]:
            selected = [
                row for row in raw_rows
                if row["code"] == code
                and row["turbulence"] == turbulence_name
                and row["coherence_bits"] == coherence_bits
                and row["depth"] == depth
            ]

            selected.sort(key=lambda row: row["ebn0_db"])

            x = np.array([row["ebn0_db"] for row in selected])
            y = np.array([row["ber"] for row in selected])
            lower = np.array([row["ber_ci_lower"] for row in selected])
            upper = np.array([row["ber_ci_upper"] for row in selected])

            axis.semilogy(
                x,
                np.maximum(y, 1e-7),
                color=colours[turbulence_name],
                marker=markers[turbulence_name],
                linewidth=2,
                label=turbulence_name,
            )

            axis.fill_between(
                x,
                np.maximum(lower, 1e-7),
                np.maximum(upper, 1e-7),
                color=colours[turbulence_name],
                alpha=0.13,
            )

        axis.scatter(
            [reference],
            [TARGET_BER],
            color="royalblue",
            marker="D",
            s=65,
            label="Moderate Task 08 reference",
            zorder=5,
        )

        axis.axhline(
            TARGET_BER,
            color="grey",
            linestyle="--",
            linewidth=1.2,
        )

        axis.set_title(
            f"{code}: Lc={coherence_bits} bits, D={depth}"
        )

        axis.set_xlabel("Information-bit Eb/N0 (dB)")
        axis.set_ylabel("Decoded information BER")
        axis.set_ylim(1e-6, 3e-2)
        axis.grid(True, which="both", alpha=0.32)
        axis.legend(fontsize=8)

    fig.suptitle(
        "Task 10: Turbulence Sensitivity of Selected RS/LDPC Designs\n"
        "Moderate Task 08 operating point with new Weak/Strong local sweeps",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(CURVES_PNG_PATH, dpi=180, bbox_inches="tight")
    fig.savefig(CURVES_PDF_PATH, bbox_inches="tight")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def plot_threshold_summary(summary_rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    turbulence_order = ["Weak", "Moderate", "Strong"]
    colours = {
        "Weak": "forestgreen",
        "Moderate": "royalblue",
        "Strong": "firebrick",
    }

    for axis, coherence_bits in zip(axes, [64, 256]):
        x_positions = np.arange(2)
        width = 0.24

        for turbulence_index, turbulence_name in enumerate(turbulence_order):
            values = []

            for code in ["RS", "LDPC"]:
                match = next(
                    row for row in summary_rows
                    if row["code"] == code
                    and row["turbulence"] == turbulence_name
                    and row["coherence_bits"] == coherence_bits
                )

                values.append(match["required_ebn0_db"])

            offsets = (
                x_positions
                + (turbulence_index - 1) * width
            )

            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=colours[turbulence_name],
                label=turbulence_name,
                alpha=0.85,
            )

            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.15,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(["RS selected D", "LDPC D=128"])
        axis.set_ylabel("Required Eb/N0 at BER=1e-3 (dB)")
        axis.set_title(f"Lc={coherence_bits} transmitted bits")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle(
        "Task 10: Sensitivity of the Selected Designs to Turbulence Strength",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(SUMMARY_PNG_PATH, dpi=180, bbox_inches="tight")
    fig.savefig(SUMMARY_PDF_PATH, bbox_inches="tight")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 92)
    print("TASK 10: SELECTED-DESIGN TURBULENCE SENSITIVITY")
    print("=" * 92)
    print(f"Run mode: {RUN_MODE.upper()}")
    print(f"SNR offsets from Moderate reference: {SNR_OFFSETS_DB}")

    rs_module = load_python_module("task10_rs_base", RS_MODULE_PATH)
    ldpc_module = load_python_module("task10_ldpc_base", LDPC_MODULE_PATH)

    moderate_references = read_task08_selected_thresholds()

    print("\nSelected Task 08 configurations:")
    for key, value in moderate_references.items():
        print(
            f"  {key[0]:4s} | Lc={key[1]:3d} bits | "
            f"D={key[2]:3d} | Moderate reference={value:.3f} dB"
        )

    print("\n[1] Running codec and interleaver self-tests...")

    np.random.seed(12345)
    rs_module.run_self_tests()

    (
        parity_check_matrix,
        generator_matrix,
        information_length,
        code_rate,
    ) = ldpc_module.construct_ldpc()

    ldpc_module.run_self_tests(
        parity_check_matrix,
        generator_matrix,
        information_length,
    )

    print("Task 10 self-tests passed.")

    existing_rows = load_existing_csv(RAW_CSV_PATH)
    completed_keys = {row_key(row) for row in existing_rows}
    raw_rows = list(existing_rows)

    conditions = []

    for configuration in SELECTED_CONFIGURATIONS:
        key = (
            configuration["code"],
            configuration["coherence_bits"],
            configuration["depth"],
        )

        moderate_reference = moderate_references[key]

        for turbulence_name in ["Weak", "Strong"]:
            for offset_db in SNR_OFFSETS_DB:
                conditions.append({
                    **configuration,
                    "turbulence": turbulence_name,
                    "ebn0_db": moderate_reference + offset_db,
                })

    print(
        f"\n[2] Running {len(conditions)} sensitivity conditions "
        f"({len(existing_rows)} already completed)..."
    )

    for condition_index, condition in enumerate(conditions, start=1):
        key = (
            condition["code"],
            condition["turbulence"],
            condition["coherence_bits"],
            condition["depth"],
            round(condition["ebn0_db"], 6),
        )

        if key in completed_keys:
            print(
                f"[{condition_index:2d}/{len(conditions)}] {key} | completed"
            )
            continue

        seed_sequence = np.random.SeedSequence([
            MASTER_SEED,
            1 if condition["code"] == "RS" else 2,
            1 if condition["turbulence"] == "Weak" else 3,
            condition["coherence_bits"],
            condition["depth"],
            int(round(1000 * condition["ebn0_db"])),
        ])

        seed = int(seed_sequence.generate_state(1)[0])

        if condition["code"] == "RS":
            result = simulate_rs_condition(
                rs_module,
                condition["turbulence"],
                condition["coherence_bits"],
                condition["depth"],
                condition["ebn0_db"],
                seed,
            )
        else:
            result = simulate_ldpc_condition(
                ldpc_module,
                parity_check_matrix,
                generator_matrix,
                information_length,
                code_rate,
                condition["turbulence"],
                condition["coherence_bits"],
                condition["depth"],
                condition["ebn0_db"],
                seed,
            )

        raw_rows.append(result)
        completed_keys.add(row_key(result))
        save_csv(RAW_CSV_PATH, raw_rows)

        print(
            f"[{condition_index:2d}/{len(conditions)}] "
            f"{result['code']:4s} | {result['turbulence']:6s} | "
            f"Lc={result['coherence_bits']:3d} | D={result['depth']:3d} | "
            f"Eb/N0={result['ebn0_db']:6.2f} dB | "
            f"BER={result['ber']:.3e} | "
            f"CI=[{result['ber_ci_lower']:.3e}, "
            f"{result['ber_ci_upper']:.3e}] | "
            f"groups={result['independent_groups']:3d} | "
            f"time={result['runtime_seconds']:.1f}s"
        )

    print("\n[3] Calculating sensitivity thresholds...")

    summary_rows = calculate_threshold_summary(
        raw_rows,
        moderate_references,
    )

    save_csv(SUMMARY_CSV_PATH, summary_rows)

    print("\n" + "=" * 108)
    print("REQUIRED Eb/N0 AT BER=1e-3")
    print("=" * 108)

    for row in summary_rows:
        threshold_text = (
            f"{row['required_ebn0_db']:.3f} dB"
            if np.isfinite(row["required_ebn0_db"])
            else "NOT BRACKETED"
        )

        shift_text = (
            f"{row['sensitivity_shift_from_moderate_db']:+.3f} dB"
            if np.isfinite(row["sensitivity_shift_from_moderate_db"])
            else "N/A"
        )

        print(
            f"{row['code']:4s} | {row['turbulence']:8s} | "
            f"Lc={row['coherence_bits']:3d} | D={row['depth']:3d} | "
            f"threshold={threshold_text:>15s} | "
            f"shift from Moderate={shift_text}"
        )

    print("\n[4] Creating figures...")

    plot_curves(raw_rows, moderate_references)

    # The quick sweep may not bracket all thresholds. Only produce the bar
    # chart when every threshold is available.
    if all(row["target_bracketed"] for row in summary_rows):
        plot_threshold_summary(summary_rows)
    else:
        print(
            "Threshold figure skipped because the quick/local grid did not "
            "bracket every target. This is acceptable in QUICK mode."
        )

    print("\n" + "=" * 92)
    print("TASK 10 FINISHED")
    print("=" * 92)
    print(f"Raw CSV: {RAW_CSV_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Curves PNG: {CURVES_PNG_PATH}")

    if RUN_MODE.lower() == "quick":
        print(
            "\nQuick validation completed. If no errors occurred, change "
            "RUN_MODE to 'formal' and rerun."
        )
    else:
        print(
            "\nFormal sensitivity study completed. Inspect bracket flags, "
            "confidence intervals and threshold shifts before drawing conclusions."
        )


if __name__ == "__main__":
    main()
