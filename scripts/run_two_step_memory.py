"""Cross-model test of the detector-temperature gauge in a two-step memory."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.assignment_fisher import (
    assignment_calibration_fisher,
    effective_target_information,
)
from memory_thermometry.classical_full_swap import (
    temperature_from_excitation_probability,
    thermal_excitation_probability,
)
from memory_thermometry.identifiability import fit_power_law
from memory_thermometry.two_step_model import (
    TwoStepCollisionParameters,
    two_step_assignment_fisher_matrix,
    two_step_record_distribution,
)


OUTPUT = ROOT / "results" / "two_step_memory"
PARAMETERS = TwoStepCollisionParameters(
    temperature=0.9,
    memory_angle=0.5,
    system_memory_angle=0.55,
)
FALSE_POSITIVE = 0.02
FALSE_NEGATIVE = 0.04
LENGTH = 6
PHIS = np.geomspace(0.01, 0.25, 12)
FIT_POINTS = 7


def fisher(polar_angles: float | tuple[float, ...]) -> np.ndarray:
    return two_step_assignment_fisher_matrix(
        PARAMETERS,
        probe_angles=0.5 * np.pi,
        length=LENGTH,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
        measurement_polar_angles=polar_angles,
    ).matrix


def schedule_label(bits: tuple[int, ...]) -> str:
    return "".join("X" if bit else "Z" for bit in bits)


def binary_joint_moment(distribution: np.ndarray, mask: int) -> float:
    """Return E[prod_{i in mask} X_i] for a lexicographic binary law."""

    indices = np.arange(distribution.size)
    return float(distribution[(indices & mask) == mask].sum())


def scan_basis_design(z_fisher: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    coarse_weights = np.linspace(0.0, 1.0, 201)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for index in range(2**LENGTH):
        bits = tuple(int(bit) for bit in f"{index:0{LENGTH}b}")
        schedule = tuple(0.5 * np.pi if bit else 0.0 for bit in bits)
        candidate = fisher(schedule)
        pure_information = effective_target_information(candidate)
        coarse_information = np.asarray(
            [
                effective_target_information(
                    (1.0 - weight) * z_fisher + weight * candidate
                )
                for weight in coarse_weights
            ]
        )
        coarse_optimum = int(np.argmax(coarse_information))
        coarse_weight = float(coarse_weights[coarse_optimum])
        fine_weights = np.linspace(
            max(0.0, coarse_weight - 0.005),
            min(1.0, coarse_weight + 0.005),
            201,
        )
        fine_information = np.asarray(
            [
                effective_target_information(
                    (1.0 - weight) * z_fisher + weight * candidate
                )
                for weight in fine_weights
            ]
        )
        optimum = int(np.argmax(fine_information))
        weight = float(fine_weights[optimum])
        mixed = (1.0 - weight) * z_fisher + weight * candidate
        row = {
            "schedule": schedule_label(bits),
            "x_readouts": int(sum(bits)),
            "pure_effective_information_per_block": pure_information,
            "optimal_complementary_block_fraction": weight,
            "mixed_effective_information_per_block": float(
                fine_information[optimum]
            ),
            "mixed_minimum_eigenvalue": float(np.linalg.eigvalsh(mixed)[0]),
        }
        rows.append(row)
        if best is None or float(row["mixed_effective_information_per_block"]) > float(
            best["information"]
        ):
            best = {
                "schedule": row["schedule"],
                "weight": weight,
                "information": row["mixed_effective_information_per_block"],
                "pure_information": pure_information,
                "fisher": mixed,
            }
    assert best is not None
    return rows, best


def optimize_external_calibration(z_fisher: np.ndarray) -> dict[str, object]:
    ground = assignment_calibration_fisher(
        FALSE_POSITIVE, FALSE_NEGATIVE, prepared_state=0
    )
    excited = assignment_calibration_fisher(
        FALSE_POSITIVE, FALSE_NEGATIVE, prepared_state=1
    )
    fractions = np.linspace(0.0, 1.0, 401)
    best_information = -np.inf
    best = (0.0, 0.0, 0.0)
    best_fisher = np.zeros((4, 4))
    for sensing in fractions:
        for ground_fraction in fractions:
            excited_fraction = 1.0 - sensing - ground_fraction
            if excited_fraction < -1e-12:
                break
            total = (
                (sensing / LENGTH) * z_fisher
                + ground_fraction * ground
                + max(excited_fraction, 0.0) * excited
            )
            information = effective_target_information(total)
            if information > best_information:
                best_information = information
                best = (float(sensing), float(ground_fraction), float(max(excited_fraction, 0.0)))
                best_fisher = total
    return {
        "sensing_readout_fraction": best[0],
        "ground_reference_fraction": best[1],
        "excited_reference_fraction": best[2],
        "effective_information_per_readout": float(best_information),
        "eigenvalues": np.linalg.eigvalsh(best_fisher).tolist(),
        "grid_step": float(fractions[1] - fractions[0]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    z_fisher = fisher(0.0)

    excitation = thermal_excitation_probability(PARAMETERS.temperature)
    contrast = 1.0 - FALSE_POSITIVE - FALSE_NEGATIVE
    original = two_step_record_distribution(
        PARAMETERS,
        0.5 * np.pi,
        length=LENGTH + 1,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
    )
    gauge_rows: list[dict[str, float]] = []
    for scale in (1.01, 1.05, 1.10, 1.20):
        transformed_excitation = scale * excitation
        transformed_temperature = temperature_from_excitation_probability(
            transformed_excitation
        )
        transformed_beta = 1.0 - FALSE_POSITIVE - contrast / scale
        transformed = two_step_record_distribution(
            TwoStepCollisionParameters(
                temperature=transformed_temperature,
                memory_angle=PARAMETERS.memory_angle,
                system_memory_angle=PARAMETERS.system_memory_angle,
            ),
            0.5 * np.pi,
            length=LENGTH + 1,
            false_positive=FALSE_POSITIVE,
            false_negative=transformed_beta,
        )
        gauge_rows.append(
            {
                "excitation_scale": scale,
                "transformed_temperature": transformed_temperature,
                "transformed_false_negative": transformed_beta,
                "maximum_absolute_probability_difference": float(
                    np.max(np.abs(original - transformed))
                ),
            }
        )

    moment_temperatures = (0.7, 1.3)
    moment_laws = []
    moment_excitations = []
    for temperature in moment_temperatures:
        moment_laws.append(
            two_step_record_distribution(
                TwoStepCollisionParameters(
                    temperature=temperature,
                    memory_angle=PARAMETERS.memory_angle,
                    system_memory_angle=PARAMETERS.system_memory_angle,
                ),
                0.5 * np.pi,
                length=LENGTH,
            )
        )
        moment_excitations.append(thermal_excitation_probability(temperature))
    moment_rows: list[dict[str, object]] = []
    for mask in range(1, 2**LENGTH):
        order = mask.bit_count()
        normalized = [
            binary_joint_moment(law, mask) / excitation_value**order
            for law, excitation_value in zip(
                moment_laws, moment_excitations, strict=True
            )
        ]
        moment_rows.append(
            {
                "subset_mask": f"{mask:0{LENGTH}b}",
                "order": order,
                "normalized_moment_T0.7": normalized[0],
                "normalized_moment_T1.3": normalized[1],
                "absolute_difference": abs(normalized[0] - normalized[1]),
            }
        )

    basis_rows, best_basis = scan_basis_design(z_fisher)
    external = optimize_external_calibration(z_fisher)

    scaling_rows: list[dict[str, float]] = []
    for phi in PHIS:
        fixed = fisher(float(phi))
        alternating = fisher((0.0, float(phi)))
        mixed = 0.5 * z_fisher + 0.5 * fixed
        scaling_rows.append(
            {
                "basis_angle": float(phi),
                "fixed_effective_information": effective_target_information(fixed),
                "alternating_effective_information": effective_target_information(
                    alternating
                ),
                "block_mixture_effective_information": effective_target_information(
                    mixed
                ),
                "fixed_minimum_eigenvalue": float(np.linalg.eigvalsh(fixed)[0]),
                "alternating_minimum_eigenvalue": float(
                    np.linalg.eigvalsh(alternating)[0]
                ),
                "block_mixture_minimum_eigenvalue": float(
                    np.linalg.eigvalsh(mixed)[0]
                ),
            }
        )
    fit_fields = {
        "fixed_effective_information": "fixed",
        "alternating_effective_information": "alternating",
        "block_mixture_effective_information": "block_mixture",
        "fixed_minimum_eigenvalue": "fixed_minimum_eigenvalue",
        "alternating_minimum_eigenvalue": "alternating_minimum_eigenvalue",
        "block_mixture_minimum_eigenvalue": "block_mixture_minimum_eigenvalue",
    }
    fits: dict[str, dict[str, float]] = {}
    controls = np.asarray(
        [row["basis_angle"] for row in scaling_rows[:FIT_POINTS]]
    )
    for field, label in fit_fields.items():
        fit = fit_power_law(
            controls,
            [row[field] for row in scaling_rows[:FIT_POINTS]],
        )
        fits[label] = {
            "exponent": fit.exponent,
            "coefficient": fit.coefficient,
            "r_squared": fit.r_squared,
        }

    best_fisher = np.asarray(best_basis["fisher"])
    internal_per_readout = float(best_basis["information"]) / LENGTH
    external_per_readout = float(external["effective_information_per_readout"])
    known_detector_per_readout = (
        effective_target_information(z_fisher[:2, :2]) / LENGTH
    )
    one_step_summary = json.loads(
        (ROOT / "results" / "basis_symmetry_scaling" / "summary.json").read_text()
    )
    summary = {
        "parameters": {
            "temperature": PARAMETERS.temperature,
            "memory_angle": PARAMETERS.memory_angle,
            "system_memory_angle": PARAMETERS.system_memory_angle,
            "false_positive": FALSE_POSITIVE,
            "false_negative": FALSE_NEGATIVE,
            "block_length": LENGTH,
        },
        "finite_gauge_test": {
            "invariant": "kappa=(1-alpha-beta)e(T)",
            "record_length": LENGTH + 1,
            "transformations": gauge_rows,
            "maximum_absolute_probability_difference": max(
                row["maximum_absolute_probability_difference"]
                for row in gauge_rows
            ),
        },
        "fermionic_moment_test": {
            "temperatures": list(moment_temperatures),
            "subsets_tested": len(moment_rows),
            "identity": "E[prod_{i in A} X_i] / e(T)^|A| is temperature independent",
            "maximum_absolute_normalized_moment_difference": max(
                float(row["absolute_difference"]) for row in moment_rows
            ),
        },
        "all_z_fisher": {
            "rank": int(
                np.count_nonzero(
                    np.linalg.eigvalsh(z_fisher)
                    > 1e-10 * np.linalg.eigvalsh(z_fisher)[-1]
                )
            ),
            "eigenvalues": np.linalg.eigvalsh(z_fisher).tolist(),
            "effective_temperature_information": effective_target_information(
                z_fisher
            ),
            "known_detector_information_per_readout": known_detector_per_readout,
        },
        "optimal_basis_control": {
            "schedules_tested": 2**LENGTH,
            "mixture_coarse_step": 0.005,
            "mixture_fine_step": 0.00005,
            "complementary_schedule": best_basis["schedule"],
            "complementary_block_fraction": best_basis["weight"],
            "all_z_block_fraction": 1.0 - float(best_basis["weight"]),
            "pure_complementary_information_per_block": best_basis[
                "pure_information"
            ],
            "effective_information_per_block": best_basis["information"],
            "effective_information_per_readout": internal_per_readout,
            "eigenvalues": np.linalg.eigvalsh(best_fisher).tolist(),
        },
        "external_calibration": external,
        "fair_readout_comparison": {
            "known_detector_information_per_readout": known_detector_per_readout,
            "internal_basis_information_per_readout": internal_per_readout,
            "external_reference_information_per_readout": external_per_readout,
            "external_to_internal_information_ratio": (
                external_per_readout / internal_per_readout
            ),
            "external_to_internal_sd_ratio": float(
                np.sqrt(external_per_readout / internal_per_readout)
            ),
        },
        "small_basis_scaling": {
            "fit_points": FIT_POINTS,
            "angle_interval": [float(controls[0]), float(controls[-1])],
            "fits": fits,
            "one_step_block_mixture_exponent": one_step_summary[
                "small_angle_fit"
            ]["fits"]["block_mixture"]["exponent"],
        },
        "interpretation": (
            "The exact detector-temperature gauge and quartic basis-control "
            "lifting survive a physically distinct two-step memory. The "
            "temperature scaling of every ideal occupation moment supplies a "
            "direct numerical check of the passive-fermionic gauge theorem."
        ),
    }

    for filename, rows in (
        ("gauge_transformations.csv", gauge_rows),
        ("fermionic_moments.csv", moment_rows),
        ("basis_schedule_scan.csv", basis_rows),
        ("basis_scaling.csv", scaling_rows),
    ):
        with (OUTPUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.25))
    for key, label, color in (
        ("fixed", "fixed tilted basis", "#756bb1"),
        ("alternating", "Z/tilted within block", "#2b8cbe"),
        ("block_mixture", "50% Z + 50% tilted blocks", "#e34a33"),
    ):
        axes[0].loglog(
            PHIS,
            [row[f"{key}_effective_information"] for row in scaling_rows],
            "o-",
            ms=3.5,
            color=color,
            label=label,
        )
    guide = scaling_rows[0]["block_mixture_effective_information"] * (
        PHIS / PHIS[0]
    ) ** 4
    axes[0].loglog(PHIS, guide, "k--", lw=1.2, label=r"$\phi^4$")
    axes[0].set_xlabel(r"basis rotation angle $\phi$")
    axes[0].set_ylabel("effective temperature information")
    axes[0].set_title("Quartic gauge lifting survives deeper memory")
    axes[0].grid(which="both", alpha=0.22)
    axes[0].legend(fontsize=8)

    labels = ["known\ndetector", "internal\nbasis", "external\nreferences"]
    values = [known_detector_per_readout, internal_per_readout, external_per_readout]
    bars = axes[1].bar(labels, values, color=["#756bb1", "#2b8cbe", "#e34a33"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("effective information per binary readout")
    axes[1].set_title("Explicit references remain resource efficient")
    axes[1].grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + 0.5 * bar.get_width(),
            value * 1.12,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(OUTPUT / "two_step_cross_model.png", dpi=220)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
