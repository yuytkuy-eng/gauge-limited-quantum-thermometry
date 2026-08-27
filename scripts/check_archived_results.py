"""Fast integrity checks for the archived numerical code-and-data release."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def main() -> None:
    gauge = load("assignment_identifiability/summary.json")
    assert set(gauge["fisher_rank_by_length"].values()) == {3}
    assert gauge["gauge_test"]["maximum_record_probability_difference"] < 1e-12

    quartic = load("basis_symmetry_scaling/summary.json")
    exponents = [
        entry["exponent"]
        for entry in quartic["small_angle_fit"]["fits"].values()
    ]
    assert all(3.9 < exponent < 4.1 for exponent in exponents)

    quantum = load("quantum_output_benchmark/summary.json")
    close(
        quantum["length_8_quantum_fisher"]["effective_information_per_readout"],
        0.036143146212501,
        1e-10,
        "N=8 QFI/readout",
    )
    assert quantum["length_8_quantum_fisher"]["maximum_mean_sld_commutator"] < 1e-15

    posterior = load("four_dimensional_inference/summary.json")
    assert posterior["record_length"] == 8
    assert posterior["mle_replicates_per_case"] == 96
    internal = next(
        item for item in posterior["posterior_cases"] if item["protocol"] == "internal_basis"
    )
    external = next(
        item
        for item in posterior["posterior_cases"]
        if item["protocol"] == "external_references"
    )
    close(internal["posterior_sd"][0], 0.03773339081587728, 1e-10, "internal posterior SD(T)")
    close(external["posterior_sd"][0], 0.004007, 2e-4, "external posterior SD(T)")

    robust = load("robust_c_optimal_design/summary.json")
    assert robust["grid_points"] == 81
    design = robust["design_families"]["external_references"]
    weights = design["summaries"]["minimax_relative"]["weights"]
    close(weights["z_sensing"], 0.7968256206, 1e-8, "relative-minimax sensing weight")
    close(weights["ground_reference"], 0.1756060875, 1e-8, "ground-reference weight")
    close(weights["excited_reference"], 0.0275682919, 1e-8, "excited-reference weight")
    close(
        design["summaries"]["minimax_relative"]["minimum_relative_c_efficiency"],
        0.9416777005,
        1e-8,
        "minimum relative c-efficiency",
    )
    validation = robust["finite_sample_validation"]
    assert validation["datasets_per_family_and_design"] == 1944
    assert validation["no_exclusions"] is True

    required_outputs = {
        "assignment_identifiability/likelihood_gauge_ridge.png",
        "basis_symmetry_scaling/basis_quartic_scaling.png",
        "quantum_output_benchmark/quantum_information_ladder.png",
        "four_dimensional_inference/four_dimensional_posterior.pdf",
        "robust_c_optimal_design/robust_c_optimal_design.pdf",
        "robust_c_optimal_design/robust_design_finite_sample.pdf",
    }
    assert all((RESULTS / relative).is_file() for relative in required_outputs)

    print("Archived-result audit passed:")
    print("  exact gauge and Fisher rank")
    print("  quartic gauge lifting")
    print("  N=8 quantum-information benchmark")
    print("  four-dimensional posterior scales")
    print("  81-point relative-minimax allocation")
    print("  finite-sample validation metadata and archived generated outputs")


if __name__ == "__main__":
    main()
