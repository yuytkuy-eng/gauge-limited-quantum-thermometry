"""Bayes and minimax c-optimal resource allocation over a four-dimensional grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.inference import (
    BlockExperiment,
    CalibrationExperiment,
    calibration_probability_jacobian,
    fisher_scoring_mle,
    quantum_probability_jacobian,
)
from memory_thermometry.model import CollisionParameters
from memory_thermometry.robust_design import (
    c_optimal_variance,
    relative_c_efficiency,
    tensor_product_weights,
)


OUTPUT = ROOT / "results" / "robust_c_optimal_design"
LENGTH = 8
TOTAL_READOUTS = 8_000_000
SYSTEM_MEMORY_ANGLE = 0.55
BOUNDS = np.asarray(
    [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
)
GRID_AXES = {
    "temperature": np.asarray([0.6, 0.9, 1.3]),
    "memory_angle": np.asarray([0.2, 0.5, 0.9]),
    "false_positive": np.asarray([0.005, 0.02, 0.06]),
    "false_negative": np.asarray([0.01, 0.04, 0.12]),
}
COMPONENT_NAMES = ("z_sensing", "x_sensing", "ground_reference", "excited_reference")
FAMILIES = {
    "internal_basis": (0, 1),
    "external_references": (0, 2, 3),
    "hybrid": (0, 1, 2, 3),
}
DESIGN_NAMES = (
    "center_local",
    "bayes_equal_grid",
    "bayes_uniform_volume",
    "minimax_absolute",
    "minimax_relative",
)
VALIDATED_DESIGNS = (
    "center_local",
    "bayes_uniform_volume",
    "minimax_absolute",
    "minimax_relative",
)
VALIDATION_FAMILIES = ("internal_basis", "external_references")
VALIDATION_REPLICATES = 24
SIMPLEX_EPSILON = 1e-8

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def grid_points() -> np.ndarray:
    return np.asarray(
        list(
            itertools.product(
                GRID_AXES["temperature"],
                GRID_AXES["memory_angle"],
                GRID_AXES["false_positive"],
                GRID_AXES["false_negative"],
            )
        ),
        dtype=float,
    )


def information(probability: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    return (jacobian / probability) @ jacobian.T


def point_components(
    vector: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return component probabilities, Jacobians, and per-readout information."""

    template = CollisionParameters(
        temperature=float(vector[0]),
        memory_angle=float(vector[1]),
        system_memory_angle=SYSTEM_MEMORY_ANGLE,
    )
    probabilities: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    rates: list[np.ndarray] = []
    for polar in (0.0, 0.5 * np.pi):
        probability, jacobian = quantum_probability_jacobian(
            vector,
            template,
            0.5 * np.pi,
            LENGTH,
            measurement_polar_angles=polar,
        )
        probabilities.append(probability)
        jacobians.append(jacobian)
        rates.append(information(probability, jacobian) / LENGTH)
    for prepared_state in (0, 1):
        probability, jacobian = calibration_probability_jacobian(
            vector, prepared_state
        )
        probabilities.append(probability)
        jacobians.append(jacobian)
        rates.append(information(probability, jacobian))
    return probabilities, jacobians, np.asarray(rates)


def component_information_grid(points: np.ndarray) -> np.ndarray:
    return np.asarray([point_components(point)[2] for point in points])


def simplex_starts(setting_count: int) -> list[np.ndarray]:
    starts = [np.full(setting_count, 1.0 / setting_count)]
    for index in range(setting_count):
        point = np.full(setting_count, 0.15 / max(setting_count - 1, 1))
        point[index] = 0.85
        starts.append(point)
    rng = np.random.default_rng(620001 + setting_count)
    starts.extend(rng.dirichlet(np.ones(setting_count), size=8))
    return starts


def optimize_simplex_objective(
    objective,
    setting_count: int,
    extra_starts: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float, bool, str]:
    """Multi-start SLSQP for a convex simplex objective with scale control."""

    starts = simplex_starts(setting_count)
    if extra_starts:
        starts = [*extra_starts, *starts]
    scale = max(float(objective(starts[0])), 1e-12)
    best = None
    for start in starts:
        result = minimize(
            lambda weights: float(objective(weights)) / scale,
            start,
            method="SLSQP",
            bounds=[(SIMPLEX_EPSILON, 1.0)] * setting_count,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"ftol": 1e-13, "maxiter": 2500},
        )
        candidate = np.maximum(result.x, SIMPLEX_EPSILON)
        candidate /= candidate.sum()
        value = float(objective(candidate))
        if best is None or value < best[1]:
            best = (candidate, value, bool(result.success), str(result.message))
    assert best is not None
    return best


def optimize_two_setting_objective(
    objective,
) -> tuple[np.ndarray, float, bool, str]:
    """Globally optimize a convex two-setting mixture on its scalar interval."""

    result = minimize_scalar(
        lambda first: float(
            objective(np.asarray([first, 1.0 - first]))
        ),
        bounds=(SIMPLEX_EPSILON, 1.0 - SIMPLEX_EPSILON),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 2000},
    )
    weights = np.asarray([result.x, 1.0 - result.x])
    return weights, float(result.fun), bool(result.success), str(result.message)


def optimize_point_oracles(component_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimize every grid point independently for the oracle denominator."""

    point_count, setting_count = component_grid.shape[:2]
    weights = np.empty((point_count, setting_count))
    variances = np.empty(point_count)
    if setting_count == 2:
        for point in range(point_count):
            matrices = component_grid[point]
            result = minimize_scalar(
                lambda first: float(
                    c_optimal_variance(
                        matrices, np.asarray([first, 1.0 - first])
                    )
                ),
                bounds=(SIMPLEX_EPSILON, 1.0 - SIMPLEX_EPSILON),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 1000},
            )
            weights[point] = [result.x, 1.0 - result.x]
            variances[point] = result.fun
        return weights, variances
    for point in range(point_count):
        result = optimize_simplex_objective(
            lambda candidate, matrices=component_grid[point]: float(
                c_optimal_variance(matrices, candidate)
            ),
            setting_count,
        )
        weights[point] = result[0]
        variances[point] = result[1]
    return weights, variances


def optimize_minimax(
    component_grid: np.ndarray,
    denominator: np.ndarray,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, bool, str]:
    """Solve the convex epigraph form of absolute or relative minimax risk."""

    setting_count = component_grid.shape[1]
    denominator = np.asarray(denominator, dtype=float)
    if setting_count == 2:
        return optimize_two_setting_objective(
            lambda weights: float(
                np.max(
                    c_optimal_variance(component_grid, weights)
                    / denominator
                )
            )
        )
    best = None
    def normalized(candidate: np.ndarray) -> np.ndarray:
        weights = np.maximum(candidate, SIMPLEX_EPSILON)
        return weights / weights.sum()

    for start in [*starts, *simplex_starts(setting_count)]:
        start_risk = c_optimal_variance(component_grid, start) / denominator
        initial = np.concatenate((start, [float(np.max(start_risk))]))

        def constraints(candidate: np.ndarray) -> np.ndarray:
            risk = (
                c_optimal_variance(
                    component_grid, normalized(candidate[:-1])
                )
                / denominator
            )
            return candidate[-1] - risk

        result = minimize(
            lambda candidate: float(candidate[-1]),
            initial,
            method="SLSQP",
            bounds=[(SIMPLEX_EPSILON, 1.0)] * setting_count + [(0.0, None)],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda candidate: candidate[:-1].sum() - 1.0,
                },
                {"type": "ineq", "fun": constraints},
            ],
            options={"ftol": 1e-12, "maxiter": 4000},
        )
        weights = normalized(result.x[:-1])
        actual = float(
            np.max(c_optimal_variance(component_grid, weights) / denominator)
        )
        if best is None or actual < best[1]:
            best = (weights, actual, bool(result.success), str(result.message))
    assert best is not None
    return best


def random_search_audit(
    component_grid: np.ndarray,
    designs: dict[str, np.ndarray],
    oracle_variance: np.ndarray,
    equal_weights: np.ndarray,
    volume_weights: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    """Audit optimized objectives against 50,000 deterministic simplex draws."""

    rng = np.random.default_rng(seed)
    candidates = rng.dirichlet(
        np.full(component_grid.shape[1], 0.65), size=50_000
    )
    objectives = {
        "bayes_equal_grid": lambda risk: float(equal_weights @ risk),
        "bayes_uniform_volume": lambda risk: float(volume_weights @ risk),
        "minimax_absolute": lambda risk: float(np.max(risk)),
        "minimax_relative": lambda risk: float(np.max(risk / oracle_variance)),
    }
    result: dict[str, float | int] = {"random_candidates": len(candidates)}
    random_best = {name: float("inf") for name in objectives}
    for start in range(0, len(candidates), 500):
        batch = candidates[start : start + 500]
        information = np.einsum("bk,gkij->bgij", batch, component_grid)
        risk = np.linalg.inv(information)[:, :, 0, 0]
        values = {
            "bayes_equal_grid": risk @ equal_weights,
            "bayes_uniform_volume": risk @ volume_weights,
            "minimax_absolute": np.max(risk, axis=1),
            "minimax_relative": np.max(
                risk / oracle_variance[None, :], axis=1
            ),
        }
        for name, value in values.items():
            random_best[name] = min(random_best[name], float(np.min(value)))
    for name, objective in objectives.items():
        optimized_risk = c_optimal_variance(component_grid, designs[name])
        optimized_value = objective(optimized_risk)
        result[f"{name}_optimized_objective"] = optimized_value
        result[f"{name}_best_random_objective"] = random_best[name]
        result[f"{name}_optimized_to_random_ratio"] = (
            optimized_value / random_best[name]
        )
    return result


def optimize_family(
    family: str,
    full_component_grid: np.ndarray,
    points: np.ndarray,
    equal_prior: np.ndarray,
    volume_prior: np.ndarray,
    external_reference: dict[str, object] | None = None,
) -> dict[str, object]:
    indices = FAMILIES[family]
    component_grid = full_component_grid[:, indices]
    setting_count = len(indices)
    oracle_weights, oracle_variance = optimize_point_oracles(component_grid)
    embedded_external: dict[str, np.ndarray] = {}
    if family == "hybrid" and external_reference is not None:
        for name, external_weights in external_reference["designs"].items():
            embedded = np.asarray(
                [external_weights[0], SIMPLEX_EPSILON, *external_weights[1:]],
                dtype=float,
            )
            embedded /= embedded.sum()
            embedded_external[name] = embedded
        for point in range(points.shape[0]):
            external_oracle = np.asarray(
                [
                    external_reference["oracle_weights"][point, 0],
                    SIMPLEX_EPSILON,
                    external_reference["oracle_weights"][point, 1],
                    external_reference["oracle_weights"][point, 2],
                ]
            )
            external_oracle /= external_oracle.sum()
            candidate_variance = float(
                c_optimal_variance(component_grid[point], external_oracle)
            )
            if candidate_variance < oracle_variance[point]:
                oracle_variance[point] = candidate_variance
                oracle_weights[point] = external_oracle
    center_index = int(
        np.flatnonzero(
            np.all(np.isclose(points, [0.9, 0.5, 0.02, 0.04]), axis=1)
        )[0]
    )
    optimizer = (
        optimize_two_setting_objective
        if setting_count == 2
        else lambda objective: optimize_simplex_objective(
            objective, setting_count
        )
    )
    center = optimizer(
        lambda weights: float(
            c_optimal_variance(component_grid[center_index], weights)
        )
    )
    bayes_equal = optimizer(
        lambda weights: float(
            equal_prior @ c_optimal_variance(component_grid, weights)
        )
    )
    bayes_volume = optimizer(
        lambda weights: float(
            volume_prior @ c_optimal_variance(component_grid, weights)
        )
    )
    if embedded_external:
        objectives = {
            "center_local": lambda weights: float(
                c_optimal_variance(component_grid[center_index], weights)
            ),
            "bayes_equal_grid": lambda weights: float(
                equal_prior @ c_optimal_variance(component_grid, weights)
            ),
            "bayes_uniform_volume": lambda weights: float(
                volume_prior @ c_optimal_variance(component_grid, weights)
            ),
        }
        current = {
            "center_local": center,
            "bayes_equal_grid": bayes_equal,
            "bayes_uniform_volume": bayes_volume,
        }
        for name, candidate in embedded_external.items():
            if name not in current:
                continue
            value = objectives[name](candidate)
            if value < current[name][1]:
                current[name] = (
                    candidate,
                    value,
                    True,
                    "embedded external-face certificate",
                )
        center = current["center_local"]
        bayes_equal = current["bayes_equal_grid"]
        bayes_volume = current["bayes_uniform_volume"]
    center_worst = float(
        np.max(c_optimal_variance(component_grid, center[0]))
    )
    minimax_absolute = optimize_minimax(
        component_grid,
        np.full(points.shape[0], center_worst),
        [center[0], bayes_equal[0], bayes_volume[0]],
    )
    minimax_relative = optimize_minimax(
        component_grid,
        oracle_variance,
        [
            center[0],
            bayes_equal[0],
            bayes_volume[0],
            minimax_absolute[0],
            *(
                [embedded_external["minimax_relative"]]
                if embedded_external
                else []
            ),
        ],
    )
    if embedded_external:
        embedded_absolute = embedded_external["minimax_absolute"]
        embedded_absolute_value = float(
            np.max(c_optimal_variance(component_grid, embedded_absolute))
            / center_worst
        )
        if embedded_absolute_value < minimax_absolute[1]:
            minimax_absolute = (
                embedded_absolute,
                embedded_absolute_value,
                True,
                "embedded external-face certificate",
            )
        embedded_relative = embedded_external["minimax_relative"]
        embedded_relative_value = float(
            np.max(
                c_optimal_variance(component_grid, embedded_relative)
                / oracle_variance
            )
        )
        if embedded_relative_value < minimax_relative[1]:
            minimax_relative = (
                embedded_relative,
                embedded_relative_value,
                True,
                "embedded external-face certificate",
            )
    designs = {
        "center_local": center[0],
        "bayes_equal_grid": bayes_equal[0],
        "bayes_uniform_volume": bayes_volume[0],
        "minimax_absolute": minimax_absolute[0],
        "minimax_relative": minimax_relative[0],
    }
    optimizer_records = {
        "center_local": center[2:],
        "bayes_equal_grid": bayes_equal[2:],
        "bayes_uniform_volume": bayes_volume[2:],
        "minimax_absolute": minimax_absolute[2:],
        "minimax_relative": minimax_relative[2:],
    }
    summaries = {}
    for name, weights in designs.items():
        variance = c_optimal_variance(component_grid, weights)
        efficiency = relative_c_efficiency(variance, oracle_variance)
        summaries[name] = {
            "weights": {
                COMPONENT_NAMES[index]: float(weight)
                for index, weight in zip(indices, weights, strict=True)
            },
            "bayes_equal_grid_variance_per_readout": float(equal_prior @ variance),
            "bayes_uniform_volume_variance_per_readout": float(volume_prior @ variance),
            "maximum_variance_per_readout": float(np.max(variance)),
            "minimum_relative_c_efficiency": float(np.min(efficiency)),
            "mean_relative_c_efficiency": float(equal_prior @ efficiency),
            "volume_weighted_relative_c_efficiency": float(volume_prior @ efficiency),
            "relative_efficiency_quantiles": np.quantile(
                efficiency, [0.0, 0.1, 0.5, 0.9, 1.0]
            ).tolist(),
            "maximum_oracle_sd_inflation": float(
                np.sqrt(np.max(variance / oracle_variance))
            ),
            "worst_predicted_temperature_sd_at_8e6": float(
                np.sqrt(np.max(variance) / TOTAL_READOUTS)
            ),
            "optimizer_success": bool(optimizer_records[name][0]),
            "optimizer_message": str(optimizer_records[name][1]),
        }
    return {
        "family": family,
        "component_indices": list(indices),
        "component_names": [COMPONENT_NAMES[index] for index in indices],
        "oracle_weights": oracle_weights,
        "oracle_variance": oracle_variance,
        "designs": designs,
        "summaries": summaries,
        "random_search_audit": random_search_audit(
            component_grid,
            designs,
            oracle_variance,
            equal_prior,
            volume_prior,
            seed=630000 + list(FAMILIES).index(family),
        ),
    }


def write_optimization_outputs(
    results: dict[str, dict[str, object]],
    points: np.ndarray,
    equal_prior: np.ndarray,
    volume_prior: np.ndarray,
) -> None:
    rows: list[dict[str, float | str]] = []
    for family, result in results.items():
        oracle = np.asarray(result["oracle_variance"])
        component_indices = result["component_indices"]
        component_grid = component_information_grid(points)[:, component_indices]
        for design_name, weights in result["designs"].items():
            variance = c_optimal_variance(component_grid, weights)
            efficiency = relative_c_efficiency(variance, oracle)
            for point_index, point in enumerate(points):
                rows.append(
                    {
                        "family": family,
                        "design": design_name,
                        "grid_index": point_index,
                        "temperature": float(point[0]),
                        "memory_angle": float(point[1]),
                        "false_positive": float(point[2]),
                        "false_negative": float(point[3]),
                        "equal_grid_prior_weight": float(equal_prior[point_index]),
                        "uniform_volume_prior_weight": float(volume_prior[point_index]),
                        "temperature_variance_per_readout": float(variance[point_index]),
                        "temperature_sd_at_8e6": float(
                            np.sqrt(variance[point_index] / TOTAL_READOUTS)
                        ),
                        "oracle_temperature_variance_per_readout": float(
                            oracle[point_index]
                        ),
                        "relative_c_efficiency": float(efficiency[point_index]),
                        "oracle_sd_inflation": float(
                            1.0 / np.sqrt(efficiency[point_index])
                        ),
                    }
                )
    with (OUTPUT / "risk_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    serializable = {
        "grid_axes": {name: values.tolist() for name, values in GRID_AXES.items()},
        "grid_points": points.tolist(),
        "equal_grid_prior": equal_prior.tolist(),
        "uniform_volume_quadrature_prior": volume_prior.tolist(),
        "record_length": LENGTH,
        "total_binary_readouts_for_reported_sd": TOTAL_READOUTS,
        "criterion": (
            "c^T M(theta,w)^-1 c per binary readout with c=(1,0,0,0)"
        ),
        "families": {},
    }
    for family, result in results.items():
        serializable["families"][family] = {
            "component_indices": result["component_indices"],
            "component_names": result["component_names"],
            "oracle_weights": np.asarray(result["oracle_weights"]).tolist(),
            "oracle_variance": np.asarray(result["oracle_variance"]).tolist(),
            "designs": {
                name: np.asarray(weights).tolist()
                for name, weights in result["designs"].items()
            },
            "summaries": result["summaries"],
            "random_search_audit": result["random_search_audit"],
        }
    (OUTPUT / "designs.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def run_optimization() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points = grid_points()
    full_grid = component_information_grid(points)
    equal_prior = np.full(points.shape[0], 1.0 / points.shape[0])
    volume_prior = tensor_product_weights(*GRID_AXES.values())
    results = {}
    for family in FAMILIES:
        results[family] = optimize_family(
            family,
            full_grid,
            points,
            equal_prior,
            volume_prior,
            results.get("external_references"),
        )
        print(f"optimized {family}", flush=True)
    write_optimization_outputs(results, points, equal_prior, volume_prior)


def allocate_readouts(
    family: str, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return full component counts and realized readout fractions exactly."""

    indices = FAMILIES[family]
    requested = np.zeros(4)
    requested[list(indices)] = weights
    counts = np.zeros(4, dtype=int)
    counts[0] = int(round(requested[0] * TOTAL_READOUTS / LENGTH))
    counts[1] = int(round(requested[1] * TOTAL_READOUTS / LENGTH))
    counts[2] = int(round(requested[2] * TOTAL_READOUTS))
    used = LENGTH * (counts[0] + counts[1]) + counts[2]
    counts[3] = TOTAL_READOUTS - used
    if np.any(counts[list(indices)] < 1) or np.any(counts < 0):
        raise ValueError("allocation produced a nonpositive active component")
    realized = np.asarray(
        [
            LENGTH * counts[0] / TOTAL_READOUTS,
            LENGTH * counts[1] / TOTAL_READOUTS,
            counts[2] / TOTAL_READOUTS,
            counts[3] / TOTAL_READOUTS,
        ]
    )
    return counts, realized


def experiments_for_components(
    counts: list[np.ndarray], active: tuple[int, ...]
) -> list[BlockExperiment | CalibrationExperiment]:
    experiments: list[BlockExperiment | CalibrationExperiment] = []
    for observed, component in zip(counts, active, strict=True):
        if component == 0:
            experiments.append(BlockExperiment(0.5 * np.pi, observed, 0.0))
        elif component == 1:
            experiments.append(
                BlockExperiment(0.5 * np.pi, observed, 0.5 * np.pi)
            )
        elif component == 2:
            experiments.append(CalibrationExperiment(0, observed))
        elif component == 3:
            experiments.append(CalibrationExperiment(1, observed))
        else:
            raise ValueError("unknown component")
    return experiments


def run_validation_cell(
    grid_index: int,
    point: np.ndarray,
    family: str,
    design_name: str,
    weights: np.ndarray,
) -> dict[str, object]:
    probabilities, jacobians, _ = point_components(point)
    active = FAMILIES[family]
    component_counts, realized = allocate_readouts(family, weights)
    counts_per_component = component_counts[list(active)]
    fisher = np.zeros((4, 4))
    for component, count in zip(active, counts_per_component, strict=True):
        fisher += count * information(probabilities[component], jacobians[component])
    covariance = np.linalg.inv(fisher)
    predicted_sd = float(np.sqrt(covariance[0, 0]))
    template = CollisionParameters(
        temperature=float(point[0]),
        memory_angle=float(point[1]),
        system_memory_angle=SYSTEM_MEMORY_ANGLE,
    )
    estimates = np.empty((VALIDATION_REPLICATES, 4))
    reported_se = np.empty(VALIDATION_REPLICATES)
    converged = np.zeros(VALIDATION_REPLICATES, dtype=bool)
    boundary = np.zeros((VALIDATION_REPLICATES, 4), dtype=bool)
    family_index = VALIDATION_FAMILIES.index(family)
    design_index = VALIDATED_DESIGNS.index(design_name)
    rng = np.random.default_rng(
        710000 + 10000 * family_index + 1000 * design_index + grid_index
    )
    for replicate in range(VALIDATION_REPLICATES):
        observed = [
            rng.multinomial(int(count), probabilities[component])
            for component, count in zip(active, counts_per_component, strict=True)
        ]
        score = np.zeros(4)
        for component, count, values in zip(
            active, counts_per_component, observed, strict=True
        ):
            probability = probabilities[component]
            jacobian = jacobians[component]
            score += jacobian @ ((values - count * probability) / probability)
        initial = np.clip(
            point + covariance @ score, BOUNDS[:, 0], BOUNDS[:, 1]
        )
        if initial[2] + initial[3] >= 0.95:
            initial = point.copy()
        fit = fisher_scoring_mle(
            experiments_for_components(observed, active),
            template,
            LENGTH,
            initial,
            bounds=BOUNDS,
            max_iterations=24,
        )
        estimates[replicate] = fit.estimate
        reported_se[replicate] = np.sqrt(
            max(float(fit.covariance[0, 0]), 0.0)
        )
        converged[replicate] = fit.converged
        boundary[replicate] = (
            (fit.estimate - BOUNDS[:, 0] < 2e-4)
            | (BOUNDS[:, 1] - fit.estimate < 2e-4)
        )
    errors = estimates[:, 0] - point[0]
    covered = np.abs(errors) <= 1.96 * reported_se
    return {
        "grid_index": grid_index,
        "family": family,
        "design": design_name,
        "temperature": float(point[0]),
        "memory_angle": float(point[1]),
        "false_positive": float(point[2]),
        "false_negative": float(point[3]),
        "replicates": VALIDATION_REPLICATES,
        "requested_weights": weights.tolist(),
        "realized_full_component_readout_fractions": realized.tolist(),
        "component_counts": component_counts.tolist(),
        "predicted_temperature_sd": predicted_sd,
        "empirical_temperature_sd": float(np.std(estimates[:, 0], ddof=1)),
        "temperature_bias": float(np.mean(errors)),
        "temperature_rmse": float(np.sqrt(np.mean(errors**2))),
        "standardized_temperature_rmse": float(
            np.sqrt(np.mean((errors / predicted_sd) ** 2))
        ),
        "wald_coverage_successes": int(np.count_nonzero(covered)),
        "wald_95_coverage": float(np.mean(covered)),
        "converged_successes": int(np.count_nonzero(converged)),
        "converged_fraction": float(np.mean(converged)),
        "temperature_boundary_fraction": float(np.mean(boundary[:, 0])),
        "assignment_boundary_fraction": float(
            np.mean(boundary[:, 2] | boundary[:, 3])
        ),
        "temperature_estimates": estimates[:, 0].tolist(),
    }


def run_validation_shard(start: int, stop: int) -> None:
    design_path = OUTPUT / "designs.json"
    if not design_path.exists():
        raise RuntimeError("run --optimize before validation shards")
    design_data = json.loads(design_path.read_text(encoding="utf-8"))
    points = grid_points()
    if start < 0 or stop > len(points) or start >= stop:
        raise ValueError("shard must satisfy 0 <= start < stop <= 81")
    rows = []
    for grid_index in range(start, stop):
        for family in VALIDATION_FAMILIES:
            for design_name in VALIDATED_DESIGNS:
                weights = np.asarray(
                    design_data["families"][family]["designs"][design_name]
                )
                rows.append(
                    run_validation_cell(
                        grid_index,
                        points[grid_index],
                        family,
                        design_name,
                        weights,
                    )
                )
        print(f"completed validation grid point {grid_index + 1}/81", flush=True)
    (OUTPUT / f"validation_shard_{start:02d}_{stop:02d}.json").write_text(
        json.dumps({"start": start, "stop": stop, "rows": rows}),
        encoding="utf-8",
    )


def binomial_wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (fraction + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def load_validation_shards() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(OUTPUT.glob("validation_shard_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    keys = {
        (int(row["grid_index"]), row["family"], row["design"])
        for row in rows
    }
    expected = {
        (point, family, design)
        for point in range(81)
        for family in VALIDATION_FAMILIES
        for design in VALIDATED_DESIGNS
    }
    if keys != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"validation shards incomplete: {len(expected - keys)} missing, "
            f"{len(rows) - len(keys)} duplicates"
        )
    rows.sort(key=lambda row: (int(row["grid_index"]), row["family"], row["design"]))
    return rows


def pooled_validation_summary(
    rows: list[dict[str, object]], family: str, design: str
) -> dict[str, object]:
    selected = sorted(
        (
            row
            for row in rows
            if row["family"] == family and row["design"] == design
        ),
        key=lambda row: int(row["grid_index"]),
    )
    if [int(row["grid_index"]) for row in selected] != list(range(81)):
        raise RuntimeError(f"incomplete validation grid for {family}/{design}")
    trials = sum(int(row["replicates"]) for row in selected)
    coverage_successes = sum(int(row["wald_coverage_successes"]) for row in selected)
    convergence_successes = sum(int(row["converged_successes"]) for row in selected)
    standardized_squared_errors = []
    raw_errors = []
    for row in selected:
        truth = float(row["temperature"])
        predicted_sd = float(row["predicted_temperature_sd"])
        estimates = np.asarray(row["temperature_estimates"], dtype=float)
        raw_errors.extend((estimates - truth).tolist())
        standardized_squared_errors.extend(((estimates - truth) / predicted_sd) ** 2)
    low_memory = [row for row in selected if np.isclose(float(row["memory_angle"]), 0.2)]
    low_trials = sum(int(row["replicates"]) for row in low_memory)
    low_converged = sum(int(row["converged_successes"]) for row in low_memory)
    cell_empirical_mse = np.asarray(
        [
            np.mean(
                (
                    np.asarray(row["temperature_estimates"], dtype=float)
                    - float(row["temperature"])
                )
                ** 2
            )
            for row in selected
        ]
    )
    cell_predicted_variance = np.asarray(
        [float(row["predicted_temperature_sd"]) ** 2 for row in selected]
    )
    equal_grid_prior = np.full(81, 1.0 / 81.0)
    volume_prior = tensor_product_weights(*GRID_AXES.values())
    return {
        "family": family,
        "design": design,
        "grid_cells": len(selected),
        "datasets": trials,
        "pooled_wald_95_coverage": coverage_successes / trials,
        "pooled_wald_coverage_wilson_95_interval": list(
            binomial_wilson_interval(coverage_successes, trials)
        ),
        "pooled_converged_fraction": convergence_successes / trials,
        "pooled_convergence_wilson_95_interval": list(
            binomial_wilson_interval(convergence_successes, trials)
        ),
        "low_memory_converged_fraction": low_converged / low_trials,
        "low_memory_convergence_wilson_95_interval": list(
            binomial_wilson_interval(low_converged, low_trials)
        ),
        "pooled_standardized_rmse": float(
            np.sqrt(np.mean(standardized_squared_errors))
        ),
        "pooled_raw_temperature_bias": float(np.mean(raw_errors)),
        "equal_grid_empirical_temperature_rmse": float(
            np.sqrt(equal_grid_prior @ cell_empirical_mse)
        ),
        "equal_grid_fisher_predicted_temperature_rmse": float(
            np.sqrt(equal_grid_prior @ cell_predicted_variance)
        ),
        "uniform_volume_empirical_temperature_rmse": float(
            np.sqrt(volume_prior @ cell_empirical_mse)
        ),
        "uniform_volume_fisher_predicted_temperature_rmse": float(
            np.sqrt(volume_prior @ cell_predicted_variance)
        ),
        "worst_grid_cell_empirical_temperature_rmse": float(
            np.sqrt(np.max(cell_empirical_mse))
        ),
        "mean_assignment_boundary_fraction": float(
            np.mean([float(row["assignment_boundary_fraction"]) for row in selected])
        ),
    }


def risk_figure(design_data: dict[str, object]) -> None:
    risk_rows = list(
        csv.DictReader((OUTPUT / "risk_grid.csv").open(encoding="utf-8"))
    )
    display_designs = list(VALIDATED_DESIGNS)
    design_labels = {
        "center_local": "Center-local",
        "bayes_uniform_volume": "Bayes-volume",
        "minimax_absolute": "Minimax-absolute",
        "minimax_relative": "Minimax-relative",
    }
    design_colors = {
        "center_local": "#636363",
        "bayes_uniform_volume": "#3182bd",
        "minimax_absolute": "#e6550d",
        "minimax_relative": "#9e1a1a",
    }
    setting_colors = {
        "z_sensing": "#6baed6",
        "x_sensing": "#31a354",
        "ground_reference": "#fd8d3c",
        "excited_reference": "#756bb1",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    for axis, family in zip(
        axes[0], ("internal_basis", "external_references"), strict=True
    ):
        for design in display_designs:
            efficiencies = sorted(
                float(row["relative_c_efficiency"])
                for row in risk_rows
                if row["family"] == family and row["design"] == design
            )
            quantiles = (np.arange(len(efficiencies)) + 1) / len(efficiencies)
            axis.plot(
                efficiencies,
                quantiles,
                color=design_colors[design],
                linewidth=1.5,
                label=design_labels[design],
            )
            axis.plot(efficiencies[0], quantiles[0], "o", ms=3, color=design_colors[design])
        axis.set_xlim(0.25 if family == "internal_basis" else 0.88, 1.005)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("pointwise c-efficiency relative to oracle")
        axis.set_ylabel("fraction of 81 grid points")
        axis.set_title(
            "Internal basis control" if family == "internal_basis" else "External references"
        )
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=6)

    allocation_axis = axes[1, 0]
    y_positions = []
    y_labels = []
    labeled_components: set[str] = set()
    cursor = 0
    for family in VALIDATION_FAMILIES:
        component_names = design_data["families"][family]["component_names"]
        for design in display_designs:
            weights = design_data["families"][family]["designs"][design]
            left = 0.0
            for component, weight in zip(component_names, weights, strict=True):
                legend_label = None
                if component not in labeled_components:
                    legend_label = component.replace("_", " ")
                    labeled_components.add(component)
                allocation_axis.barh(
                    cursor,
                    100.0 * weight,
                    left=left,
                    color=setting_colors[component],
                    height=0.72,
                    edgecolor="white",
                    linewidth=0.35,
                    label=legend_label,
                )
                left += 100.0 * weight
            y_positions.append(cursor)
            y_labels.append(design_labels[design])
            cursor += 1
        cursor += 0.7
    allocation_axis.set_yticks(y_positions, y_labels)
    allocation_axis.invert_yaxis()
    allocation_axis.set_xlim(0, 100)
    allocation_axis.set_xlabel("binary-readout allocation (%)")
    allocation_axis.set_title("Robust allocations")
    allocation_axis.text(
        -0.39,
        0.77,
        "Internal",
        transform=allocation_axis.transAxes,
        rotation=90,
        rotation_mode="anchor",
        va="center",
        fontweight="bold",
    )
    allocation_axis.text(
        -0.39,
        0.22,
        "External",
        transform=allocation_axis.transAxes,
        rotation=90,
        rotation_mode="anchor",
        va="center",
        fontweight="bold",
    )
    allocation_axis.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.36),
        fontsize=6,
    )

    risk_axis = axes[1, 1]
    x = np.arange(len(display_designs))
    offsets = {"internal_basis": -0.11, "external_references": 0.11}
    markers = {"internal_basis": "o", "external_references": "s"}
    for family in VALIDATION_FAMILIES:
        summaries = design_data["families"][family]["summaries"]
        worst = [
            summaries[design]["worst_predicted_temperature_sd_at_8e6"]
            for design in display_designs
        ]
        bayes_rms = [
            np.sqrt(
                summaries[design]["bayes_uniform_volume_variance_per_readout"]
                / TOTAL_READOUTS
            )
            for design in display_designs
        ]
        risk_axis.plot(
            x + offsets[family],
            worst,
            marker=markers[family],
            linestyle="-",
            color="#9e1a1a" if family == "internal_basis" else "#2171b5",
            label=("Internal worst" if family == "internal_basis" else "External worst"),
        )
        risk_axis.plot(
            x + offsets[family],
            bayes_rms,
            marker=markers[family],
            linestyle="--",
            color="#9e1a1a" if family == "internal_basis" else "#2171b5",
            alpha=0.72,
            label=("Internal Bayes RMS" if family == "internal_basis" else "External Bayes RMS"),
        )
    risk_axis.set_yscale("log")
    risk_axis.set_xticks(x, [design_labels[name] for name in display_designs], rotation=25, rotation_mode="anchor", ha="right")
    risk_axis.set_ylabel("predicted temperature SD at 8×10⁶ readouts")
    risk_axis.set_title("Average-versus-worst precision")
    risk_axis.grid(axis="y", alpha=0.2)
    risk_axis.legend(frameon=False, fontsize=6, ncol=2)
    for axis, label in zip(axes.flat, ("a", "b", "c", "d"), strict=True):
        axis.text(
            -0.18,
            1.10,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    figure.savefig(OUTPUT / "robust_c_optimal_design.png", dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_c_optimal_design.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_c_optimal_design.svg", bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_c_optimal_design.tiff", dpi=600, bbox_inches="tight")
    plt.close(figure)


def validation_figure(
    pooled: list[dict[str, object]], rows: list[dict[str, object]]
) -> None:
    design_labels = {
        "center_local": "Center-local",
        "bayes_uniform_volume": "Bayes-volume",
        "minimax_absolute": "Minimax-absolute",
        "minimax_relative": "Minimax-relative",
    }
    colors = {
        "center_local": "#636363",
        "bayes_uniform_volume": "#3182bd",
        "minimax_absolute": "#e6550d",
        "minimax_relative": "#9e1a1a",
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    x = np.arange(len(VALIDATED_DESIGNS))
    for family, offset, marker in (
        ("internal_basis", -0.10, "o"),
        ("external_references", 0.10, "s"),
    ):
        selected = [
            next(
                item
                for item in pooled
                if item["family"] == family and item["design"] == design
            )
            for design in VALIDATED_DESIGNS
        ]
        coverage = np.asarray([item["pooled_wald_95_coverage"] for item in selected])
        coverage_ci = np.asarray(
            [item["pooled_wald_coverage_wilson_95_interval"] for item in selected]
        )
        convergence = np.asarray([item["pooled_converged_fraction"] for item in selected])
        convergence_ci = np.asarray(
            [item["pooled_convergence_wilson_95_interval"] for item in selected]
        )
        axes[0].errorbar(
            x + offset,
            coverage,
            yerr=np.vstack((coverage - coverage_ci[:, 0], coverage_ci[:, 1] - coverage)),
            fmt=marker,
            capsize=2,
            color="#9e1a1a" if family == "internal_basis" else "#2171b5",
            label="Internal" if family == "internal_basis" else "External",
        )
        axes[1].errorbar(
            x + offset,
            convergence,
            yerr=np.vstack((convergence - convergence_ci[:, 0], convergence_ci[:, 1] - convergence)),
            fmt=marker,
            capsize=2,
            color="#9e1a1a" if family == "internal_basis" else "#2171b5",
        )
        for design_index, design in enumerate(VALIDATED_DESIGNS):
            by_mu = []
            for mu in GRID_AXES["memory_angle"]:
                cells = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["design"] == design
                    and np.isclose(float(row["memory_angle"]), mu)
                ]
                by_mu.append(
                    sum(int(row["converged_successes"]) for row in cells)
                    / sum(int(row["replicates"]) for row in cells)
                )
            if family == "internal_basis":
                axes[2].plot(
                    GRID_AXES["memory_angle"],
                    by_mu,
                    marker="o",
                    ms=3,
                    color=colors[design],
                    label=design_labels[design],
                )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0.90, 0.99)
    axes[0].set_ylabel("pooled 95% Wald coverage")
    axes[0].set_title(
        f"Coverage across {81 * VALIDATION_REPLICATES:,} datasets per design"
    )
    axes[0].legend(frameon=False)
    axes[1].set_ylim(0.72, 1.01)
    axes[1].set_ylabel("MLE convergence fraction")
    axes[1].set_title("Global optimizer stability")
    axes[2].set_ylim(0.45, 1.02)
    axes[2].set_xlabel(r"memory angle $\mu$")
    axes[2].set_ylabel("internal MLE convergence")
    axes[2].set_title("Failure is localized at low memory")
    axes[2].legend(frameon=False, fontsize=5.8)
    for axis in axes[:2]:
        axis.set_xticks(
            x,
            [design_labels[name] for name in VALIDATED_DESIGNS],
            rotation=28,
            rotation_mode="anchor",
            ha="right",
        )
    for axis, label in zip(axes, ("a", "b", "c"), strict=True):
        axis.grid(alpha=0.18)
        axis.text(
            -0.20,
            1.10,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    figure.savefig(OUTPUT / "robust_design_finite_sample.png", dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_design_finite_sample.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_design_finite_sample.svg", bbox_inches="tight")
    figure.savefig(OUTPUT / "robust_design_finite_sample.tiff", dpi=600, bbox_inches="tight")
    plt.close(figure)


def assemble() -> None:
    design_data = json.loads((OUTPUT / "designs.json").read_text(encoding="utf-8"))
    rows = load_validation_shards()
    flat_rows = [{key: value for key, value in row.items() if key != "temperature_estimates"} for row in rows]
    with (OUTPUT / "finite_sample_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    pooled = [
        pooled_validation_summary(rows, family, design)
        for family in VALIDATION_FAMILIES
        for design in VALIDATED_DESIGNS
    ]
    hybrid_x = {
        design: float(design_data["families"]["hybrid"]["designs"][design][1])
        for design in DESIGN_NAMES
    }
    summary = {
        "criterion": design_data["criterion"],
        "parameter_grid": design_data["grid_axes"],
        "grid_points": 81,
        "bayes_priors": {
            "equal_grid": "equal mass on each of the 81 prespecified scenarios",
            "uniform_volume": "tensor-product trapezoidal quadrature for a uniform continuous density on the rectangular grid domain",
        },
        "design_families": design_data["families"],
        "hybrid_x_readout_fraction_by_design": hybrid_x,
        "hybrid_conclusion": (
            "All optimized hybrid designs allocate less than 1e-6 to X-basis "
            "blocks and collapse numerically to the external-reference simplex."
        ),
        "finite_sample_validation": {
            "total_binary_readouts_per_dataset": TOTAL_READOUTS,
            "replicates_per_grid_cell": VALIDATION_REPLICATES,
            "datasets_per_family_and_design": 81 * VALIDATION_REPLICATES,
            "no_exclusions": True,
            "pooled_results": pooled,
        },
        "interpretation": (
            "Bayes and absolute-minimax allocations trade local relative "
            "efficiency for lower absolute risk in the weak-memory region. "
            "Relative minimax protects pointwise oracle efficiency. External "
            "references are intrinsically robust and dominate the hybrid design."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    risk_figure(design_data)
    validation_figure(pooled, rows)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--validation-shard", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.optimize:
        run_optimization()
        return
    if args.validation_shard is not None:
        run_validation_shard(*args.validation_shard)
        return
    if args.assemble:
        assemble()
        return
    run_optimization()
    run_validation_shard(0, 81)
    assemble()


if __name__ == "__main__":
    main()
