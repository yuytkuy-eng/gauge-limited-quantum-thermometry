from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.fisher import classical_fisher_matrix
from memory_thermometry.assignment_fisher import (
    assignment_calibration_fisher,
    assignment_fisher_matrix,
    quantum_assignment_fisher_matrix,
)
from memory_thermometry.classical_full_swap import (
    classical_record_distribution,
    classical_record_log_likelihood,
    critical_ignored_false_positive_rate,
    critical_ignored_readout_error,
    full_swap_instrument_matrices,
    pseudo_true_markov_temperature,
    pseudo_true_temperature_with_assignment_error,
    pseudo_true_temperature_with_readout_error,
    stationary_memory_excitation,
    thermal_excitation_probability,
    temperature_from_excitation_probability,
)
from memory_thermometry.model import (
    CollisionParameters,
    _kron,
    initial_state,
    instrument_step,
    partial_swap_xy,
    readout_kraus,
    record_distribution,
    record_distribution_superoperator,
    record_log_likelihood,
    thermal_state,
)
from memory_thermometry.inference import (
    BlockExperiment,
    CalibrationExperiment,
    calibration_probability_jacobian,
    fisher_scoring_mle,
    quantum_probability_jacobian,
)
from memory_thermometry.identifiability import (
    efficient_score_residual,
    fit_power_law,
    record_probability_tangent,
    similarity_tangent,
)
from memory_thermometry.quantum_output import (
    full_swap_probe_output_state,
    full_swap_probe_quantum_fisher,
)
from memory_thermometry.posterior import (
    BoundedUniformPrior,
    bulk_effective_sample_size,
    gauge_log_absolute_jacobian,
    gauge_to_physical_coordinates,
    importance_reweighted_summary,
    normalized_importance_weights,
    physical_to_gauge_coordinates,
    physical_to_gauge_jacobian,
    posterior_summary,
    run_random_walk_metropolis,
    split_rhat,
    weighted_quantile,
)
from memory_thermometry.two_step_model import (
    TwoStepCollisionParameters,
    two_step_assignment_fisher_matrix,
    two_step_record_distribution,
)
from memory_thermometry.robust_design import (
    c_optimal_variance,
    relative_c_efficiency,
    tensor_product_weights,
    trapezoidal_axis_weights,
)


class ModelTests(unittest.TestCase):
    def test_c_optimal_variance_matches_diagonal_analytic_result(self) -> None:
        components = np.asarray(
            [
                np.diag([2.0, 1.0]),
                np.diag([1.0, 4.0]),
            ]
        )
        weights = np.asarray([0.25, 0.75])
        expected = 1.0 / (0.25 * 2.0 + 0.75 * 1.0)
        self.assertAlmostEqual(
            float(c_optimal_variance(components, weights)), expected
        )

    def test_c_optimal_variance_is_convex_in_design_weights(self) -> None:
        components = np.asarray(
            [
                [[3.0, 0.7], [0.7, 1.2]],
                [[1.1, -0.2], [-0.2, 2.5]],
            ]
        )
        first = np.asarray([0.2, 0.8])
        second = np.asarray([0.75, 0.25])
        midpoint = 0.5 * (first + second)
        midpoint_risk = float(c_optimal_variance(components, midpoint))
        endpoint_average = 0.5 * (
            float(c_optimal_variance(components, first))
            + float(c_optimal_variance(components, second))
        )
        self.assertLessEqual(midpoint_risk, endpoint_average + 1e-13)

    def test_tensor_trapezoidal_weights_are_normalized(self) -> None:
        axis = np.asarray([0.0, 1.0, 3.0])
        weights = trapezoidal_axis_weights(axis)
        self.assertTrue(np.allclose(weights, [1.0 / 6.0, 0.5, 1.0 / 3.0]))
        tensor = tensor_product_weights(axis, np.asarray([2.0, 5.0]))
        self.assertAlmostEqual(float(tensor.sum()), 1.0)
        self.assertEqual(tensor.shape, (6,))

    def test_relative_c_efficiency_is_oracle_over_candidate(self) -> None:
        candidate = np.asarray([2.0, 5.0])
        oracle = np.asarray([1.0, 4.0])
        self.assertTrue(
            np.allclose(relative_c_efficiency(candidate, oracle), [0.5, 0.8])
        )

    def test_superoperator_record_distribution_matches_branching(self) -> None:
        parameters = CollisionParameters(temperature=0.87, memory_angle=0.48)
        for polar in (0.0, 0.5 * np.pi, (0.0, 0.5 * np.pi)):
            branching = record_distribution(
                parameters,
                probe_angles=(0.7, 0.5 * np.pi),
                length=5,
                false_positive=0.02,
                false_negative=0.04,
                measurement_polar_angles=polar,
            )
            vectorized = record_distribution_superoperator(
                parameters,
                probe_angles=(0.7, 0.5 * np.pi),
                length=5,
                false_positive=0.02,
                false_negative=0.04,
                measurement_polar_angles=polar,
            )
            self.assertTrue(np.allclose(branching, vectorized, atol=2e-14))

    def test_block_experiment_supports_measurement_basis_control(self) -> None:
        vector = np.asarray([0.9, 0.5, 0.02, 0.04])
        template = CollisionParameters(temperature=0.9, memory_angle=0.5)
        probability, jacobian = quantum_probability_jacobian(
            vector,
            template,
            probe_angles=0.5 * np.pi,
            length=4,
            measurement_polar_angles=0.5 * np.pi,
        )
        experiment = BlockExperiment(
            probe_angles=0.5 * np.pi,
            counts=1000.0 * probability,
            measurement_polar_angles=0.5 * np.pi,
        )
        from memory_thermometry.inference import score_and_fisher

        _, score, fisher = score_and_fisher(
            vector, template, [experiment], length=4
        )
        expected = 1000.0 * (jacobian / probability) @ jacobian.T
        self.assertTrue(np.allclose(score, 0.0, atol=2e-7))
        self.assertTrue(np.allclose(fisher, expected, rtol=2e-7, atol=2e-9))

    def test_bounded_prior_enforces_detector_simplex(self) -> None:
        prior = BoundedUniformPrior(
            np.asarray(
                [[0.3, 2.0], [0.02, 1.35], [0.0, 0.8], [0.0, 0.8]]
            ),
            maximum_assignment_sum=0.95,
        )
        self.assertTrue(prior.contains(np.asarray([0.9, 0.5, 0.02, 0.04])))
        self.assertFalse(prior.contains(np.asarray([0.9, 0.5, 0.5, 0.5])))

    def test_gauge_coordinate_transform_and_jacobian(self) -> None:
        physical = np.asarray([0.9, 0.5, 0.02, 0.04])
        gauge = physical_to_gauge_coordinates(physical)
        self.assertTrue(
            np.allclose(gauge_to_physical_coordinates(gauge), physical)
        )
        step = 1e-6
        numerical = np.empty((4, 4))
        for index in range(4):
            plus = physical.copy()
            minus = physical.copy()
            plus[index] += step
            minus[index] -= step
            numerical[:, index] = (
                physical_to_gauge_coordinates(plus)
                - physical_to_gauge_coordinates(minus)
            ) / (2.0 * step)
        analytic = physical_to_gauge_jacobian(physical)
        self.assertTrue(np.allclose(analytic, numerical, atol=2e-10))
        self.assertAlmostEqual(
            gauge_log_absolute_jacobian(gauge),
            -np.log(thermal_excitation_probability(physical[0])),
        )

    def test_mcmc_diagnostics_and_summary_for_gaussian_target(self) -> None:
        covariance = np.diag([0.04, 0.02, 0.01, 0.005])
        inverse = np.linalg.inv(covariance)

        def target(vector: np.ndarray) -> float:
            return float(-0.5 * vector @ inverse @ vector)

        initials = np.asarray(
            [
                [-0.1, 0.1, -0.05, 0.02],
                [0.1, -0.1, 0.05, -0.02],
                [0.05, 0.05, -0.02, -0.03],
                [-0.05, -0.05, 0.02, 0.03],
            ]
        )
        result = run_random_walk_metropolis(
            target,
            initials,
            proposal_covariance=0.45 * covariance,
            draws=1200,
            burn_in=400,
            seed=1234,
        )
        summary = posterior_summary(result.samples)
        self.assertTrue(np.all(np.abs(summary.mean) < 0.04))
        self.assertTrue(np.all(result.split_rhat < 1.08))
        self.assertTrue(np.all(result.bulk_effective_sample_size > 80.0))
        self.assertTrue(np.allclose(split_rhat(result.samples), result.split_rhat))
        self.assertTrue(
            np.allclose(
                bulk_effective_sample_size(result.samples),
                result.bulk_effective_sample_size,
            )
        )

    def test_importance_reweighting_matches_discrete_distribution(self) -> None:
        samples = np.asarray(
            [
                [0.5, 0.2, 0.01, 0.02],
                [1.0, 0.4, 0.03, 0.05],
                [1.5, 0.6, 0.05, 0.08],
            ]
        )
        log_weights = np.log(np.asarray([1.0, 2.0, 1.0]))
        weights = normalized_importance_weights(log_weights)
        result = importance_reweighted_summary(samples, log_weights)
        self.assertTrue(np.allclose(weights, [0.25, 0.5, 0.25]))
        self.assertTrue(
            np.allclose(result.summary.mean, weights @ samples)
        )
        self.assertAlmostEqual(result.effective_sample_size, 8.0 / 3.0)
        self.assertTrue(
            np.allclose(
                weighted_quantile(
                    samples[:, 0], np.asarray([0.5]), weights
                ),
                [1.0],
            )
        )

    def test_importance_reweighting_accepts_zero_support_weights(self) -> None:
        samples = np.tile(np.asarray([0.9, 0.5, 0.02, 0.04]), (5, 1))
        log_weights = np.asarray([0.0, 0.0, -np.inf, -np.inf, -np.inf])
        result = importance_reweighted_summary(samples, log_weights)
        self.assertAlmostEqual(result.effective_sample_size, 2.0)
        self.assertAlmostEqual(result.maximum_normalized_weight, 0.5)

    def test_two_step_ideal_moments_scale_as_excitation_power(self) -> None:
        length = 4
        temperatures = (0.7, 1.3)
        normalized_moments = []
        for temperature in temperatures:
            distribution = two_step_record_distribution(
                TwoStepCollisionParameters(
                    temperature=temperature,
                    memory_angle=0.5,
                    system_memory_angle=0.55,
                ),
                probe_angles=0.5 * np.pi,
                length=length,
            )
            excitation = thermal_excitation_probability(temperature)
            indices = np.arange(distribution.size)
            normalized_moments.append(
                np.asarray(
                    [
                        distribution[(indices & mask) == mask].sum()
                        / excitation ** mask.bit_count()
                        for mask in range(1, 2**length)
                    ]
                )
            )
        self.assertTrue(
            np.allclose(normalized_moments[0], normalized_moments[1], atol=2e-13)
        )

    def test_two_step_model_is_normalized_and_recovers_markov_limit(self) -> None:
        two_step = TwoStepCollisionParameters(temperature=0.9, memory_angle=0.0)
        distribution = two_step_record_distribution(
            two_step,
            probe_angles=0.5 * np.pi,
            length=6,
            false_positive=0.02,
            false_negative=0.04,
        )
        one_step = classical_record_distribution(
            CollisionParameters(temperature=0.9, memory_angle=0.0),
            length=6,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=13)
        self.assertTrue(np.allclose(distribution, one_step, atol=2e-13))

    def test_two_step_model_retains_exact_temperature_detector_gauge(self) -> None:
        parameters = TwoStepCollisionParameters(temperature=0.9, memory_angle=0.5)
        alpha = 0.02
        beta = 0.04
        excitation = thermal_excitation_probability(parameters.temperature)
        contrast = 1.0 - alpha - beta
        scale = 1.1
        transformed_excitation = scale * excitation
        transformed_temperature = temperature_from_excitation_probability(
            transformed_excitation
        )
        transformed_beta = 1.0 - alpha - contrast / scale
        original = two_step_record_distribution(
            parameters,
            probe_angles=0.5 * np.pi,
            length=7,
            false_positive=alpha,
            false_negative=beta,
        )
        transformed = two_step_record_distribution(
            replace(parameters, temperature=transformed_temperature),
            probe_angles=0.5 * np.pi,
            length=7,
            false_positive=alpha,
            false_negative=transformed_beta,
        )
        self.assertTrue(np.allclose(original, transformed, atol=3e-13))

    def test_two_step_basis_control_breaks_gauge(self) -> None:
        parameters = TwoStepCollisionParameters(temperature=0.9, memory_angle=0.5)
        energy = two_step_assignment_fisher_matrix(
            parameters,
            probe_angles=0.5 * np.pi,
            length=6,
            false_positive=0.02,
            false_negative=0.04,
        )
        controlled = two_step_assignment_fisher_matrix(
            parameters,
            probe_angles=0.5 * np.pi,
            measurement_polar_angles=(0.0, 0.5 * np.pi),
            length=6,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertEqual(energy.rank, 3)
        self.assertLess(energy.effective_temperature_information, 1e-10)
        self.assertEqual(controlled.rank, 4)
        self.assertGreater(controlled.effective_temperature_information, 1e-5)

    def test_output_probe_energy_diagonal_matches_classical_records(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        for length in range(1, 6):
            output = full_swap_probe_output_state(parameters, length)
            classical = classical_record_distribution(parameters, length)
            self.assertTrue(
                np.allclose(np.diag(output).real, classical, atol=2e-13)
            )
            self.assertAlmostEqual(float(np.trace(output).real), 1.0, places=13)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(output)[0]), -1e-12)

    def test_output_probe_qfi_bounds_energy_measurement_fisher(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        quantum = full_swap_probe_quantum_fisher(parameters, length=5)
        classical = classical_fisher_matrix(
            parameters, probe_angles=0.5 * np.pi, length=5
        )
        difference = quantum.matrix - classical.matrix
        self.assertTrue(np.all(np.linalg.eigvalsh(difference) >= -2e-8))
        self.assertGreaterEqual(
            quantum.effective_temperature_information,
            classical.effective_temperature_information - 2e-8,
        )
        self.assertLess(
            float(np.max(np.abs(quantum.mean_sld_commutator))), 1e-10
        )

    def test_similarity_tangent_leaves_all_word_probabilities_stationary(self) -> None:
        instruments = (
            np.array([[0.42, 0.11], [0.18, 0.29]]),
            np.array([[0.21, 0.27], [0.19, 0.16]]),
        )
        initial = np.array([0.7, 0.3])
        final = np.array([1.0, 1.0])
        generator = np.array([[0.2, -0.1], [0.05, -0.2]])
        tangent, initial_tangent, final_tangent = similarity_tangent(
            instruments, initial, final, generator
        )
        for length in range(1, 6):
            for record in product((0, 1), repeat=length):
                _, derivative = record_probability_tangent(
                    instruments,
                    initial,
                    final,
                    record,
                    tangent,
                    initial_tangent,
                    final_tangent,
                )
                self.assertAlmostEqual(derivative, 0.0, places=13)

    def test_efficient_score_residual_reproduces_schur_information(self) -> None:
        probability = np.array([0.2, 0.3, 0.5])
        jacobian = np.array(
            [[0.03, -0.02, -0.01], [0.01, 0.04, -0.05]]
        )
        residual = efficient_score_residual(probability, jacobian)
        fisher = (jacobian / probability) @ jacobian.T
        schur = fisher[0, 0] - fisher[0, 1] ** 2 / fisher[1, 1]
        self.assertAlmostEqual(float(residual @ residual), schur, places=13)

    def test_power_law_fit_recovers_exact_exponent(self) -> None:
        control = np.geomspace(0.01, 0.2, 8)
        fit = fit_power_law(control, 2.7 * control**4)
        self.assertAlmostEqual(fit.exponent, 4.0, places=12)
        self.assertAlmostEqual(fit.coefficient, 2.7, places=12)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=13)

    def test_local_kron_matches_numpy(self) -> None:
        left = np.array([[1.0, 2.0j], [-0.5, 3.0]])
        right = np.array([[0.2, 0.3], [0.4j, -0.7]])
        self.assertTrue(np.allclose(_kron(left, right), np.kron(left, right)))

    def test_thermal_state_is_normalized(self) -> None:
        rho = thermal_state(temperature=0.8)
        self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=14)
        self.assertTrue(np.all(np.linalg.eigvalsh(rho) >= 0.0))

    def test_partial_swap_is_unitary(self) -> None:
        unitary = partial_swap_xy(0.73)
        identity = unitary.conj().T @ unitary
        self.assertTrue(np.allclose(identity, np.eye(4), atol=1e-13))

    def test_rotated_readout_kraus_operators_are_complete(self) -> None:
        operators = [
            readout_kraus(0.83, outcome, 0.71, -0.36)
            for outcome in (0, 1)
        ]
        completeness = sum(operator.conj().T @ operator for operator in operators)
        self.assertTrue(np.allclose(completeness, np.eye(2), atol=1e-13))

    def test_zero_basis_rotation_recovers_energy_readout(self) -> None:
        angle = 0.63
        expected_zero = np.diag([1.0, np.cos(angle)]).astype(complex)
        expected_one = np.array(
            [[0.0, -1j * np.sin(angle)], [0.0, 0.0]], dtype=complex
        )
        self.assertTrue(np.allclose(readout_kraus(angle, 0), expected_zero))
        self.assertTrue(np.allclose(readout_kraus(angle, 1), expected_one))

    def test_instrument_is_trace_preserving_when_outcomes_are_summed(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.4)
        rho = initial_state(parameters)
        output = instrument_step(rho, parameters, 0.6, 0) + instrument_step(
            rho, parameters, 0.6, 1
        )
        self.assertAlmostEqual(float(np.trace(output).real), 1.0, places=12)

    def test_record_distribution_is_normalized(self) -> None:
        parameters = CollisionParameters(temperature=1.2, memory_angle=0.5)
        distribution = record_distribution(parameters, (0.4, 0.9), length=7)
        self.assertEqual(distribution.shape, (128,))
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=11)
        self.assertTrue(np.all(distribution >= 0.0))

    def test_zero_probe_angle_has_only_the_all_zero_record(self) -> None:
        parameters = CollisionParameters(temperature=1.0, memory_angle=0.8)
        distribution = record_distribution(parameters, 0.0, length=5)
        self.assertAlmostEqual(float(distribution[0]), 1.0, places=12)
        self.assertAlmostEqual(float(distribution[1:].sum()), 0.0, places=12)

    def test_fisher_matrix_is_positive_semidefinite(self) -> None:
        parameters = CollisionParameters(temperature=1.0, memory_angle=0.5)
        result = classical_fisher_matrix(parameters, (0.4, 1.0), length=6)
        self.assertLess(result.normalization_error, 1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.matrix) >= -1e-9))
        self.assertGreaterEqual(result.effective_temperature_information, 0.0)

    def test_assignment_fisher_is_positive_semidefinite(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        result = assignment_fisher_matrix(
            parameters,
            length=7,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertEqual(result.matrix.shape, (4, 4))
        self.assertLess(result.normalization_error, 1e-12)
        self.assertTrue(np.all(result.eigenvalues >= -1e-9))
        self.assertEqual(result.rank, 3)
        self.assertLess(result.effective_temperature_information, 1e-10)
        known_detector = result.matrix[:2, :2]
        known_effective = (
            known_detector[0, 0]
            - known_detector[0, 1] ** 2 / known_detector[1, 1]
        )
        self.assertLessEqual(
            result.effective_temperature_information, known_effective + 1e-10
        )

    def test_assignment_fisher_physical_block_matches_quantum_fisher(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        augmented = assignment_fisher_matrix(
            parameters,
            length=6,
            false_positive=0.0,
            false_negative=0.0,
        )
        physical = classical_fisher_matrix(
            parameters, 0.5 * np.pi, length=6
        )
        self.assertTrue(
            np.allclose(augmented.matrix[:2, :2], physical.matrix, atol=2e-9)
        )

    def test_quantum_assignment_fisher_matches_full_swap_reduction(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        classical = assignment_fisher_matrix(
            parameters,
            length=5,
            false_positive=0.02,
            false_negative=0.04,
        )
        quantum = quantum_assignment_fisher_matrix(
            parameters,
            probe_angles=0.5 * np.pi,
            length=5,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertTrue(np.allclose(classical.matrix, quantum.matrix, atol=2e-8))

    def test_non_full_swap_readout_breaks_assignment_gauge(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        result = quantum_assignment_fisher_matrix(
            parameters,
            probe_angles=0.85,
            length=6,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertEqual(result.rank, 4)
        self.assertGreater(result.effective_temperature_information, 1e-6)

    def test_basis_control_breaks_full_swap_assignment_gauge(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.65)
        result = quantum_assignment_fisher_matrix(
            parameters,
            probe_angles=0.5 * np.pi,
            measurement_polar_angles=(0.0, 0.5 * np.pi),
            length=8,
            false_positive=0.03,
            false_negative=0.04,
        )
        self.assertEqual(result.rank, 4)
        self.assertGreater(result.effective_temperature_information, 1e-5)

    def test_inference_jacobian_reproduces_quantum_fisher(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        vector = np.array([0.9, 0.5, 0.02, 0.04])
        probability, jacobian = quantum_probability_jacobian(
            vector, parameters, probe_angles=0.85, length=5
        )
        reconstructed = (jacobian / probability) @ jacobian.T
        direct = quantum_assignment_fisher_matrix(
            parameters,
            probe_angles=0.85,
            length=5,
            false_positive=0.02,
            false_negative=0.04,
        )
        self.assertTrue(np.allclose(reconstructed, direct.matrix, atol=2e-8))

    def test_fisher_scoring_recovers_expected_count_maximum(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        schedule = (0.7, 1.0)
        probability = record_distribution(
            parameters,
            schedule,
            length=4,
            false_positive=0.02,
            false_negative=0.04,
        )
        fit = fisher_scoring_mle(
            [BlockExperiment(schedule, 100_000 * probability)],
            parameters,
            length=4,
            initial=np.array([0.93, 0.47, 0.023, 0.05]),
            max_iterations=15,
        )
        self.assertTrue(fit.converged)
        self.assertTrue(
            np.allclose(fit.estimate, [0.9, 0.5, 0.02, 0.04], atol=2e-6)
        )

    def test_reference_calibration_counts_break_full_swap_gauge(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        vector = np.array([0.9, 0.5, 0.02, 0.04])
        trajectory_probability = record_distribution(
            parameters,
            0.5 * np.pi,
            length=4,
            false_positive=0.02,
            false_negative=0.04,
        )
        ground_probability, _ = calibration_probability_jacobian(vector, 0)
        excited_probability, _ = calibration_probability_jacobian(vector, 1)
        experiments = [
            BlockExperiment(0.5 * np.pi, 80_000 * trajectory_probability),
            CalibrationExperiment(0, 20_000 * ground_probability),
            CalibrationExperiment(1, 5_000 * excited_probability),
        ]
        fit = fisher_scoring_mle(
            experiments,
            parameters,
            length=4,
            initial=np.array([0.93, 0.47, 0.023, 0.05]),
            max_iterations=15,
        )
        self.assertTrue(fit.converged)
        self.assertTrue(
            np.allclose(fit.estimate, vector, atol=2e-6)
        )

    def test_reference_calibration_fisher_targets_assignment_errors(self) -> None:
        ground = assignment_calibration_fisher(0.02, 0.04, prepared_state=0)
        excited = assignment_calibration_fisher(0.02, 0.04, prepared_state=1)
        self.assertAlmostEqual(ground[2, 2], 1.0 / (0.02 * 0.98))
        self.assertAlmostEqual(excited[3, 3], 1.0 / (0.04 * 0.96))
        self.assertEqual(np.count_nonzero(ground), 1)
        self.assertEqual(np.count_nonzero(excited), 1)

    def test_full_swap_classical_reduction_matches_quantum_model(self) -> None:
        parameters = CollisionParameters(
            temperature=0.83,
            memory_angle=0.47,
            system_memory_angle=0.61,
        )
        quantum = record_distribution(parameters, 0.5 * np.pi, length=7)
        classical = classical_record_distribution(parameters, length=7)
        self.assertTrue(np.allclose(quantum, classical, atol=2e-13))

    def test_noisy_full_swap_quantum_model_matches_classical_reduction(self) -> None:
        parameters = CollisionParameters(
            temperature=0.83,
            memory_angle=0.47,
            system_memory_angle=0.61,
        )
        quantum = record_distribution(
            parameters,
            0.5 * np.pi,
            length=7,
            false_positive=0.025,
            false_negative=0.065,
        )
        classical = classical_record_distribution(
            parameters,
            length=7,
            false_positive=0.025,
            false_negative=0.065,
        )
        self.assertTrue(np.allclose(quantum, classical, atol=2e-13))

    def test_full_swap_instrument_is_stochastic_when_summed(self) -> None:
        parameters = CollisionParameters(temperature=1.1, memory_angle=0.8)
        matrix_zero, matrix_one = full_swap_instrument_matrices(parameters)
        total = matrix_zero + matrix_one
        self.assertTrue(np.all(matrix_zero >= 0.0))
        self.assertTrue(np.all(matrix_one >= 0.0))
        self.assertTrue(np.allclose(total.sum(axis=0), 1.0, atol=1e-14))

    def test_classical_log_likelihood_matches_enumerated_probability(self) -> None:
        parameters = CollisionParameters(temperature=1.2, memory_angle=0.35)
        length = 6
        distribution = classical_record_distribution(parameters, length)
        index = 13
        record = tuple(int(bit) for bit in f"{index:0{length}b}")
        log_likelihood = classical_record_log_likelihood(parameters, record)
        self.assertAlmostEqual(np.exp(log_likelihood), distribution[index], places=13)

    def test_pseudo_true_markov_temperature_is_lower_with_memory(self) -> None:
        parameters = CollisionParameters(temperature=1.3, memory_angle=0.7)
        pseudo_temperature = pseudo_true_markov_temperature(parameters)
        self.assertGreater(pseudo_temperature, 0.0)
        self.assertLess(pseudo_temperature, parameters.temperature)

        matrix_zero, matrix_one = full_swap_instrument_matrices(parameters)
        stationary = np.array(
            [1.0 - stationary_memory_excitation(parameters),
             stationary_memory_excitation(parameters)]
        )
        self.assertTrue(
            np.allclose((matrix_zero + matrix_one) @ stationary, stationary)
        )

    def test_general_log_likelihood_matches_enumeration(self) -> None:
        parameters = CollisionParameters(temperature=0.91, memory_angle=0.42)
        schedule = (0.31, 0.88)
        length = 7
        distribution = record_distribution(parameters, schedule, length)
        index = 77
        record = tuple(int(bit) for bit in f"{index:0{length}b}")
        log_likelihood = record_log_likelihood(parameters, schedule, record)
        self.assertAlmostEqual(np.exp(log_likelihood), distribution[index], places=12)

    def test_noisy_general_log_likelihood_matches_enumeration(self) -> None:
        parameters = CollisionParameters(temperature=0.91, memory_angle=0.42)
        schedule = (0.31, 0.88)
        length = 7
        distribution = record_distribution(
            parameters,
            schedule,
            length,
            false_positive=0.02,
            false_negative=0.05,
        )
        index = 77
        record = tuple(int(bit) for bit in f"{index:0{length}b}")
        log_likelihood = record_log_likelihood(
            parameters,
            schedule,
            record,
            false_positive=0.02,
            false_negative=0.05,
        )
        self.assertAlmostEqual(np.exp(log_likelihood), distribution[index], places=12)

    def test_noisy_readout_instruments_remain_trace_preserving(self) -> None:
        parameters = CollisionParameters(temperature=0.8, memory_angle=0.6)
        ideal_zero, ideal_one = full_swap_instrument_matrices(parameters)
        noisy_zero, noisy_one = full_swap_instrument_matrices(
            parameters, readout_error=0.07
        )
        self.assertTrue(np.all(noisy_zero >= 0.0))
        self.assertTrue(np.all(noisy_one >= 0.0))
        self.assertTrue(
            np.allclose(noisy_zero + noisy_one, ideal_zero + ideal_one)
        )

    def test_calibrated_readout_error_preserves_markov_bias_formula(self) -> None:
        parameters = CollisionParameters(temperature=1.1, memory_angle=0.65)
        ideal = pseudo_true_markov_temperature(parameters)
        calibrated = pseudo_true_temperature_with_readout_error(
            parameters,
            true_readout_error=0.03,
            assumed_readout_error=0.03,
        )
        self.assertAlmostEqual(ideal, calibrated, places=13)

    def test_critical_unmodeled_error_cancels_memory_bias(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        critical = critical_ignored_readout_error(parameters)
        fitted = pseudo_true_temperature_with_readout_error(
            parameters,
            true_readout_error=critical,
            assumed_readout_error=0.0,
        )
        self.assertAlmostEqual(fitted, parameters.temperature, places=12)

    def test_symmetric_and_asymmetric_assignment_interfaces_agree(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        symmetric = full_swap_instrument_matrices(
            parameters, readout_error=0.03
        )
        asymmetric = full_swap_instrument_matrices(
            parameters, false_positive=0.03, false_negative=0.03
        )
        for old_matrix, new_matrix in zip(symmetric, asymmetric):
            self.assertTrue(np.allclose(old_matrix, new_matrix, atol=1e-15))

    def test_asymmetric_instruments_preserve_unconditional_dynamics(self) -> None:
        parameters = CollisionParameters(temperature=0.8, memory_angle=0.6)
        ideal_zero, ideal_one = full_swap_instrument_matrices(parameters)
        noisy_zero, noisy_one = full_swap_instrument_matrices(
            parameters, false_positive=0.02, false_negative=0.08
        )
        self.assertTrue(np.all(noisy_zero >= 0.0))
        self.assertTrue(np.all(noisy_one >= 0.0))
        self.assertTrue(
            np.allclose(noisy_zero + noisy_one, ideal_zero + ideal_one)
        )

    def test_asymmetric_record_likelihood_matches_enumeration(self) -> None:
        parameters = CollisionParameters(temperature=0.86, memory_angle=0.43)
        length = 7
        distribution = classical_record_distribution(
            parameters,
            length,
            false_positive=0.025,
            false_negative=0.065,
        )
        index = 51
        record = tuple(int(bit) for bit in f"{index:0{length}b}")
        log_likelihood = classical_record_log_likelihood(
            parameters,
            record,
            false_positive=0.025,
            false_negative=0.065,
        )
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=13)
        self.assertAlmostEqual(np.exp(log_likelihood), distribution[index], places=13)

    def test_temperature_false_negative_gauge_invariance(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        alpha = 0.02
        beta = 0.04
        excitation = thermal_excitation_probability(parameters.temperature)
        contrast_excitation = (1.0 - alpha - beta) * excitation
        alternative_temperature = 1.3
        alternative_excitation = thermal_excitation_probability(
            alternative_temperature
        )
        alternative_beta = (
            1.0 - alpha - contrast_excitation / alternative_excitation
        )
        original = classical_record_distribution(
            parameters,
            length=9,
            false_positive=alpha,
            false_negative=beta,
        )
        alternative = classical_record_distribution(
            CollisionParameters(
                temperature=alternative_temperature,
                memory_angle=parameters.memory_angle,
                system_memory_angle=parameters.system_memory_angle,
            ),
            length=9,
            false_positive=alpha,
            false_negative=alternative_beta,
        )
        self.assertTrue(np.allclose(original, alternative, atol=2e-15))

    def test_calibrated_asymmetric_error_preserves_memory_bias(self) -> None:
        parameters = CollisionParameters(temperature=1.1, memory_angle=0.65)
        ideal = pseudo_true_markov_temperature(parameters)
        calibrated = pseudo_true_temperature_with_assignment_error(
            parameters,
            true_false_positive=0.02,
            true_false_negative=0.07,
            assumed_false_positive=0.02,
            assumed_false_negative=0.07,
        )
        self.assertAlmostEqual(ideal, calibrated, places=13)

    def test_critical_false_positive_cancels_bias_at_fixed_false_negative(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        false_negative = 0.02
        critical = critical_ignored_false_positive_rate(
            parameters, false_negative
        )
        fitted = pseudo_true_temperature_with_assignment_error(
            parameters,
            true_false_positive=critical,
            true_false_negative=false_negative,
            assumed_false_positive=0.0,
            assumed_false_negative=0.0,
        )
        self.assertAlmostEqual(fitted, parameters.temperature, places=12)

    def test_asymmetric_boundary_recovers_symmetric_threshold(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        symmetric = critical_ignored_readout_error(parameters)
        asymmetric = critical_ignored_false_positive_rate(
            parameters, false_negative=symmetric
        )
        self.assertAlmostEqual(symmetric, asymmetric, places=14)

    def test_asymmetric_assignment_validation(self) -> None:
        parameters = CollisionParameters(temperature=0.9, memory_angle=0.5)
        with self.assertRaises(ValueError):
            full_swap_instrument_matrices(parameters, false_positive=0.02)
        with self.assertRaises(ValueError):
            full_swap_instrument_matrices(
                parameters, false_positive=0.6, false_negative=0.4
            )


if __name__ == "__main__":
    unittest.main()
