# Reproducibility guide

## Verification levels

### Level 1: identities and archived-claim audit

Expected runtime is below a few minutes on a laptop.

```bash
python -m unittest discover -s tests -v
python scripts/check_archived_results.py
```

### Level 2: deterministic core figures

```bash
python scripts/run_assignment_identifiability.py
python scripts/run_basis_symmetry_scaling.py
python scripts/run_quantum_output_benchmark.py
python scripts/run_two_step_memory.py
```

These scripts overwrite files in their corresponding `results/` directories.

### Level 3: centre-point four-dimensional inference

```bash
python scripts/run_four_dimensional_inference.py
```

This regenerates the bounded four-parameter MLE ensembles, posterior samples, coverage diagnostics, Figures 4 and the associated summary. Fixed seeds are defined near the end of the script.

### Level 4: full 81-point coverage and prior sensitivity

The grid calculation can be split across workers. Run nonoverlapping half-open intervals, then six posterior cases, then assemble:

```bash
python scripts/run_parameter_grid_robustness.py --grid-shard 0 10
python scripts/run_parameter_grid_robustness.py --grid-shard 10 20
python scripts/run_parameter_grid_robustness.py --grid-shard 20 30
python scripts/run_parameter_grid_robustness.py --grid-shard 30 40
python scripts/run_parameter_grid_robustness.py --grid-shard 40 50
python scripts/run_parameter_grid_robustness.py --grid-shard 50 60
python scripts/run_parameter_grid_robustness.py --grid-shard 60 70
python scripts/run_parameter_grid_robustness.py --grid-shard 70 81
python scripts/run_parameter_grid_robustness.py --prior-case 0
python scripts/run_parameter_grid_robustness.py --prior-case 1
python scripts/run_parameter_grid_robustness.py --prior-case 2
python scripts/run_parameter_grid_robustness.py --prior-case 3
python scripts/run_parameter_grid_robustness.py --prior-case 4
python scripts/run_parameter_grid_robustness.py --prior-case 5
python scripts/run_parameter_grid_robustness.py --assemble
```

### Level 5: robust c-optimal design and 15,552-fit validation

First optimize the design families, then run the validation shards, then assemble:

```bash
python scripts/run_robust_c_optimal_design.py --optimize
python scripts/run_robust_c_optimal_design.py --validation-shard 0 10
python scripts/run_robust_c_optimal_design.py --validation-shard 10 20
python scripts/run_robust_c_optimal_design.py --validation-shard 20 30
python scripts/run_robust_c_optimal_design.py --validation-shard 30 40
python scripts/run_robust_c_optimal_design.py --validation-shard 40 50
python scripts/run_robust_c_optimal_design.py --validation-shard 50 60
python scripts/run_robust_c_optimal_design.py --validation-shard 60 70
python scripts/run_robust_c_optimal_design.py --validation-shard 70 81
python scripts/run_robust_c_optimal_design.py --assemble
```

Running the script without arguments performs optimization, all 81 validation cells, and assembly sequentially.

## Determinism and numerical tolerances

Monte Carlo and MCMC seeds are fixed in the scripts. Exact optimizer output can vary slightly across BLAS, NumPy, and SciPy builds. Reproduction should therefore compare scientific tolerances and rounded manuscript values, not require byte-identical floating-point files. `scripts/check_archived_results.py` implements conservative claim-level checks.

## Verified environment

The public archive was audited on 2026-08-28 with Python 3.14.4, NumPy 2.4.4, SciPy 1.17.1, and Matplotlib 3.10.9. The test workflow also exercises supported Python versions on GitHub Actions.

The project evolved across several compatible Python environments, so the archived result files are not claimed to be byte-for-byte products of the final verification environment. The analytical identities, tests, rounded manuscript numbers, and output schemas are the reproducibility targets.
