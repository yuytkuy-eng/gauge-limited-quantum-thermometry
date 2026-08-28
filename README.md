# Gauge limits to self-calibrating quantum thermometry: code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22132062.svg)](https://doi.org/10.5281/zenodo.22132062)

This repository contains only the reusable source code, tests, fixed-seed simulation outputs, and data provenance needed to reproduce the associated numerical study. Manuscript and journal-submission files are intentionally excluded.

## Repository layout

```text
src/memory_thermometry/   Reusable model, Fisher, inference, posterior, and design code
scripts/                  End-to-end numerical experiments and data/plot generators
tests/                    Unit and identity tests
results/                  Archived numerical outputs and generated plots
docs/                     Reproduction guide and data dictionary
```

The archived data are deterministic simulations or fixed-seed Monte Carlo outputs; no human, clinical, or third-party restricted data are present.

## Installation

Python 3.11 or later is supported.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`requirements-verified.txt` records the environment used for the final archive audit. `environment.yml` provides a portable Conda alternative.

## Fast verification

These checks do not rerun the expensive Monte Carlo ensembles:

```bash
python -m unittest discover -s tests -v
python scripts/check_archived_results.py
```

The first command currently runs 56 tests. The second validates the archived gauge, quartic-scaling, QFI, finite-sample, and robust-design claims against the JSON result records.

## Reproducing the archived outputs

The inexpensive deterministic outputs can be regenerated directly:

```bash
python scripts/run_assignment_identifiability.py
python scripts/run_basis_symmetry_scaling.py
python scripts/run_quantum_output_benchmark.py
```

The full four-dimensional inference and global-design validations are computationally expensive. Exact commands, sharding instructions, fixed seeds, and expected outputs are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Archived outputs are supplied so that the numerical results can be inspected without rerunning all 15,552 bounded MLE fits.

## Data availability

The `results/` directory contains the processed numerical records, source tables, fixed-seed ensembles, optimization results, and figure files supporting the manuscript. See [`results/README.md`](results/README.md) and [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for file-level interpretation and provenance.

The live code and data repository is <https://github.com/yuytkuy-eng/gauge-limited-quantum-thermometry>. The verified code-and-data-only archive for release `v0.2.0` is available at <https://doi.org/10.5281/zenodo.22138708>. The concept DOI <https://doi.org/10.5281/zenodo.22132062> resolves to the latest archived version.

## Licences

- Source code: MIT (`LICENSE`).
- Author-generated data and figures: CC BY 4.0 (`LICENSE-DATA`).

## Author

Shifan Yu, School of Physics, Xi'an Jiaotong University
<yuytkuy@stu.xjtu.edu.cn>
