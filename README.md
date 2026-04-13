# MTO-CL Optimizer

[![PyPI version](https://img.shields.io/pypi/v/mtocl-optimizer.svg)](https://pypi.org/project/mtocl-optimizer/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/mtocl-optimizer.svg)](https://pypi.org/project/mtocl-optimizer/)

`mtocl-optimizer` is a Python library for continuous black-box minimization using
**Mother Tree Optimization with Climate Change (MTO-CL)**.

The method is inspired by communication and resource exchange in Douglas-fir
forests through mycorrhizal networks, and was introduced in:

- Wael Korani, *Mother Tree Optimization for Solving Continuous and Discrete
  Optimization Problems* (PhD thesis, 2021).

This repository contains a practical, vectorized implementation of the continuous
variant with fixed-offspring topology and climate-change diversification.

## Why this package

- Simple API for objective-driven optimization (`FunctionWrapper` or custom
  objective classes).
- Reproducible runs through deterministic random seeds.
- Built-in export of final population and metrics to CSV.
- MTO and MTO-CL behavior controlled by explicit configuration.

## Algorithm Summary (from Korani 2021, Chapter 3)

MTO models a population of candidate solutions as active food sources and splits the
population into biologically motivated groups:

- `TMT` (Top Mother Tree): best-ranked agent, explores around itself using random
  signals.
- `FPCT` (First Partially Connected Trees): stronger mid-ranked agents.
- `FCT` (Fully Connected Trees): central connector subgroup.
- `LPCT` (Last Partially Connected Trees): weaker tail-ranked agents.

Fixed-offspring topology defines how each group receives influence from better ranked
agents. In the thesis notation:

- `N_OS = N_T / 2 - 1`
- `N_FCT = 3`
- `N_PCT = N_T - 4`, split into FPCT and LPCT

MTO-CL adds periodic diversification in two phases:

1. **Elimination**: replace the least healthy fraction with random in-domain agents.
2. **Distortion**: apply small perturbations to the survivors.

The intended effect is a better exploration/exploitation balance and reduced risk of
premature convergence.

In the thesis experiments (30D benchmark suite discussed in Chapter 3), MTO-CL is
reported to generally improve reliability and function-evaluation efficiency compared
to MTO without climate change and several PSO variants.

## Implementation Notes

This package is a minimizer: lower objective value is better (`best_loss`).

Mapping to code:

- Optimizer config and validation: [`src/mtocl_optimizer/config.py`](src/mtocl_optimizer/config.py)
- Objective interfaces and wrappers: [`src/mtocl_optimizer/objectives.py`](src/mtocl_optimizer/objectives.py)
- Core MTO-CL loop: [`src/mtocl_optimizer/optimizer.py`](src/mtocl_optimizer/optimizer.py)
- Optional domain losses (e.g., asinh-based): [`src/mtocl_optimizer/losses.py`](src/mtocl_optimizer/losses.py)
- End-to-end example: [`examples/example1.py`](examples/example1.py)

## Installation

```bash
pip install mtocl-optimizer
```

For local development (tests + lint):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer


def sphere(candidate: np.ndarray) -> float:
    return float(np.sum(candidate ** 2))


objective = FunctionWrapper(
    fun=sphere,
    bounds=[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],
)

config = MTOCLConfig(
    population_size=40,
    max_iterations=300,
    climate_change_freq=25,
    elimination_rate=0.2,
    distortion_sigma=0.05,
    root_signal_sigma=0.05,
    tol=1e-6,
    patience=50,
    export_csv="result.csv",
    random_seed=42,
)

optimizer = MTOCLOptimizer(config=config, objective=objective)
result = optimizer.optimize(top_n=3)

print("Best loss:", result["best_loss"])
print("Best parameters:", result["best_parameters"])
print("Iterations:", result["iterations_run"])
```

## Public API

```python
from mtocl_optimizer import (
    MTOCLConfig,
    BaseObjectiveFunction,
    FunctionWrapper,
    DataFittingObjective,
    asinh_mse,
    asinh_wls_residuals,
    MTOCLOptimizer,
)
```

Main entry point:

- `MTOCLOptimizer(config, objective).optimize(top_n=1)`

Returned dictionary includes:

- `best_parameters`, `best_loss`, `top_parameters`, `top_fitness`
- `history`, `iterations_run`, `converged`, timing fields
- CSV export status: `csv_exported`, `csv_error`, `export_csv`
- frozen config snapshot: `config`

## Configuration Reference

`MTOCLConfig` parameters:

- `population_size` (`>= 6`)
- `max_iterations` (`int | None`)
- `climate_change_freq` (apply climate event every `k` iterations)
- `elimination_rate` (`0.0 <= rate < 1.0`)
- `distortion_sigma` (survivor perturbation strength)
- `root_signal_sigma` (TMT random step scale)
- `tol`, `patience` (early stopping)
- `export_csv` (output CSV path)
- `random_seed` (reproducibility)

## Objective Interfaces

Two standard options:

1. **FunctionWrapper**

- Use when you already have `f(x) -> float` for a single candidate.
- The optimizer evaluates a population by repeatedly calling `f`.

2. **DataFittingObjective**

- Use for parameter estimation from `(x_data, y_data)`.
- Provide `model_func(x_data, params)` and `loss_func(y_true, y_pred)`.
- Exposes built-in custom metrics (`mae`, `r2`) for CSV export.

## Validation and QA

Project includes smoke/regression tests under [`tests/`](tests):

- config validation
- objective wrappers
- optimizer output shape/contract
- deterministic behavior with fixed seed
- CSV export success/error paths

Run locally:

```bash
ruff check src tests examples
pytest
```

## Citation and References

- Korani, W. (2021). *Mother Tree Optimization for Solving Continuous and
  Discrete Optimization Problems*. University of Regina.
- Full thesis (University of Regina repository):
  <https://uregina.scholaris.ca/server/api/core/bitstreams/72725429-5919-4075-a216-6a54da70a244/content>

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
