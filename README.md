# MTO-CL Optimizer

[![PyPI version](https://img.shields.io/pypi/v/mtocl-optimizer.svg)](https://pypi.org/project/mtocl-optimizer/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/mtocl-optimizer.svg)](https://pypi.org/project/mtocl-optimizer/)

A Python implementation of the **Mother Tree Optimization with Climate Change (MTO-CL)** global optimization algorithm.

MTO-CL is inspired by ecological intelligence in forest systems, particularly the symbiotic relationship between **Douglas Fir trees** and **mycorrhizal fungi networks**, where nutrients and signals are exchanged to support collective survival and adaptation.

## Algorithm Details

This package implements the **Fixed-Offspring (FO) topology** of MTO:

- **Feeder candidate solutions** transfer nutrients/signals to non-feeder solutions.
- Population roles include the Top Mother Tree (TMT), FPCT, FCT, and LPCT groups.
- The optimization step is fully vectorized via FO masks and weight matrices.

It also includes the **Climate Change** mechanism with two phases:

1. **Elimination**: weak agents are replaced by new random agents sampled in-domain.
2. **Distortion**: surviving agents are perturbed to diversify search and avoid stagnation.

Together, these mechanisms balance **exploitation** and **exploration**, helping reduce premature convergence.

## Loss Functions for High-Dynamic-Range MSM Extraction

The `source/losses.py` module provides reusable, numerically stable losses for extracting physical parameters from MSM diode U(I) measurements across large dynamic ranges.

Current functions include:

- `asinh_mse`: Mean-squared error in inverse hyperbolic sine space.
- `asinh_wls_residuals`: Weighted residual vector in inverse hyperbolic sine space.

These transformations improve robustness when current/voltage magnitudes span many orders of magnitude.

## Installation

```bash
pip install mtocl-optimizer
```

## Usage Example

```python
import numpy as np

from mtocl_optimizer import MTOCLConfig, MTOCLOptimizer, FunctionWrapper


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
objective = FunctionWrapper(fun=sphere, bounds=bounds)

config = MTOCLConfig(
    population_size=40,
    max_iterations=300,
    climate_change_freq=25,
    export_csv="result.csv",
    random_seed=42,
)

optimizer = MTOCLOptimizer(config=config, objective=objective)
result = optimizer.optimize(top_n=3)

print("Best loss:", result["best_loss"])
print("Best parameters:", result["best_parameters"])
```

## Running the Examples

The repository includes runnable scripts in `examples/`.

From the repository root:

```bash
PYTHONPATH=src:. python examples/example1.py
```

The script demonstrates MSM-style parameter extraction using MTO-CL and `asinh_mse` from `source.losses`.

## Acknowledgments

- Korani, W. (2021). *Mother Tree Optimization for Solving Continuous and Discrete Optimization Problems*. University of Regina.

## References

- Wael Korani - Mother Tree Optimization for Solving Continuous and Discrete Optimization Problems: https://uregina.scholaris.ca/server/api/core/bitstreams/72725429-5919-4075-a216-6a54da70a244/content

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
