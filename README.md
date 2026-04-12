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

## Acknowledgments / References

- Korani, W. (2021). *Mother Tree Optimization for Solving Continuous and Discrete Optimization Problems*. University of Regina.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
