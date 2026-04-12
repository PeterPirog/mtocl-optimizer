# MTO-CL Optimizer

`mtocl-optimizer` is a Python implementation of **Mother Tree Optimization with Climate Change (MTO-CL)**, a global optimization metaheuristic inspired by the symbiotic resource-sharing behavior between **Douglas Fir trees** and **mycorrhizal fungi networks**.

The optimizer combines a fixed-topology, vectorized population update strategy with periodic climate-change perturbations (elimination and distortion) to preserve exploration while improving convergence behavior.

## Features

- Vectorized MTO-CL implementation using NumPy
- Fixed-Offspring population topology
- Climate Change mechanism (Elimination + Distortion)
- Objective-function wrappers for generic optimization workflows
- CSV export of final population and custom metrics

## Installation

```bash
pip install mtocl-optimizer
```

## Quick Start

```python
import numpy as np

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
objective = FunctionWrapper(fun=sphere, bounds=bounds)

config = MTOCLConfig(
    population_size=40,
    max_iterations=300,
    climate_change_freq=25,
    random_seed=42,
)

optimizer = MTOCLOptimizer(config=config, objective=objective)
result = optimizer.optimize(top_n=3)

print("Best fitness:", result["best_fitness"])
print("Best parameters:", result["best_parameters"])
```

## Package Layout

```text
src/mtocl_optimizer/
├── __init__.py
├── config.py
├── objectives.py
└── optimizer.py
```
