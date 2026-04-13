# Copilot Instructions

This repository implements the MTO-CL algorithm for optimization. When modifying `optimizer.py`, strictly adhere to the Fixed-Offspring topology logic and do not use unvectorized loops over dimensions.

Ensure all numerical operations heavily utilize numpy.

Never introduce uncertainty propagation (e.g., Monte Carlo) inside the core optimizer module; it must remain a pure deterministic/stochastic global optimizer.
