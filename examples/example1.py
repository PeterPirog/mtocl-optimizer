"""Example: MSM U(I) parameter extraction using MTO-CL + asinh_mse."""

from __future__ import annotations

import numpy as np

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer
from source.losses import asinh_mse

ArrayFloat = np.ndarray


def msm_u_model(i_data: ArrayFloat, params: ArrayFloat) -> ArrayFloat:
    """Simple mock MSM voltage model with two barriers and series resistance."""
    phi_b1, phi_b2, n1, n2, r_s = params

    v_t = 0.02585  # Thermal voltage at ~300 K [V]

    # Mock saturation currents based on barrier heights.
    i_s1 = 1.0e-8 * np.exp(-phi_b1 / max(v_t, 1.0e-12))
    i_s2 = 1.0e-8 * np.exp(-phi_b2 / max(v_t, 1.0e-12))

    # Mock MSM relation (demonstrative, not a full physical derivation).
    branch_1 = n1 * v_t * np.arcsinh(i_data / (2.0 * max(i_s1, 1.0e-30)))
    branch_2 = n2 * v_t * np.arcsinh(i_data / (2.0 * max(i_s2, 1.0e-30)))

    return 0.5 * (branch_1 + branch_2) + r_s * i_data


def main() -> None:
    rng = np.random.default_rng(123)

    # Mock MSM current sweep data (A): wide dynamic range.
    i_data = np.logspace(-10, -2, 180, dtype=np.float64)

    # Ground-truth synthetic parameters: [Phi_B1, Phi_B2, n1, n2, Rs].
    true_params = np.array([0.70, 0.82, 1.30, 1.55, 12.0], dtype=np.float64)

    # Synthetic measured U(I) with noise.
    u_clean = msm_u_model(i_data, true_params)
    u_measured = u_clean + rng.normal(loc=0.0, scale=5.0e-4, size=i_data.shape)

    def fitness(candidate: ArrayFloat) -> float:
        u_pred = msm_u_model(i_data, candidate)
        return asinh_mse(u_measured, u_pred, scale=0.02585)

    bounds = [
        (0.40, 1.20),  # Phi_B1 [eV]
        (0.40, 1.20),  # Phi_B2 [eV]
        (1.00, 2.20),  # n1 [-]
        (1.00, 2.20),  # n2 [-]
        (0.00, 60.0),  # Rs [ohm]
    ]

    objective = FunctionWrapper(fun=fitness, bounds=bounds)

    config = MTOCLConfig(
        population_size=60,
        max_iterations=500,
        climate_change_freq=25,
        elimination_rate=0.2,
        distortion_sigma=0.03,
        root_signal_sigma=0.05,
        tol=1.0e-8,
        patience=80,
        export_csv="example1_population.csv",
        random_seed=42,
    )

    optimizer = MTOCLOptimizer(config=config, objective=objective)
    result = optimizer.optimize(top_n=3)

    print("Best loss:", result["best_loss"])
    print("Best parameters [Phi_B1, Phi_B2, n1, n2, Rs]:")
    print(result["best_parameters"])


if __name__ == "__main__":
    main()
