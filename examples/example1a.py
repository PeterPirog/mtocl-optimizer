"""Example: MSM U(I) fit with MTO-CL and log-scale visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer, asinh_mse

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


def _safe_for_log(values: ArrayFloat, floor: float = 1.0e-9) -> ArrayFloat:
    """Prepare data for logarithmic Y scale used only for visualization."""
    return np.maximum(np.asarray(values, dtype=np.float64), floor)


def _format_params(params: ArrayFloat) -> str:
    return (
        f"[Phi_B1={params[0]:.4f}, Phi_B2={params[1]:.4f}, "
        f"n1={params[2]:.4f}, n2={params[3]:.4f}, Rs={params[4]:.4f}]"
    )


def main() -> None:
    rng = np.random.default_rng(123)

    # Mock MSM current sweep data (A): wide dynamic range.
    i_data = np.logspace(-10, -2, 180, dtype=np.float64)

    # Ground-truth synthetic parameters: [Phi_B1, Phi_B2, n1, n2, Rs].
    true_params = np.array([0.70, 0.82, 1.30, 1.55, 12.0], dtype=np.float64)

    # Synthetic measured U(I) with noise (treated as experimental data).
    u_theoretical = msm_u_model(i_data, true_params)
    u_measured = u_theoretical + rng.normal(loc=0.0, scale=5.0e-4, size=i_data.shape)

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
        export_csv="example1a_population.csv",
        random_seed=42,
    )

    optimizer = MTOCLOptimizer(config=config, objective=objective)
    result = optimizer.optimize(top_n=3)

    best_params = np.asarray(result["best_parameters"], dtype=np.float64)
    u_mto = msm_u_model(i_data, best_params)

    loss_theoretical = asinh_mse(u_measured, u_theoretical, scale=0.02585)
    loss_mto = asinh_mse(u_measured, u_mto, scale=0.02585)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(
        i_data,
        _safe_for_log(u_measured),
        "-o",
        linewidth=1.2,
        markersize=2.8,
        alpha=0.7,
        label="Dane eksperymentalne",
    )
    ax.plot(
        i_data,
        _safe_for_log(u_theoretical),
        linewidth=2.0,
        label="Krzywa teoretyczna",
    )
    ax.plot(
        i_data,
        _safe_for_log(u_mto),
        "--",
        linewidth=2.2,
        label="Krzywa dla parametrów MTO-CL",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prąd I [A]")
    ax.set_ylabel("Napięcie U [V] (skala log)")
    ax.set_title("MSM U(I): eksperyment, teoria i dopasowanie MTO-CL")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left")

    params_text = (
        "Parametry teoretyczne:\n"
        f"{_format_params(true_params)}\n\n"
        "Parametry wyliczone (MTO-CL):\n"
        f"{_format_params(best_params)}\n\n"
        f"asinh_mse(teoria)={loss_theoretical:.4e}\n"
        f"asinh_mse(MTO-CL)={loss_mto:.4e}"
    )
    ax.text(
        0.02,
        0.03,
        params_text,
        transform=ax.transAxes,
        fontsize=8.8,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    plot_path = "example1a_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)

    print("Best loss:", result["best_loss"])
    print("Best parameters [Phi_B1, Phi_B2, n1, n2, Rs]:")
    print(best_params)
    print("Theoretical parameters:")
    print(true_params)
    print(f"Saved plot: {plot_path}")
    print(f"Saved population CSV: {config.export_csv}")


if __name__ == "__main__":
    main()
