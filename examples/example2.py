"""MSM validation benchmark inspired by Wangyang et al. (JMST 2025, Fig. 5).

This example reproduces MSM-like I-V cases with:
- different Schottky barrier pairs (Delta Phi_B = 0.2 V and 0.5 V),
- equal ideality factors for both junctions (n1 = n2),
- different series (bulk) resistance Rs.

Then it verifies whether MTO-CL can recover parameters across diverse conditions
and checks parameter non-uniqueness (ambiguity) via multi-seed fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer, asinh_mse

ArrayFloat = np.ndarray


# Thermal voltage at 300 K in eV and V units.
V_T_EV = 0.02585
V_T_V = 0.02585


@dataclass(frozen=True)
class Scenario:
    name: str
    phi_b1: float  # eV
    phi_b2: float  # eV
    n: float
    rs: float  # ohm*cm^2 (effective areal series resistance in this toy model)
    noise_sigma: float = 0.03


PARAM_LOW = np.array([0.45, 0.45, 1.00, 0.0], dtype=np.float64)
PARAM_HIGH = np.array([1.25, 1.25, 2.20, 300.0], dtype=np.float64)
NORM_BOUNDS = [(0.0, 1.0)] * 4


def _to_norm(params: ArrayFloat) -> ArrayFloat:
    return (params - PARAM_LOW) / (PARAM_HIGH - PARAM_LOW)


def _from_norm(params_norm: ArrayFloat) -> ArrayFloat:
    return PARAM_LOW + params_norm * (PARAM_HIGH - PARAM_LOW)


def _j_saturation(phi_b: float, a_star: float = 120.0, temp_k: float = 300.0) -> float:
    """Thermionic-emission-like saturation current density [A/cm^2]."""
    return float(a_star * (temp_k**2) * np.exp(-phi_b / V_T_EV))


def _msm_current_no_series(
    voltage: ArrayFloat,
    phi_b1: float,
    phi_b2: float,
    n: float,
) -> ArrayFloat:
    """Back-to-back MSM current density from Eq. (17)-style expression, Rs=0."""
    voltage_vec = np.asarray(voltage, dtype=np.float64)
    n_safe = max(float(n), 1.0e-9)

    jst_1 = _j_saturation(phi_b1)
    jst_2 = _j_saturation(phi_b2)

    x = voltage_vec / (2.0 * n_safe * V_T_V)
    x_clip = np.clip(x, -90.0, 90.0)

    numerator = 2.0 * jst_1 * jst_2 * np.sinh(x_clip)
    denominator = jst_1 * np.exp(-x_clip) + jst_2 * np.exp(x_clip)

    current = numerator / np.maximum(denominator, 1.0e-300)
    return np.asarray(current, dtype=np.float64)


def msm_current_with_series(
    voltage: ArrayFloat,
    phi_b1: float,
    phi_b2: float,
    n: float,
    rs: float,
    max_iter: int = 80,
    tol: float = 1.0e-13,
) -> ArrayFloat:
    """Implicit Rs solution via fixed-point iteration: J = f(V - J*Rs)."""
    voltage_vec = np.asarray(voltage, dtype=np.float64)
    rs_safe = max(float(rs), 0.0)

    current = _msm_current_no_series(voltage_vec, phi_b1=phi_b1, phi_b2=phi_b2, n=n)
    for _ in range(max_iter):
        next_current = _msm_current_no_series(
            voltage_vec - current * rs_safe,
            phi_b1=phi_b1,
            phi_b2=phi_b2,
            n=n,
        )
        if np.max(np.abs(next_current - current)) < tol:
            current = next_current
            break
        current = next_current

    return np.asarray(current, dtype=np.float64)


def _safe_log(values: ArrayFloat, floor: float = 1.0e-15) -> ArrayFloat:
    return np.maximum(np.abs(np.asarray(values, dtype=np.float64)), floor)


def _format_params(params: ArrayFloat) -> str:
    return (
        f"[Phi_B1={params[0]:.4f}, Phi_B2={params[1]:.4f}, "
        f"n={params[2]:.4f}, Rs={params[3]:.3f}]"
    )


def _fit_single_run(
    voltage: ArrayFloat,
    measured_current: ArrayFloat,
    seed: int,
    max_iterations: int,
    population_size: int,
    export_csv: str,
) -> tuple[ArrayFloat, float]:
    def loss_fn(candidate_norm: ArrayFloat) -> float:
        params = _from_norm(np.asarray(candidate_norm, dtype=np.float64))
        phi_b1, phi_b2, n, rs = params
        predicted = msm_current_with_series(voltage, phi_b1=phi_b1, phi_b2=phi_b2, n=n, rs=rs)
        scale = max(np.median(_safe_log(measured_current)), 1.0e-15)
        loss = asinh_mse(measured_current, predicted, scale=scale)
        return float(loss if np.isfinite(loss) else np.inf)

    objective = FunctionWrapper(fun=loss_fn, bounds=NORM_BOUNDS)
    config = MTOCLConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        climate_change_freq=20,
        elimination_rate=0.2,
        distortion_sigma=0.05,
        root_signal_sigma=0.05,
        tol=1.0e-10,
        patience=50,
        export_csv=export_csv,
        random_seed=seed,
    )

    result = MTOCLOptimizer(config=config, objective=objective).optimize(top_n=3)
    best_norm = np.asarray(result["best_parameters"], dtype=np.float64)
    best_params = _from_norm(best_norm)
    return best_params, float(result["best_loss"])


def _build_scenarios() -> list[Scenario]:
    # Fig. 5d in the review discusses Delta Phi_B around 0.2 V and 0.5 V.
    # We keep n1=n2 and vary Rs to stress-test extraction.
    return [
        Scenario(name="A: dPhi=0.20, Rs=10", phi_b1=0.80, phi_b2=1.00, n=1.12, rs=10.0),
        Scenario(name="B: dPhi=0.50, Rs=10", phi_b1=0.65, phi_b2=1.15, n=1.12, rs=10.0),
        Scenario(name="C: dPhi=0.20, Rs=120", phi_b1=0.80, phi_b2=1.00, n=1.12, rs=120.0),
        Scenario(name="D: dPhi=0.50, Rs=220", phi_b1=0.65, phi_b2=1.15, n=1.12, rs=220.0),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MSM validation benchmark (MTO-CL): scenario sweep with ambiguity check."
        )
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=200,
        help="Population size used in all optimizer runs (default: 200).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations used in all optimizer runs (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.population_size < 6:
        raise ValueError("--population-size must be at least 6.")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive.")

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    voltage = np.linspace(0.03, 1.25, 170, dtype=np.float64)

    scenarios = _build_scenarios()
    ambiguity_seeds = [11, 17, 23, 29, 35, 41]

    summary_rows: list[dict[str, float | str | bool]] = []
    multi_rows: list[dict[str, float | str]] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, scenario in zip(axes_flat, scenarios):
        true_params = np.array(
            [scenario.phi_b1, scenario.phi_b2, scenario.n, scenario.rs],
            dtype=np.float64,
        )
        current_true = msm_current_with_series(
            voltage,
            phi_b1=scenario.phi_b1,
            phi_b2=scenario.phi_b2,
            n=scenario.n,
            rs=scenario.rs,
        )

        rng = np.random.default_rng(1000 + int(round(scenario.rs)))
        noise = rng.normal(loc=0.0, scale=scenario.noise_sigma, size=current_true.shape)
        current_measured = current_true * np.exp(noise)

        # One high-quality fit for visualization.
        best_params, best_loss = _fit_single_run(
            voltage=voltage,
            measured_current=current_measured,
            seed=42,
            max_iterations=args.max_iterations,
            population_size=args.population_size,
            export_csv=str(output_dir / f"example2_{scenario.name.replace(':', '').replace(',', '').replace(' ', '_')}_best_population.csv"),
        )
        current_fit = msm_current_with_series(
            voltage,
            phi_b1=float(best_params[0]),
            phi_b2=float(best_params[1]),
            n=float(best_params[2]),
            rs=float(best_params[3]),
        )

        # Multi-seed fitting for ambiguity check.
        seed_fits: list[tuple[int, ArrayFloat, float]] = []
        for seed in ambiguity_seeds:
            fitted, loss = _fit_single_run(
                voltage=voltage,
                measured_current=current_measured,
                seed=seed,
                max_iterations=args.max_iterations,
                population_size=args.population_size,
                export_csv=str(output_dir / f"example2_tmp_seed_{seed}.csv"),
            )
            seed_fits.append((seed, fitted, loss))
            multi_rows.append(
                {
                    "scenario": scenario.name,
                    "seed": seed,
                    "loss": loss,
                    "phi_b1": float(fitted[0]),
                    "phi_b2": float(fitted[1]),
                    "n": float(fitted[2]),
                    "rs": float(fitted[3]),
                }
            )

        losses = np.array([x[2] for x in seed_fits], dtype=np.float64)
        params_matrix = np.vstack([x[1] for x in seed_fits])
        global_best_idx = int(np.argmin(losses))
        global_best_params = params_matrix[global_best_idx]
        global_best_loss = float(losses[global_best_idx])

        near_mask = losses <= (global_best_loss * 1.03 + 1.0e-12)
        near_params = params_matrix[near_mask]
        near_count = int(np.sum(near_mask))

        if near_count >= 2:
            span = np.ptp(near_params, axis=0)
        else:
            span = np.zeros(4, dtype=np.float64)

        swapped = global_best_params.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        swapped_current = msm_current_with_series(
            voltage,
            phi_b1=float(swapped[0]),
            phi_b2=float(swapped[1]),
            n=float(swapped[2]),
            rs=float(swapped[3]),
        )
        scale = max(np.median(_safe_log(current_measured)), 1.0e-15)
        swapped_loss = asinh_mse(current_measured, swapped_current, scale=scale)
        swap_ratio = float(swapped_loss / max(global_best_loss, 1.0e-20))

        ambiguity_flag = bool(
            (near_count >= 2 and (span[0] > 0.05 or span[1] > 0.05 or span[3] > 25.0))
            or (swap_ratio < 1.15)
        )

        summary_rows.append(
            {
                "scenario": scenario.name,
                "phi_b1_true": float(true_params[0]),
                "phi_b2_true": float(true_params[1]),
                "n_true": float(true_params[2]),
                "rs_true": float(true_params[3]),
                "phi_b1_fit": float(global_best_params[0]),
                "phi_b2_fit": float(global_best_params[1]),
                "n_fit": float(global_best_params[2]),
                "rs_fit": float(global_best_params[3]),
                "best_loss": global_best_loss,
                "near_opt_count": near_count,
                "phi_b1_span_near_opt": float(span[0]),
                "phi_b2_span_near_opt": float(span[1]),
                "n_span_near_opt": float(span[2]),
                "rs_span_near_opt": float(span[3]),
                "swapped_loss_ratio": swap_ratio,
                "ambiguity_detected": ambiguity_flag,
            }
        )

        ax.semilogy(voltage, _safe_log(current_measured), "-o", markersize=2.5, linewidth=1.0, alpha=0.7, label="Dane eksperymentalne (synt.)")
        ax.semilogy(voltage, _safe_log(current_true), linewidth=2.0, label="Krzywa teoretyczna")
        ax.semilogy(voltage, _safe_log(current_fit), "--", linewidth=2.2, label="Krzywa MTO-CL")

        ax.set_title(scenario.name, fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

        text = (
            "Teoria:\n"
            f"{_format_params(true_params)}\n"
            "MTO-CL:\n"
            f"{_format_params(global_best_params)}\n"
            f"loss={global_best_loss:.3e}\n"
            f"ambiguous={ambiguity_flag}"
        )
        ax.text(
            0.02,
            0.03,
            text,
            transform=ax.transAxes,
            fontsize=7.8,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86},
        )

    for ax in axes[1]:
        ax.set_xlabel("Napiecie V [V]")
    for ax in axes[:, 0]:
        ax.set_ylabel("|Gestosc pradu| |J| [A/cm^2] (skala log)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.suptitle(
        "MSM (na podstawie MSM08): dPhi_B = 0.2/0.5 V, n1=n2, rozne Rs",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    cases_plot_path = output_dir / "example2_cases.png"
    fig.savefig(cases_plot_path)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    multi_df = pd.DataFrame(multi_rows)
    summary_path = output_dir / "example2_summary.csv"
    multi_path = output_dir / "example2_multiseed.csv"
    summary_df.to_csv(summary_path, index=False)
    multi_df.to_csv(multi_path, index=False)

    print("=== Example 2: MSM validation summary ===")
    print(
        f"Settings: population_size={args.population_size}, "
        f"max_iterations={args.max_iterations}"
    )
    for row in summary_rows:
        print(
            f"{row['scenario']}: "
            f"best_loss={row['best_loss']:.3e}, "
            f"fit=[Phi1={row['phi_b1_fit']:.4f}, Phi2={row['phi_b2_fit']:.4f}, "
            f"n={row['n_fit']:.4f}, Rs={row['rs_fit']:.2f}], "
            f"ambiguity={row['ambiguity_detected']}, "
            f"swap_ratio={row['swapped_loss_ratio']:.3f}"
        )

    print(f"Saved plot: {cases_plot_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved multiseed CSV: {multi_path}")


if __name__ == "__main__":
    main()
