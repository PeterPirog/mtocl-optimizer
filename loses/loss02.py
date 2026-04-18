from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


@dataclass(frozen=True)
class PhysicalConstants:
    q_c: float = 1.602e-19  # Elementary charge [C]
    k_b_j_per_k: float = 1.38e-23  # Boltzmann constant [J/K]
    a_star_a_per_m2_k2: float = 1.2e6  # Reduced Richardson constant [A/(m^2*K^2)]
    area_m2: float = 1e-6  # Active area [m^2] == 1 mm^2


@dataclass(frozen=True)
class MSMParams:
    phi_b1_ev: float
    phi_b2_ev: float
    n1: float
    n2: float
    rs_ohm: float
    temperature_k: float = 300.0

    @classmethod
    def from_array(cls, values: np.ndarray, temperature_k: float = 300.0) -> "MSMParams":
        return cls(
            phi_b1_ev=float(values[0]),
            phi_b2_ev=float(values[1]),
            n1=float(values[2]),
            n2=float(values[3]),
            rs_ohm=float(values[4]),
            temperature_k=temperature_k,
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.phi_b1_ev, self.phi_b2_ev, self.n1, self.n2, self.rs_ohm], dtype=np.float64)


@dataclass(frozen=True)
class TestCase:
    name: str
    gt: MSMParams


class AsymmetricMSMVIModel:
    """
    Numerically stable V(I) generator for asymmetric back-to-back MSM Schottky diodes.
    Uses delta-I asymptotic sampling to avoid catastrophic cancellation near I -> I02.
    """

    def __init__(self, constants: PhysicalConstants | None = None) -> None:
        self.c = constants or PhysicalConstants()
        self.exp_clip_bounds = (-700.0, 700.0)

    def thermal_voltage(self, temperature_k: float) -> float:
        return (self.c.k_b_j_per_k * temperature_k) / self.c.q_c

    def saturation_current(self, phi_b_ev: float, temperature_k: float) -> float:
        exponent = np.clip(
            -self.c.q_c * phi_b_ev / (self.c.k_b_j_per_k * temperature_k),
            self.exp_clip_bounds[0],
            self.exp_clip_bounds[1],
        )
        return self.c.area_m2 * self.c.a_star_a_per_m2_k2 * (temperature_k**2) * np.exp(exponent)

    def generate_vi_curve(self, params: MSMParams) -> tuple[np.ndarray, np.ndarray]:
        i01 = max(self.saturation_current(params.phi_b1_ev, params.temperature_k), 1e-300)
        i02 = max(self.saturation_current(params.phi_b2_ev, params.temperature_k), 1e-300)
        vt = self.thermal_voltage(params.temperature_k)

        # Two-phase current grid
        i_part1_hi = 0.9 * i02
        i_part1_lo = min(1e-14, i_part1_hi * 1e-6)
        i_part1_lo = max(i_part1_lo, 1e-300)
        i_part1 = np.logspace(np.log10(i_part1_lo), np.log10(i_part1_hi), 500, dtype=np.float64)

        delta_start = 0.1 * i02
        delta_end = max(1e-300, min(1e-80, delta_start * 1e-6))
        delta_i = np.logspace(np.log10(delta_start), np.log10(delta_end), 1000, dtype=np.float64)
        i_total = np.concatenate((i_part1, i02 - delta_i))
        delta_i_exact = np.concatenate((i02 - i_part1, delta_i))
        delta_i_exact = np.clip(delta_i_exact, 1e-300, None)

        voltage_v = (
            params.n1 * vt * np.log1p(i_total / i01)
            - params.n2 * vt * np.log(np.clip(delta_i_exact / i02, 1e-300, None))
            + params.rs_ohm * i_total
        )
        return voltage_v, i_total

    def predict_current_from_voltage(self, voltage_v: np.ndarray, params: MSMParams) -> np.ndarray:
        """
        Predict I(V) for measurement-domain voltages via interpolation over stable V(I) curve.
        """
        v_curve, i_curve = self.generate_vi_curve(params)
        order = np.argsort(v_curve)
        v_sorted = v_curve[order]
        i_sorted = i_curve[order]

        # Remove duplicate voltages before interpolation
        v_unique, unique_idx = np.unique(v_sorted, return_index=True)
        i_unique = i_sorted[unique_idx]

        # Interpolate in log-current space for better conditioning
        log_i_unique = np.log(np.clip(i_unique, 1e-300, None))
        v_query = np.asarray(voltage_v, dtype=np.float64)
        v_query_clipped = np.clip(v_query, v_unique[0], v_unique[-1])
        log_i_pred = np.interp(v_query_clipped, v_unique, log_i_unique)
        return np.exp(log_i_pred)


def t_alst(values_a: np.ndarray, eps_a: float = 1e-10) -> np.ndarray:
    """
    ALST transform:
    T(y) = sgn(y) * ln(1 + |y|/eps + y^2 / (2*eps^2*sqrt(1+(y/eps)^2))
    """
    z = np.clip(np.abs(values_a) / eps_a, 0.0, 1e150)
    smooth_term = (z**2) / (2.0 * np.sqrt(1.0 + z**2))
    return np.sign(values_a) * np.log1p(z + smooth_term)


def alst_cost_function(
    theta: np.ndarray,
    v_data_v: np.ndarray,
    i_data_a: np.ndarray,
    model: AsymmetricMSMVIModel,
    eps_a: float,
    temperature_k: float,
) -> float:
    try:
        params = MSMParams.from_array(theta, temperature_k=temperature_k)
        i_pred_a = model.predict_current_from_voltage(v_data_v, params)
        if not np.all(np.isfinite(i_pred_a)):
            return 1e300
        delta = t_alst(i_data_a, eps_a=eps_a) - t_alst(i_pred_a, eps_a=eps_a)
        if not np.all(np.isfinite(delta)):
            return 1e300
        loss = float(np.mean(delta * delta))
        if not np.isfinite(loss):
            return 1e300
        return loss
    except Exception:
        return 1e300


def perturb_initial_guess(gt: MSMParams, bounds: list[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    gt_array = gt.as_array()
    perturbation = rng.uniform(-0.3, 0.3, size=gt_array.size)
    x0 = gt_array * (1.0 + perturbation)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(x0, lower, upper)


def fit_case_alst(
    case: TestCase,
    model: AsymmetricMSMVIModel,
    eps_a: float,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
) -> tuple[OptimizeResult, np.ndarray, np.ndarray]:
    v_data_v, i_data_a = model.generate_vi_curve(case.gt)
    objective: Callable[[np.ndarray], float] = lambda theta: alst_cost_function(
        theta,
        v_data_v=v_data_v,
        i_data_a=i_data_a,
        model=model,
        eps_a=eps_a,
        temperature_k=case.gt.temperature_k,
    )

    best_result: OptimizeResult | None = None
    n_restarts = 5
    for _ in range(n_restarts):
        x0 = perturb_initial_guess(case.gt, bounds=bounds, rng=rng)
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800, "ftol": 1e-16, "gtol": 1e-10},
        )
        if best_result is None:
            best_result = result
            continue
        if np.isfinite(result.fun) and (not np.isfinite(best_result.fun) or result.fun < best_result.fun):
            best_result = result

    assert best_result is not None
    return best_result, v_data_v, i_data_a


def build_result_table(cases: list[TestCase], extracted: dict[str, MSMParams]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    param_order = ["phi_b1_ev", "phi_b2_ev", "n1", "n2", "rs_ohm"]
    param_labels = {
        "phi_b1_ev": r"$\Phi_{B1}$ [eV]",
        "phi_b2_ev": r"$\Phi_{B2}$ [eV]",
        "n1": r"$n_1$ [-]",
        "n2": r"$n_2$ [-]",
        "rs_ohm": r"$R_s$ [Ohm]",
    }

    for case in cases:
        gt = case.gt
        est = extracted[case.name]
        for key in param_order:
            gt_val = float(getattr(gt, key))
            est_val = float(getattr(est, key))
            error_pct = 100.0 * abs(est_val - gt_val) / abs(gt_val)
            rows.append(
                {
                    "Test": case.name,
                    "Parametr": param_labels[key],
                    "Simulated": gt_val,
                    "Extracted ALST": est_val,
                    "Error %": error_pct,
                }
            )

    return pd.DataFrame(rows)


def plot_benchmark_results(
    cases: list[TestCase],
    extracted: dict[str, MSMParams],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    model: AsymmetricMSMVIModel,
) -> None:
    fig, (ax_parity, ax_fit) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # ---- Left panel: parity plot for Phi_B and n ----
    sim_phi: list[float] = []
    est_phi: list[float] = []
    sim_n: list[float] = []
    est_n: list[float] = []

    for case in cases:
        gt = case.gt
        est = extracted[case.name]
        sim_phi.extend([gt.phi_b1_ev, gt.phi_b2_ev])
        est_phi.extend([est.phi_b1_ev, est.phi_b2_ev])
        sim_n.extend([gt.n1, gt.n2])
        est_n.extend([est.n1, est.n2])

    ax_parity.scatter(sim_phi, est_phi, marker="o", s=70, alpha=0.85, label=r"$\Phi_B$")
    ax_parity.scatter(sim_n, est_n, marker="s", s=70, alpha=0.85, label=r"$n$")

    all_vals = np.array(sim_phi + est_phi + sim_n + est_n, dtype=np.float64)
    lo = float(np.min(all_vals) * 0.95)
    hi = float(np.max(all_vals) * 1.05)
    ax_parity.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label=r"Reference $y=x$")
    ax_parity.set_xlabel("Wartości Symulowane")
    ax_parity.set_ylabel("Wartości Wyekstrahowane ALST")
    ax_parity.set_title(r"Parity Plot: $\Phi_B$ i $n$")
    ax_parity.set_xlim(lo, hi)
    ax_parity.set_ylim(lo, hi)
    ax_parity.grid(True, linestyle="--", alpha=0.35)
    ax_parity.legend(loc="best")

    # ---- Right panel: semilog I-V fitting ----
    colors = {"Test A": "tab:blue", "Test B": "tab:green", "Test C": "tab:red"}
    for case in cases:
        v_data_v, i_data_a = curves[case.name]
        est_params = extracted[case.name]
        i_fit_a = model.predict_current_from_voltage(v_data_v, est_params)
        color = colors[case.name]

        ax_fit.semilogy(v_data_v, i_data_a, "o", markersize=2.0, alpha=0.5, color=color, label=f"{case.name} GT")
        ax_fit.semilogy(v_data_v, i_fit_a, "-", linewidth=2.0, color=color, label=f"{case.name} ALST fit")

    ax_fit.set_xlabel("Voltage V [V]")
    ax_fit.set_ylabel("Current I [A]")
    ax_fit.set_title("Dopasowanie krzywych I-V (ALST)")
    ax_fit.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_fit.legend(loc="best", fontsize=9)

    plt.show()


def main() -> None:
    # Reproducibility
    rng = np.random.default_rng(20260418)

    # Model and ALST setup
    model = AsymmetricMSMVIModel(constants=PhysicalConstants())
    eps_alst_a = 1e-10

    # Ground-truth test suite (Wangyang-inspired)
    cases = [
        TestCase(
            name="Test A",
            gt=MSMParams(phi_b1_ev=0.40, phi_b2_ev=0.35, n1=2.0, n2=4.0, rs_ohm=5.0, temperature_k=300.0),
        ),
        TestCase(
            name="Test B",
            gt=MSMParams(phi_b1_ev=0.80, phi_b2_ev=0.50, n1=1.5, n2=2.0, rs_ohm=200.0, temperature_k=300.0),
        ),
        TestCase(
            name="Test C",
            gt=MSMParams(phi_b1_ev=0.80, phi_b2_ev=0.75, n1=1.4, n2=1.6, rs_ohm=2.5e6, temperature_k=300.0),
        ),
    ]

    # Physical bounds for optimization
    bounds = [
        (0.1, 1.2),  # Phi_B1 [eV]
        (0.1, 1.2),  # Phi_B2 [eV]
        (1.0, 6.0),  # n1 [-]
        (1.0, 6.0),  # n2 [-]
        (1e-1, 1e8),  # Rs [Ohm]
    ]

    extracted_params: dict[str, MSMParams] = {}
    gt_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for case in cases:
        result, v_data_v, i_data_a = fit_case_alst(
            case=case,
            model=model,
            eps_a=eps_alst_a,
            bounds=bounds,
            rng=rng,
        )
        extracted_params[case.name] = MSMParams.from_array(result.x, temperature_k=case.gt.temperature_k)
        gt_curves[case.name] = (v_data_v, i_data_a)
        print(f"{case.name}: success={result.success}, iterations={result.nit}, final_loss={result.fun:.6e}")

    # Console table
    df_results = build_result_table(cases=cases, extracted=extracted_params)
    pd.set_option("display.max_rows", 200)
    print("\n=== ALST Parameter Extraction Benchmark ===")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    # Visualization
    plot_benchmark_results(
        cases=cases,
        extracted=extracted_params,
        curves=gt_curves,
        model=model,
    )


if __name__ == "__main__":
    main()
