from __future__ import annotations

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants and default device setup."""

    q_c: float = 1.602e-19  # Elementary charge [C]
    k_b_j_per_k: float = 1.38e-23  # Boltzmann constant [J/K]
    a_star_a_per_m2_k2: float = 1.2e6  # Reduced Richardson constant [A/(m^2*K^2)]
    area_m2: float = 1e-6  # Junction area [m^2] = 1 mm^2


@dataclass(frozen=True)
class MSMParameters:
    """One asymmetric MSM curve configuration."""

    phi_b1_ev: float  # Barrier of diode 1 [eV]
    phi_b2_ev: float  # Barrier of diode 2 [eV]
    n1: float = 1.0  # Ideality factor of diode 1 [-]
    n2: float = 1.0  # Ideality factor of diode 2 [-]
    rs_ohm: float = 50.0  # Series resistance [Ohm]
    temperature_k: float = 300.0  # Temperature [K]


@dataclass(frozen=True)
class CurveSpec:
    label: str
    params: MSMParameters


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    title: str
    xlim: tuple[float, float]
    curves: list[CurveSpec]


class AsymmetricMSMModel:
    """
    Precision-safe MSM curve generator using V(I) and delta-I asymptote sampling.
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

    def _current_grid_and_delta(self, i02: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-phase current sampling:
        1) forward rise
        2) asymptotic approach to saturation I02
        """
        i_part1 = np.logspace(-14, np.log10(0.9 * i02), 500, dtype=np.float64)
        delta_i = np.logspace(np.log10(0.1 * i02), -80, 1000, dtype=np.float64)

        i_total = np.concatenate((i_part1, i02 - delta_i))
        delta_i_exact = np.concatenate((i02 - i_part1, delta_i))
        return i_total, delta_i_exact

    def voltage_from_current(
        self,
        i_total: np.ndarray,
        delta_i_exact: np.ndarray,
        i01: float,
        i02: float,
        n1: float,
        n2: float,
        rs_ohm: float,
        temperature_k: float,
    ) -> np.ndarray:
        """
        Precision-safe equivalent of:
        V = n1*Vt*ln(I/I01 + 1) - n2*Vt*ln(1 - I/I02) + R*I

        implemented as:
        V = n1*Vt*ln(I/I01 + 1) - n2*Vt*ln(delta_I_exact / I02) + R*I
        """
        vt = self.thermal_voltage(temperature_k)
        term_forward_v = n1 * vt * np.log1p(i_total / i01)
        term_reverse_v = -n2 * vt * np.log(delta_i_exact / i02)
        term_series_v = rs_ohm * i_total
        return term_forward_v + term_reverse_v + term_series_v

    def generate_curve(self, params: MSMParameters) -> tuple[np.ndarray, np.ndarray]:
        i01 = self.saturation_current(params.phi_b1_ev, params.temperature_k)
        i02 = self.saturation_current(params.phi_b2_ev, params.temperature_k)

        i_total, delta_i_exact = self._current_grid_and_delta(i02)
        voltage_v = self.voltage_from_current(
            i_total=i_total,
            delta_i_exact=delta_i_exact,
            i01=i01,
            i02=i02,
            n1=params.n1,
            n2=params.n2,
            rs_ohm=params.rs_ohm,
            temperature_k=params.temperature_k,
        )
        return voltage_v, i_total


class Figure2BenchmarkPlotter:
    """Reproduces a 12-panel benchmark (a-l) analogous to Wangyang Figure 2."""

    def __init__(self, model: AsymmetricMSMModel) -> None:
        self.model = model
        self.v_cutoff_v = 5.0
        self.i_ylim = (1e-15, 1e0)

    @staticmethod
    def _format_r_label(value: float) -> str:
        if value >= 1000.0:
            exponent = int(round(math.log10(value)))
            if np.isclose(value, 10.0**exponent):
                return rf"10^{{{exponent}}}"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    def _build_panels(self) -> list[PanelSpec]:
        t_k = 300.0
        default_n = 1.0
        default_r = 50.0
        panels: list[PanelSpec] = []

        # (a)-(d): influence of reverse barrier Phi_B2
        barrier_sets = [
            ("a", 0.2, [0.10, 0.15, 0.20, 0.25, 0.30]),
            ("b", 0.4, [0.35, 0.40, 0.45, 0.55, 0.65]),
            ("c", 0.6, [0.55, 0.60, 0.65, 0.70, 0.75]),
            ("d", 0.8, [0.70, 0.75, 0.80, 0.85, 0.90]),
        ]
        for letter, phi1, phi2_values in barrier_sets:
            curves = [
                CurveSpec(
                    label=rf"$\Phi_{{B2}}={phi2:.2f}\,\mathrm{{eV}}$",
                    params=MSMParameters(
                        phi_b1_ev=phi1,
                        phi_b2_ev=phi2,
                        n1=default_n,
                        n2=default_n,
                        rs_ohm=default_r,
                        temperature_k=t_k,
                    ),
                )
                for phi2 in phi2_values
            ]
            panels.append(
                PanelSpec(
                    letter=letter,
                    title=rf"$\Phi_{{B1}}={phi1:.2f}\,\mathrm{{eV}}$, varying $\Phi_{{B2}}$",
                    xlim=(0.0, 2.0),
                    curves=curves,
                )
            )

        # (e)-(g): combinations of (n1, n2)
        n_pairs = [(1, 1), (2, 1), (1, 2), (2, 2), (5, 1), (1, 5), (5, 5)]
        ng_sets = [("e", 0.8, 0.75), ("f", 0.4, 0.35), ("g", 0.8, 0.4)]
        for letter, phi1, phi2 in ng_sets:
            curves = [
                CurveSpec(
                    label=rf"$n_1={n1},\,n_2={n2}$",
                    params=MSMParameters(
                        phi_b1_ev=phi1,
                        phi_b2_ev=phi2,
                        n1=float(n1),
                        n2=float(n2),
                        rs_ohm=default_r,
                        temperature_k=t_k,
                    ),
                )
                for n1, n2 in n_pairs
            ]
            panels.append(
                PanelSpec(
                    letter=letter,
                    title=rf"$\Phi_{{B1}}={phi1:.2f}$ eV, $\Phi_{{B2}}={phi2:.2f}$ eV, varying $n_1,n_2$",
                    xlim=(0.0, 2.0),
                    curves=curves,
                )
            )

        # (h): varying (Phi_B1, n1) with Phi_B2 = 0
        h_pairs = [(0.3, 5), (0.6, 2), (0.9, 1), (0.9, 2), (0.9, 5)]
        curves_h = [
            CurveSpec(
                label=rf"$\Phi_{{B1}}={phi1:.1f}\,\mathrm{{eV}},\ n_1={n1}$",
                params=MSMParameters(
                    phi_b1_ev=phi1,
                    phi_b2_ev=0.0,
                    n1=float(n1),
                    n2=1.0,
                    rs_ohm=default_r,
                    temperature_k=t_k,
                ),
            )
            for phi1, n1 in h_pairs
        ]
        panels.append(
            PanelSpec(
                letter="h",
                title=rf"$\Phi_{{B2}}=0\,\mathrm{{eV}}$, varying $(\Phi_{{B1}}, n_1)$",
                xlim=(0.0, 5.0),
                curves=curves_h,
            )
        )

        # (i)-(k): impact of R
        r_values = [1e-1, 10, 100, 1e4, 1e5, 1e6, 1e7]
        rk_sets = [("i", 0.8, 0.75), ("j", 0.4, 0.35), ("k", 0.8, 0.4)]
        for letter, phi1, phi2 in rk_sets:
            curves = [
                CurveSpec(
                    label=rf"$R={self._format_r_label(float(r))}\,\Omega$",
                    params=MSMParameters(
                        phi_b1_ev=phi1,
                        phi_b2_ev=phi2,
                        n1=default_n,
                        n2=default_n,
                        rs_ohm=float(r),
                        temperature_k=t_k,
                    ),
                )
                for r in r_values
            ]
            panels.append(
                PanelSpec(
                    letter=letter,
                    title=rf"$\Phi_{{B1}}={phi1:.2f}$ eV, $\Phi_{{B2}}={phi2:.2f}$ eV, varying $R$",
                    xlim=(0.0, 2.0),
                    curves=curves,
                )
            )

        # (l): impact of extreme R and Phi_B1 range, Phi_B2 = 0
        phi1_values_l = [0.4, 0.5, 0.6, 0.7, 0.8]
        r_extreme_l = [1e-1, 1e7]
        curves_l: list[CurveSpec] = []
        for r in r_extreme_l:
            for phi1 in phi1_values_l:
                curves_l.append(
                    CurveSpec(
                        label=rf"$\Phi_{{B1}}={phi1:.1f}\,\mathrm{{eV}},\ R={self._format_r_label(float(r))}\,\Omega$",
                        params=MSMParameters(
                            phi_b1_ev=phi1,
                            phi_b2_ev=0.0,
                            n1=default_n,
                            n2=default_n,
                            rs_ohm=float(r),
                            temperature_k=t_k,
                        ),
                    )
                )
        panels.append(
            PanelSpec(
                letter="l",
                title=rf"$\Phi_{{B2}}=0\,\mathrm{{eV}}$, varying $\Phi_{{B1}} \in [0.4,0.8]$ eV at extreme $R$",
                xlim=(0.0, 5.0),
                curves=curves_l,
            )
        )

        return panels

    def _plot_curve(self, ax: plt.Axes, curve: CurveSpec, xlim: tuple[float, float]) -> None:
        voltage_v, current_a = self.model.generate_curve(curve.params)
        mask = (
            np.isfinite(voltage_v)
            & np.isfinite(current_a)
            & (current_a > 0.0)
            & (voltage_v >= xlim[0])
            & (voltage_v <= xlim[1])
            & (voltage_v <= self.v_cutoff_v)
        )
        if np.count_nonzero(mask) > 1:
            ax.semilogy(voltage_v[mask], current_a[mask], linewidth=1.5, label=curve.label)

    def plot(self) -> None:
        panel_specs = self._build_panels()
        fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=True)

        for ax, panel in zip(axes.flat, panel_specs):
            for curve in panel.curves:
                self._plot_curve(ax, curve, panel.xlim)
            ax.set_title(f"({panel.letter}) {panel.title}", fontsize=11)
            ax.set_xlabel(r"Voltage $V$ [V]")
            ax.set_ylabel(r"Current $I$ [A]")
            ax.set_xlim(panel.xlim)
            ax.set_ylim(self.i_ylim)
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=8)

        plt.show()


def main() -> None:
    model = AsymmetricMSMModel()
    plotter = Figure2BenchmarkPlotter(model)
    plotter.plot()


if __name__ == "__main__":
    main()
