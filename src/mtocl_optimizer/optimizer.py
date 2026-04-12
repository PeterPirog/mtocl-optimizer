from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from .config import MTOCLConfig
from .objectives import BaseObjectiveFunction

ArrayFloat = NDArray[np.float64]


class OptimizationResult(TypedDict):
    best_parameters: ArrayFloat
    best_loss: float
    best_fitness: float
    top_parameters: ArrayFloat
    top_fitness: ArrayFloat
    history: ArrayFloat
    elapsed_time: float
    elapsed_seconds: float
    iterations_run: int
    converged: bool
    export_csv: str
    csv_exported: bool
    csv_error: str | None
    config: dict[str, Any]


class MTOCLOptimizer:
    def __init__(
        self,
        config: MTOCLConfig,
        objective: BaseObjectiveFunction,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.objective = objective
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.rng = np.random.default_rng(self.config.random_seed)

        lower_bounds, upper_bounds = self.objective.get_bounds()
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
        self.dimension = int(self.lower_bounds.size)
        self.population_size = self.config.population_size

        if self.dimension == 0:
            raise ValueError("The objective must define at least one dimension.")
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have the same shape.")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError(
                "Each lower bound must be strictly smaller than the upper bound."
            )

        self._weight_matrix = self._build_fixed_offspring_weights(self.population_size)

    def optimize(self, top_n: int = 1) -> OptimizationResult:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        start_time = time.perf_counter()
        population = self._initialize_population()

        best_parameters = population[0].copy()
        best_fitness = np.inf
        prev_best_fitness = np.inf
        no_improve_count = 0
        history: list[float] = []
        iteration = 0
        converged = False

        if self.config.max_iterations is not None:
            log_interval = max(1, self.config.max_iterations // 10)
        else:
            log_interval = 50

        while True:
            iteration += 1
            population, fitness = self._evaluate_and_sort(population)

            current_best_fitness = float(fitness[0])
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_parameters = population[0].copy()

            improvement = prev_best_fitness - best_fitness
            if improvement < self.config.tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best_fitness = best_fitness

            history.append(best_fitness)

            if iteration == 1 or iteration % log_interval == 0:
                self.logger.info(
                    "Iteration %d | best_fitness=%.8e | no_improve=%d/%d",
                    iteration,
                    best_fitness,
                    no_improve_count,
                    self.config.patience,
                )

            if no_improve_count >= self.config.patience:
                converged = True
                self.logger.info(
                    "Early stopping triggered at iteration %d (tol=%.3e, patience=%d).",
                    iteration,
                    self.config.tol,
                    self.config.patience,
                )
                break

            if (
                self.config.max_iterations is not None
                and iteration >= self.config.max_iterations
            ):
                self.logger.info(
                    "Stopping at max_iterations=%d.",
                    self.config.max_iterations,
                )
                break

            if iteration % self.config.climate_change_freq == 0:
                population = self._apply_climate_change(population)

            population = self._update_population(population)

        population, fitness = self._evaluate_and_sort(population)

        sort_idx = np.argsort(fitness)
        final_population = population[sort_idx]
        final_fitness = fitness[sort_idx]

        best_parameters = final_population[0].copy()
        best_fitness = float(final_fitness[0])

        top_k = min(top_n, self.population_size)

        n_agents = self.population_size
        half = n_agents // 2

        agent_types = np.full(n_agents, "TMT", dtype=object)
        agent_types[1 : half - 1] = "FPCT"
        agent_types[half - 1 : half + 2] = "FCT"
        agent_types[half + 2 :] = "LPCT"

        column_width = max(2, len(str(self.dimension)))
        columns = [f"p{i + 1:0{column_width}d}" for i in range(self.dimension)]

        csv_exported = False
        csv_error: str | None = None

        try:
            import pandas as pd

            df = pd.DataFrame(final_population, columns=columns)
            df.insert(0, "agent_type", agent_types)
            df["loss"] = final_fitness

            custom_metrics = self.objective.get_custom_metrics(final_population)
            for metric_name, metric_values in custom_metrics.items():
                metric_array = np.asarray(metric_values, dtype=np.float64).reshape(-1)
                if metric_array.shape != (final_population.shape[0],):
                    raise ValueError(
                        f"Custom metric '{metric_name}' must have shape "
                        f"({final_population.shape[0]},)."
                    )
                df[metric_name] = metric_array

            df.to_csv(self.config.export_csv, index=False)
            csv_exported = True
            self.logger.info("Final population exported to %s", self.config.export_csv)

        except (OSError, IOError, PermissionError) as exc:
            csv_error = str(exc)
            self.logger.warning(
                "Could not export final population to '%s': %s",
                self.config.export_csv,
                exc,
            )

        elapsed_time = float(time.perf_counter() - start_time)

        return {
            "best_parameters": best_parameters.copy(),
            "best_loss": best_fitness,
            "best_fitness": best_fitness,
            "top_parameters": final_population[:top_k].copy(),
            "top_fitness": final_fitness[:top_k].copy(),
            "history": np.asarray(history, dtype=np.float64),
            "elapsed_time": elapsed_time,
            "elapsed_seconds": elapsed_time,
            "iterations_run": iteration,
            "converged": converged,
            "export_csv": self.config.export_csv,
            "csv_exported": csv_exported,
            "csv_error": csv_error,
            "config": self.config.to_dict(),
        }

    def _initialize_population(self) -> ArrayFloat:
        return self.rng.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dimension),
        ).astype(np.float64)

    def _evaluate_and_sort(
        self,
        population: ArrayFloat,
    ) -> tuple[ArrayFloat, ArrayFloat]:
        fitness = np.asarray(self.objective(population), dtype=np.float64).reshape(-1)
        if fitness.shape != (population.shape[0],):
            raise ValueError("Objective must return a vector with shape (N,).")

        safe_fitness = np.where(np.isfinite(fitness), fitness, np.inf)
        order = np.argsort(safe_fitness)
        return population[order], safe_fitness[order]

    def _update_population(self, population: ArrayFloat) -> ArrayFloat:
        updated = population.copy()

        diff = population[np.newaxis, :, :] - population[:, np.newaxis, :]
        update_step = np.sum(self._weight_matrix[:, :, np.newaxis] * diff, axis=1)

        updated[1:] = population[1:] + update_step[1:]
        updated[0] = (
            population[0]
            + self.config.root_signal_sigma
            * self.rng.standard_normal(self.dimension)
        )

        return np.clip(updated, self.lower_bounds, self.upper_bounds)

    def _apply_climate_change(self, population: ArrayFloat) -> ArrayFloat:
        updated = population.copy()
        n_eliminated = int(self.config.elimination_rate * self.population_size)
        n_survivors = self.population_size - n_eliminated

        if n_eliminated > 0:
            updated[n_survivors:] = self.rng.uniform(
                low=self.lower_bounds,
                high=self.upper_bounds,
                size=(n_eliminated, self.dimension),
            )

        if n_survivors > 0:
            span = self.upper_bounds - self.lower_bounds
            distortion = (
                self.config.distortion_sigma
                * span
                * self.rng.standard_normal(size=(n_survivors, self.dimension))
            )
            updated[:n_survivors] = updated[:n_survivors] + distortion

        return np.clip(updated, self.lower_bounds, self.upper_bounds)

    @staticmethod
    def _build_fixed_offspring_weights(population_size: int) -> ArrayFloat:
        n_t = population_size
        half = n_t // 2
        n_os = int(n_t / 2) - 1

        n_indices = np.arange(n_t, dtype=np.int64)[:, np.newaxis]
        i_indices = np.arange(n_t, dtype=np.int64)[np.newaxis, :]

        row_fpct = (n_indices >= 1) & (n_indices < half - 1)
        mask_fpct = row_fpct & (i_indices >= 0) & (i_indices <= n_indices - 1)

        fct_lower = np.maximum(0, n_indices - n_os)
        row_fct = (n_indices >= half - 1) & (n_indices <= half + 1)
        mask_fct = row_fct & (i_indices >= fct_lower) & (i_indices <= n_indices - 1)

        lpct_lower = np.maximum(0, n_indices - n_os)
        lpct_upper = min(n_t - 1, n_t - n_os)
        row_lpct = n_indices > half + 1
        mask_lpct = row_lpct & (i_indices >= lpct_lower) & (i_indices <= lpct_upper)

        mask = mask_fpct | mask_fct | mask_lpct

        denom = n_indices - i_indices + 1
        safe_denom = np.where(denom == 0, 1.0, denom)
        weights = np.where(mask, 1.0 / safe_denom, 0.0)

        weights[0, :] = 0.0
        return weights.astype(np.float64)
