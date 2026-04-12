from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.float64]
ScalarObjective = Callable[[ArrayFloat], float]
BoundsLike = Sequence[tuple[float, float]]
ModelFunction = Callable[[ArrayFloat, ArrayFloat], ArrayFloat]
LossFunction = Callable[[ArrayFloat, ArrayFloat], float]


def _bounds_from_pairs(bounds: BoundsLike) -> tuple[ArrayFloat, ArrayFloat]:
    if len(bounds) == 0:
        raise ValueError("bounds must contain at least one dimension.")

    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise ValueError("bounds must have shape (D, 2).")

    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError(
            "Each lower bound must be strictly smaller than the upper bound."
        )

    return lower_bounds.astype(np.float64), upper_bounds.astype(np.float64)


class BaseObjectiveFunction(ABC):
    @abstractmethod
    def __call__(self, parameters: ArrayFloat) -> ArrayFloat:
        raise NotImplementedError

    @abstractmethod
    def get_bounds(self) -> tuple[ArrayFloat, ArrayFloat]:
        raise NotImplementedError

    def get_custom_metrics(self, parameters: ArrayFloat) -> dict[str, ArrayFloat]:
        return {}


class FunctionWrapper(BaseObjectiveFunction):
    def __init__(self, fun: ScalarObjective, bounds: BoundsLike) -> None:
        if not callable(fun):
            raise ValueError("fun must be callable.")

        self.fun = fun
        self._lower_bounds, self._upper_bounds = _bounds_from_pairs(bounds)

    def __call__(self, parameters: ArrayFloat) -> ArrayFloat:
        parameter_matrix = np.asarray(parameters, dtype=np.float64)
        if parameter_matrix.ndim != 2:
            raise ValueError("parameters must be a 2D array with shape (N, D).")

        values = np.empty(parameter_matrix.shape[0], dtype=np.float64)
        for idx, row in enumerate(parameter_matrix):
            values[idx] = float(self.fun(row))
        return values

    def get_bounds(self) -> tuple[ArrayFloat, ArrayFloat]:
        return self._lower_bounds.copy(), self._upper_bounds.copy()


class DataFittingObjective(BaseObjectiveFunction):
    def __init__(
        self,
        x_data: ArrayFloat,
        y_data: ArrayFloat,
        model_func: ModelFunction,
        loss_func: LossFunction,
        bounds: BoundsLike,
    ) -> None:
        self.x_data = np.asarray(x_data, dtype=np.float64).reshape(-1)
        self.y_data = np.asarray(y_data, dtype=np.float64).reshape(-1)

        if self.x_data.shape != self.y_data.shape:
            raise ValueError("x_data and y_data must have identical shapes.")
        if not callable(model_func):
            raise ValueError("model_func must be callable.")
        if not callable(loss_func):
            raise ValueError("loss_func must be callable.")

        self.model_func = model_func
        self.loss_func = loss_func
        self._lower_bounds, self._upper_bounds = _bounds_from_pairs(bounds)
        self.dimension = self._lower_bounds.size

    def __call__(self, parameters: ArrayFloat) -> ArrayFloat:
        parameter_matrix = np.asarray(parameters, dtype=np.float64)
        if parameter_matrix.ndim != 2 or parameter_matrix.shape[1] != self.dimension:
            raise ValueError(f"parameters must have shape (N, {self.dimension}).")

        losses = np.empty(parameter_matrix.shape[0], dtype=np.float64)
        for idx, params in enumerate(parameter_matrix):
            y_pred = self._predict(params)
            losses[idx] = float(self.loss_func(self.y_data, y_pred))

        return np.where(np.isfinite(losses), losses, np.inf)

    def get_bounds(self) -> tuple[ArrayFloat, ArrayFloat]:
        return self._lower_bounds.copy(), self._upper_bounds.copy()

    def get_custom_metrics(self, parameters: ArrayFloat) -> dict[str, ArrayFloat]:
        parameter_matrix = np.asarray(parameters, dtype=np.float64)
        if parameter_matrix.ndim != 2 or parameter_matrix.shape[1] != self.dimension:
            raise ValueError(f"parameters must have shape (N, {self.dimension}).")

        mae_values = np.empty(parameter_matrix.shape[0], dtype=np.float64)
        r2_values = np.empty(parameter_matrix.shape[0], dtype=np.float64)

        y_mean = float(np.mean(self.y_data))
        centered = self.y_data - y_mean
        ss_tot = float(np.sum(np.square(centered)))

        for idx, params in enumerate(parameter_matrix):
            y_pred = self._predict(params)
            residuals = self.y_data - y_pred

            mae_values[idx] = float(np.mean(np.abs(residuals)))

            ss_res = float(np.sum(np.square(residuals)))
            if ss_tot <= 1.0e-12:
                r2_values[idx] = 1.0 if ss_res <= 1.0e-12 else 0.0
            else:
                r2_values[idx] = 1.0 - (ss_res / ss_tot)

        return {
            "mae": mae_values,
            "r2": r2_values,
        }

    def _predict(self, params: ArrayFloat) -> ArrayFloat:
        y_pred = np.asarray(
            self.model_func(self.x_data, params),
            dtype=np.float64,
        ).reshape(-1)

        if y_pred.shape != self.y_data.shape:
            raise ValueError(
                "model_func must return predictions with shape matching y_data."
            )
        return y_pred
