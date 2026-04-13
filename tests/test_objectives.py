import numpy as np
import pytest

from mtocl_optimizer import DataFittingObjective, FunctionWrapper


def test_function_wrapper_evaluates_population() -> None:
    def sphere(candidate: np.ndarray) -> float:
        return float(np.sum(candidate**2))

    objective = FunctionWrapper(fun=sphere, bounds=[(-2.0, 2.0), (-2.0, 2.0)])
    candidates = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [-0.5, 0.25],
        ],
        dtype=np.float64,
    )

    values = objective(candidates)

    np.testing.assert_allclose(values, np.array([0.0, 2.0, 0.3125], dtype=np.float64))


def test_data_fitting_objective_losses_and_metrics_shapes() -> None:
    x_data = np.linspace(0.0, 1.0, 24, dtype=np.float64)

    def model_func(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return params[0] * x + params[1]

    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    true_params = np.array([2.0, 1.0], dtype=np.float64)
    y_data = model_func(x_data, true_params)

    objective = DataFittingObjective(
        x_data=x_data,
        y_data=y_data,
        model_func=model_func,
        loss_func=mse,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    )

    candidates = np.array(
        [
            [2.0, 1.0],
            [0.0, 0.0],
            [1.5, 1.2],
        ],
        dtype=np.float64,
    )

    losses = objective(candidates)
    metrics = objective.get_custom_metrics(candidates)

    assert losses.shape == (3,)
    assert losses[0] <= losses[1]
    assert set(metrics) == {"mae", "r2"}
    assert metrics["mae"].shape == (3,)
    assert metrics["r2"].shape == (3,)
    assert metrics["r2"][0] > 0.999


def test_data_fitting_objective_rejects_bad_prediction_shape() -> None:
    x_data = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_data = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    def bad_model_func(_x: np.ndarray, _params: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0], dtype=np.float64)

    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    objective = DataFittingObjective(
        x_data=x_data,
        y_data=y_data,
        model_func=bad_model_func,
        loss_func=mse,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    )

    with pytest.raises(
        ValueError,
        match="model_func must return predictions with shape matching y_data",
    ):
        objective(np.array([[0.5, -0.2]], dtype=np.float64))
