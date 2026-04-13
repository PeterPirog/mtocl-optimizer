from pathlib import Path

import numpy as np

from mtocl_optimizer import FunctionWrapper, MTOCLConfig, MTOCLOptimizer


def _sphere_objective(dimension: int) -> FunctionWrapper:
    def sphere(candidate: np.ndarray) -> float:
        return float(np.sum(candidate**2))

    return FunctionWrapper(fun=sphere, bounds=[(-5.12, 5.12)] * dimension)


def _build_config(export_csv: Path, random_seed: int = 42) -> MTOCLConfig:
    return MTOCLConfig(
        population_size=12,
        max_iterations=20,
        climate_change_freq=5,
        elimination_rate=0.2,
        distortion_sigma=0.03,
        root_signal_sigma=0.05,
        tol=1.0e-10,
        patience=10,
        export_csv=str(export_csv),
        random_seed=random_seed,
    )


def test_optimize_returns_expected_shapes_and_exports_csv(tmp_path: Path) -> None:
    optimizer = MTOCLOptimizer(
        config=_build_config(tmp_path / "population.csv"),
        objective=_sphere_objective(dimension=3),
    )

    result = optimizer.optimize(top_n=4)

    assert result["best_parameters"].shape == (3,)
    assert result["top_parameters"].shape == (4, 3)
    assert result["top_fitness"].shape == (4,)
    assert result["history"].shape == (result["iterations_run"],)
    assert result["iterations_run"] <= 20
    assert np.isfinite(result["best_loss"])
    assert result["best_loss"] == result["best_fitness"]
    assert result["csv_exported"] is True
    assert result["csv_error"] is None
    assert (tmp_path / "population.csv").exists()


def test_optimize_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    result_a = MTOCLOptimizer(
        config=_build_config(tmp_path / "run_a.csv", random_seed=123),
        objective=_sphere_objective(dimension=2),
    ).optimize(top_n=3)

    result_b = MTOCLOptimizer(
        config=_build_config(tmp_path / "run_b.csv", random_seed=123),
        objective=_sphere_objective(dimension=2),
    ).optimize(top_n=3)

    np.testing.assert_allclose(result_a["best_parameters"], result_b["best_parameters"])
    np.testing.assert_allclose(result_a["top_fitness"], result_b["top_fitness"])
    np.testing.assert_allclose(result_a["history"], result_b["history"])


def test_optimize_caps_top_n_to_population_size(tmp_path: Path) -> None:
    optimizer = MTOCLOptimizer(
        config=_build_config(tmp_path / "population.csv"),
        objective=_sphere_objective(dimension=2),
    )

    result = optimizer.optimize(top_n=100)

    assert result["top_parameters"].shape == (12, 2)
    assert result["top_fitness"].shape == (12,)


def test_optimize_reports_csv_error_when_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing" / "population.csv"
    optimizer = MTOCLOptimizer(
        config=_build_config(missing_path),
        objective=_sphere_objective(dimension=2),
    )

    result = optimizer.optimize(top_n=2)

    assert result["csv_exported"] is False
    assert isinstance(result["csv_error"], str)
    assert result["csv_error"]


def test_evaluate_and_sort_moves_nonfinite_values_to_the_end(tmp_path: Path) -> None:
    def objective_with_inf(candidate: np.ndarray) -> float:
        if candidate[0] > 0.0:
            return float(np.inf)
        return float(np.sum(candidate**2))

    objective = FunctionWrapper(
        fun=objective_with_inf,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    )
    optimizer = MTOCLOptimizer(
        config=_build_config(tmp_path / "population.csv"),
        objective=objective,
    )

    population = np.array(
        [
            [0.8, 0.1],
            [-0.5, 0.2],
            [-0.1, -0.2],
        ],
        dtype=np.float64,
    )
    sorted_population, sorted_fitness = optimizer._evaluate_and_sort(population)

    assert np.isfinite(sorted_fitness[0])
    assert sorted_fitness[-1] == np.inf
    np.testing.assert_allclose(sorted_population[-1], population[0])
