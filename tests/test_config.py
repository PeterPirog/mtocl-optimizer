import pytest

from mtocl_optimizer import MTOCLConfig


def test_config_to_dict_contains_all_public_fields() -> None:
    config = MTOCLConfig(max_iterations=25, export_csv="result.csv", random_seed=7)

    data = config.to_dict()

    assert set(data) == {
        "population_size",
        "max_iterations",
        "climate_change_freq",
        "elimination_rate",
        "distortion_sigma",
        "root_signal_sigma",
        "tol",
        "patience",
        "export_csv",
        "random_seed",
    }
    assert data["max_iterations"] == 25
    assert data["export_csv"] == "result.csv"
    assert data["random_seed"] == 7


def test_config_rejects_too_small_population() -> None:
    with pytest.raises(ValueError, match="population_size must be at least 6"):
        MTOCLConfig(population_size=5)
