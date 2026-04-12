from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MTOCLConfig:
    """Configuration of the MTO-CL optimizer."""

    population_size: int = 40
    max_iterations: int | None = None
    climate_change_freq: int = 50
    elimination_rate: float = 0.2
    distortion_sigma: float = 0.05
    root_signal_sigma: float = 0.05
    tol: float = 1e-6
    patience: int = 50
    export_csv: str = "result.csv"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 6:
            raise ValueError("population_size must be at least 6.")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive or None.")
        if self.climate_change_freq < 1:
            raise ValueError("climate_change_freq must be positive.")
        if not 0.0 <= self.elimination_rate < 1.0:
            raise ValueError("elimination_rate must satisfy 0.0 <= value < 1.0.")
        if self.distortion_sigma < 0.0:
            raise ValueError("distortion_sigma must be non-negative.")
        if self.root_signal_sigma < 0.0:
            raise ValueError("root_signal_sigma must be non-negative.")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if not self.export_csv:
            raise ValueError("export_csv must be a non-empty path.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
