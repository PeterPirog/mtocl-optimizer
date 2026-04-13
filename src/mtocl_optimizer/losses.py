"""Specialized loss functions for high-dynamic-range MSM parameter extraction."""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayFloat = Union[np.ndarray, float]


def asinh_mse(
    y_true: ArrayFloat,
    y_pred: ArrayFloat,
    scale: float = 0.02585,
) -> float:
    """
    Mean Squared Arcus Sine Hyperbolic Error.

    Designed for MSM structures with huge dynamic ranges in U(I)
    characteristics.

    Args:
        y_true: Measured voltage U.
        y_pred: Predicted voltage U from the theoretical model.
        scale: Thermal voltage V_T = k_B * T / q
            (approximately 0.02585 V at 300 K).
    """
    safe_scale = np.maximum(scale, 1.0e-12)
    asinh_true = np.arcsinh(y_true / safe_scale)
    asinh_pred = np.arcsinh(y_pred / safe_scale)
    return float(np.mean(np.square(asinh_true - asinh_pred)))


def asinh_wls_residuals(
    y_true: ArrayFloat,
    y_pred: ArrayFloat,
    u_y: ArrayFloat,
    scale: float = 0.02585,
) -> np.ndarray:
    """
    Residuals vector for Weighted Least Squares (WLS) in the asinh space.

    Args:
        y_true: Measured voltage U.
        y_pred: Predicted voltage U from the theoretical model.
        u_y: Standard measurement uncertainty of U.
        scale: Thermal voltage V_T.
    """
    safe_scale = np.maximum(scale, 1.0e-12)
    asinh_true = np.arcsinh(y_true / safe_scale)
    asinh_pred = np.arcsinh(y_pred / safe_scale)
    weights = 1.0 / (u_y / safe_scale)
    return np.asarray(weights * (asinh_true - asinh_pred), dtype=np.float64)
