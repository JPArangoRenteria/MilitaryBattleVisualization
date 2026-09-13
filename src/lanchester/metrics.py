"""
Metrics for comparing simulation results with reference data.

Standard metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R².
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray


def mae(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> np.floating[Any]:
    """
    Compute Mean Absolute Error.

    MAE = (1/n) * Σ|observed - predicted|

    Args:
        observed: Reference/ground-truth values
        predicted: Model predictions

    Returns:
        Mean absolute error (same units as input)

    Raises:
        ValueError: If arrays have different shapes
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs predicted {predicted.shape}"
        )

    return float(np.mean(np.abs(observed - predicted)))


def rmse(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> np.floating[Any]:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt((1/n) * Σ(observed - predicted)²)

    Args:
        observed: Reference/ground-truth values
        predicted: Model predictions

    Returns:
        Root mean squared error (same units as input)

    Raises:
        ValueError: If arrays have different shapes
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs predicted {predicted.shape}"
        )

    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> np.floating[Any]:
    """
    Compute coefficient of determination (R²).

    R² = 1 - (SS_res / SS_tot)
    where:
      SS_res = Σ(observed - predicted)²
      SS_tot = Σ(observed - mean(observed))²

    R² ranges from -∞ to 1, with:
      R² = 1: Perfect fit
      R² = 0: Model no better than mean
      R² < 0: Model worse than mean

    Args:
        observed: Reference/ground-truth values
        predicted: Model predictions

    Returns:
        R² value (dimensionless)

    Raises:
        ValueError: If arrays have different shapes or SS_tot is zero
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs predicted {predicted.shape}"
        )

    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        if ss_res == 0:
            return 1.0  # Perfect fit (both constant)
        else:
            raise ValueError(
                "SS_tot is zero and SS_res is nonzero: observed data has no variance"
            )

    return float(1.0 - (ss_res / ss_tot))
