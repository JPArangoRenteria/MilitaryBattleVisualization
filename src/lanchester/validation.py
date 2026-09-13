"""
Validation and analytical checks for Lanchester models.

This module provides functions to verify that simulations satisfy
theoretical constraints and invariants.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray


def square_law_invariant(
    model: Any, t: NDArray[np.float64], A: NDArray[np.float64], B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the Square Law invariant along a trajectory.

    For the Square Law model, the quantity I = α·A² - β·B² should remain
    approximately constant during the engagement.

    Args:
        model: A SquareLaw instance with .alpha and .beta attributes
        t: Array of time points (not directly used, but included for clarity)
        A: Array of force A strengths
        B: Array of force B strengths

    Returns:
        Array of invariant values at each time point

    Notes:
        - Small variations (numerical error) are expected from ODE solvers
        - Typical tolerance: abs(I(t) - I(0)) < 1e-4 to 1e-6 depending on solver
        - If invariant drifts significantly, solver accuracy may need improvement
    """
    return model.alpha * A**2 - model.beta * B**2


def invariant_error(
    invariant_values: NDArray[np.float64], tolerance: float = 1e-6
) -> tuple[float, bool]:
    """
    Analyze drift in the invariant over time.

    Args:
        invariant_values: Array of invariant values from square_law_invariant()
        tolerance: Acceptable relative error threshold

    Returns:
        Tuple (max_relative_error, within_tolerance) where:
        - max_relative_error: Maximum relative change in invariant
        - within_tolerance: Boolean indicating if error is acceptable
    """
    if len(invariant_values) < 2:
        return 0.0, True

    initial = invariant_values[0]
    if abs(initial) < 1e-15:  # Guard against division by very small number
        # If invariant is near zero, use absolute tolerance
        max_error = float(np.max(np.abs(invariant_values - initial)))
        within = max_error < tolerance
    else:
        # Relative error
        relative_errors = np.abs(invariant_values - initial) / np.abs(initial)
        max_error = float(np.max(relative_errors))
        within = max_error < tolerance

    return max_error, within


def convergence_check(
    result_coarse: Any, result_fine: Any, tolerance: float = 1e-3
) -> tuple[float, bool]:
    """
    Compare two simulations with different mesh densities.

    Provides a way to verify that the numerical solution is converging
    as the time discretization is refined.

    Args:
        result_coarse: SimulationResult from coarser time grid
        result_fine: SimulationResult from finer time grid
        tolerance: Acceptable relative difference

    Returns:
        Tuple (max_relative_difference, converged) where:
        - max_relative_difference: Max relative error between solutions
        - converged: Boolean indicating if solutions agree within tolerance
    """
    # Interpolate coarse solution to fine time points
    from scipy.interpolate import interp1d

    f_A_coarse = interp1d(
        result_coarse.t, result_coarse.A, kind="cubic", fill_value="extrapolate"
    )
    f_B_coarse = interp1d(
        result_coarse.t, result_coarse.B, kind="cubic", fill_value="extrapolate"
    )

    A_coarse_at_fine = f_A_coarse(result_fine.t)
    B_coarse_at_fine = f_B_coarse(result_fine.t)

    # Compute relative errors
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error_A = np.abs(result_fine.A - A_coarse_at_fine) / (
            np.abs(result_fine.A) + 1e-10
        )
        rel_error_B = np.abs(result_fine.B - B_coarse_at_fine) / (
            np.abs(result_fine.B) + 1e-10
        )

    max_error = float(np.nanmax([np.max(rel_error_A), np.max(rel_error_B)]))
    converged = max_error < tolerance

    return max_error, converged
