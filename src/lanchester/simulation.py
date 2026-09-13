"""
Numerical simulation of Lanchester dynamics.

This module provides the ODE solver integration and result handling.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass
class SimulationResult:
    """
    Result object from a Lanchester dynamics simulation.

    Attributes:
        t: Array of time points where solution was computed
        A: Array of force A strengths at each time point
        B: Array of force B strengths at each time point
        success: Whether the solver completed successfully
        message: Solver status message
        status_code: Solver status code (0=success)
        events_occurred: Dict mapping event name to information
    """

    t: NDArray[np.float64]
    A: NDArray[np.float64]
    B: NDArray[np.float64]
    success: bool
    message: str
    status_code: int
    events_occurred: dict[str, Any]

    def final_state(self) -> tuple[float, float]:
        """Get the final force strengths (A, B) at end of simulation."""
        return float(self.A[-1]), float(self.B[-1])

    def winner(self) -> Optional[str]:
        """
        Determine the winner of the engagement.

        Returns:
            "A" if force A survives (B reaches zero first)
            "B" if force B survives (A reaches zero first)
            None if both reach zero simultaneously or neither reaches zero
        """
        A_final, B_final = self.final_state()

        if A_final > 0 and B_final <= 0:
            return "A"
        elif B_final > 0 and A_final <= 0:
            return "B"
        elif A_final <= 0 and B_final <= 0:
            return None  # Draw
        else:
            return None  # Inconclusive


def simulate(
    model: Any,
    t_span: tuple[float, float],
    y0: NDArray[np.float64],
    t_eval: Optional[NDArray[np.float64]] = None,
    num_points: int = 100,
    max_step: Optional[float] = None,
    method: str = "RK45",
    events: Optional[list] = None,
) -> SimulationResult:
    """
    Simulate a Lanchester dynamics model using scipy.integrate.solve_ivp.

    Args:
        model: An object with a `derivatives(t, state) -> ndarray` method.
               (e.g., LinearLaw or SquareLaw instance)
        t_span: Tuple (t0, tf) with start and end times
        y0: Initial state [A0, B0]
        t_eval: Explicit time points for evaluation. If None, generated automatically.
        num_points: Number of points to generate if t_eval is None
        max_step: Maximum internal step size for solver
        method: Integration method (default "RK45", alternatives: "RK23", "DOP853", etc.)
        events: List of event functions (callable) to detect during integration.
                Events should return 0 when triggered.

    Returns:
        SimulationResult with time, force strengths, and solver status

    Notes:
        - Forces cannot be negative. If they reach zero during integration,
          the solver stops.
        - The simulation tracks when forces reach zero via event detection.
        - For best accuracy with the Square Law invariant, use RK45 or better.
    """

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

    # Wrap the model's derivatives method to match solve_ivp signature
    def _ode_wrapper(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return model.derivatives(t, y)

    # Set up event functions if not provided
    # We add a default event to stop integration if forces go negative
    all_events = events if events is not None else []

    # Solve the ODE
    solution = solve_ivp(
        _ode_wrapper,
        t_span,
        y0,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        events=all_events,
        dense_output=False,
    )

    # Extract results
    t_result = solution.t
    y_result = solution.y  # Shape: (2, len(t))
    A_result = y_result[0]
    B_result = y_result[1]

    # Build events_occurred dictionary
    events_dict: dict[str, Any] = {}
    if hasattr(solution, "t_events") and solution.t_events is not None:
        for i, event_times in enumerate(solution.t_events):
            if len(event_times) > 0:
                events_dict[f"event_{i}"] = event_times

    return SimulationResult(
        t=t_result,
        A=A_result,
        B=B_result,
        success=solution.status == 0,
        message=solution.message,
        status_code=solution.status,
        events_occurred=events_dict,
    )
