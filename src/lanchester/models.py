"""
Mathematical models for Lanchester dynamics.

This module defines the classical Lanchester models and provides their
derivative functions for use with ODE solvers.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class LinearLaw:
    """
    Classical Lanchester Linear Law model.

    Represents ancient warfare where each side suffers constant attrition
    regardless of opponent strength (e.g., fixed-rate casualties).

    Mathematical form:
        dA/dt = -β
        dB/dt = -α

    where:
        A(t) = strength of force A at time t
        B(t) = strength of force B at time t
        α = attrition rate of force B (constant)
        β = attrition rate of force A (constant)

    Attributes:
        alpha: Attrition rate of force B (units: strength/time)
        beta: Attrition rate of force A (units: strength/time)

    Note:
        The term "linear law" refers to the linearity of attrition in time,
        not to the mathematical form of the coupling between forces.
        Terminology varies across literature; this convention is explicit
        and follows historical usage for ancient warfare models.
    """

    alpha: float
    beta: float

    def derivatives(
        self, t: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute derivatives for the Linear Law model.

        Args:
            t: Current time (not used, included for solver compatibility)
            state: Array [A, B] with current force strengths

        Returns:
            Array [dA/dt, dB/dt] with derivatives
        """
        return np.array([-self.beta, -self.alpha], dtype=np.float64)


@dataclass
class SquareLaw:
    """
    Classical Lanchester Square Law model.

    Represents modern industrial warfare where force effectiveness is
    proportional to strength (e.g., more soldiers can concentrate fire).

    Mathematical form:
        dA/dt = -β·B
        dB/dt = -α·A

    where:
        A(t) = strength of force A at time t
        B(t) = strength of force B at time t
        α = effectiveness coefficient of force A against B
        β = effectiveness coefficient of force B against A

    The Square Law admits an invariant:
        I = α·A² - β·B² = constant

    This invariant is conserved during engagement and can be used to
    validate numerical solutions.

    Attributes:
        alpha: Effectiveness coefficient of force A (units: 1/strength/time)
        beta: Effectiveness coefficient of force B (units: 1/strength/time)

    References:
        Lanchester, F. W. (1914). Aircraft in Warfare: The Dawn of the Fourth Arm.
    """

    alpha: float
    beta: float

    def derivatives(
        self, t: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute derivatives for the Square Law model.

        Args:
            t: Current time (not used, included for solver compatibility)
            state: Array [A, B] with current force strengths

        Returns:
            Array [dA/dt, dB/dt] with derivatives
        """
        A, B = state
        return np.array([-self.beta * B, -self.alpha * A], dtype=np.float64)

    def invariant(self, A: float, B: float) -> float:
        """
        Compute the Square Law invariant.

        The quantity I = α·A² - β·B² remains constant during engagement
        (up to numerical precision).

        Args:
            A: Strength of force A
            B: Strength of force B

        Returns:
            Invariant value
        """
        return self.alpha * A**2 - self.beta * B**2
