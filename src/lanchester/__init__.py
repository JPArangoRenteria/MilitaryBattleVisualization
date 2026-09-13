"""
Lanchester Dynamics: A scientific computing package for classical Lanchester combat models.

This package provides tools for simulating and analyzing two-sided force engagements
using classical Lanchester models (Linear and Square Law).

Mathematical Models:
  - Square Law: dA/dt = -β·B, dB/dt = -α·A
  - Linear Law: dA/dt = -β, dB/dt = -α

Key Components:
  - models: Define Lanchester models
  - simulation: Numerically solve differential equations
  - validation: Verify theoretical invariants
  - metrics: Compare simulations with reference data
"""

__version__ = "0.1.0"
__author__ = "Juan Arango Renteria"

from lanchester.models import LinearLaw, SquareLaw
from lanchester.simulation import SimulationResult, simulate

__all__ = [
    "LinearLaw",
    "SquareLaw",
    "SimulationResult",
    "simulate",
]
