"""Spectral basis modules for PHRIKE.

This package provides interchangeable spectral bases with a common 1D API so the
solver can switch bases via configuration without modifying core logic.

Modules:
    - base: Abstract interface for 1D/2D/3D spectral bases
    - fourier: Periodic Fourier basis using FFT
    - legendre: Legendre–Gauss–Lobatto collocation basis
"""

from .base import Basis1D  # noqa: F401
from .legendre import LegendreLobattoBasis1D  # noqa: F401


