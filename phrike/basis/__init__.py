"""Spectral basis modules for PHRIKE.

This package provides interchangeable spectral bases (e.g., Fourier, Chebyshev)
with a common 1D API so the solver can switch bases via configuration without
modifying core logic.

Modules:
    - base: Abstract interface for 1D spectral bases
    - fourier: Periodic Fourier basis using FFT
    - chebyshev: Chebyshev–Gauss–Lobatto basis using FFT/DCT algorithms
"""

from .base import Basis1D  # noqa: F401


