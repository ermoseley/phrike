"""Multi-GPU (MPI slab-decomposed) spectral solvers.

One MPI rank drives one GPU.  The same code runs serially on a single device
when no MPI launcher is present.  See ``docs/parallel.rst``.
"""

from .comm import Comm, Timers, launched_under_mpi
from .decomp import SlabDecomposition
from .fft import SlabFFT

__all__ = ["Comm", "Timers", "SlabDecomposition", "SlabFFT", "launched_under_mpi"]
