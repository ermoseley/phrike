"""Problem registry and base classes for SpectralHydro simulations."""

from .base import BaseProblem
from .registry import ProblemRegistry

# Import problem classes from problem_list
from .problem_list.sod import SodProblem
from .problem_list.khi2d import KHI2DProblem
from .problem_list.tgv3d import TGV3DProblem
from .problem_list.turb3d import Turb3DProblem
from .problem_list.acoustic1d import Acoustic1DProblem
from .problem_list.shu_osher1d import ShuOsher1DProblem
from .problem_list.gaussian_wave1d import GaussianWave1DProblem
from .problem_list.rti import RTIProblem

__all__ = [
    "BaseProblem", 
    "ProblemRegistry",
    "SodProblem",
    "KHI2DProblem", 
    "TGV3DProblem",
    "Turb3DProblem",
    "Acoustic1DProblem",
    "ShuOsher1DProblem",
    "GaussianWave1DProblem",
    "RTIProblem"
]
