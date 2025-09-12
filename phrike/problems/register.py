"""Register all available problems."""

from .registry import ProblemRegistry
from .sod import SodProblem
from .khi2d import KHI2DProblem
from .tgv3d import TGV3DProblem
from .turb3d import Turb3DProblem
from .acoustic1d import Acoustic1DProblem
from .gaussian_wave1d import GaussianWave1DProblem


# Register all problems
ProblemRegistry.register("sod", SodProblem)
ProblemRegistry.register("khi2d", KHI2DProblem)
ProblemRegistry.register("tgv3d", TGV3DProblem)
ProblemRegistry.register("turb3d", Turb3DProblem)
ProblemRegistry.register("acoustic1d", Acoustic1DProblem)
ProblemRegistry.register("gaussian_wave1d", GaussianWave1DProblem)
