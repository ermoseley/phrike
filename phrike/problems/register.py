"""Register all available problems."""

from .registry import ProblemRegistry
from .problem_list.sod import SodProblem
from .problem_list.khi2d import KHI2DProblem
from .problem_list.tgv3d import TGV3DProblem
from .problem_list.turb3d import Turb3DProblem
from .problem_list.acoustic1d import Acoustic1DProblem
from .problem_list.gaussian_wave1d import GaussianWave1DProblem
from .problem_list.shu_osher1d import ShuOsher1DProblem


# Register all problems
ProblemRegistry.register("sod", SodProblem)
ProblemRegistry.register("khi2d", KHI2DProblem)
ProblemRegistry.register("tgv3d", TGV3DProblem)
ProblemRegistry.register("turb3d", Turb3DProblem)
ProblemRegistry.register("acoustic1d", Acoustic1DProblem)
ProblemRegistry.register("gaussian_wave1d", GaussianWave1DProblem)
ProblemRegistry.register("shu_osher1d", ShuOsher1DProblem)
