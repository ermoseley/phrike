"""Register all available problems."""

from .registry import ProblemRegistry
from .problem_list.sod import SodProblem
from .problem_list.khi2d import KHI2DProblem
from .problem_list.tgv3d import TGV3DProblem
from .problem_list.turb3d import Turb3DProblem
from .problem_list.acoustic1d import Acoustic1DProblem
from .problem_list.gaussian_wave1d import GaussianWave1DProblem
from .problem_list.shu_osher1d import ShuOsher1DProblem
from .problem_list.rti import RTIProblem
from .problem_list.hse_1d import HSE1DProblem
from .problem_list.rti_1d import RTI1DProblem
from .problem_list.uniform2d import Uniform2DProblem
from .problem_list.gaussian_wave2d import GaussianWave2DProblem
from .problem_list.gaussian_wave2d_y import GaussianWave2DYProblem
from .problem_list.alfven1d import Alfven1DProblem
from .problem_list.brio_wu1d import BrioWu1DProblem
from .problem_list.adsi2d import ADSI2DProblem
from .problem_list.orszag_tang2d import OrszagTang2DProblem
from .problem_list.orszag_tang3d import OrszagTang3DProblem
from .problem_list.alfven3d import Alfven3DProblem
from .problem_list.mhd_turb3d import MHDTurb3DProblem
from .problem_list.hall_mri1d import HallMRI1DProblem
from .problem_list.adsi_shear import ADSIShearProblem

from .problem_list.dusty_box import DustyBoxProblem
from .problem_list.dustywave1d import DustyWave1DProblem
from .problem_list.dust_beams1d import DustBeams1DProblem
from .problem_list.sgs_acoustic1d import SGSAcoustic1DProblem
from .problem_list.dusty_turb2d import DustyTurb2DProblem
from .problem_list.dusty_turb3d import DustyTurb3DProblem
from .problem_list.charged_dust_box import ChargedDustBoxProblem
from .problem_list.dusty_alfven1d import DustyAlfven1DProblem
from .problem_list.dusty_orszag_tang2d import DustyOrszagTang2DProblem

# Register all problems
ProblemRegistry.register("sod", SodProblem)
ProblemRegistry.register("khi2d", KHI2DProblem)
ProblemRegistry.register("tgv3d", TGV3DProblem)
ProblemRegistry.register("turb3d", Turb3DProblem)
ProblemRegistry.register("acoustic1d", Acoustic1DProblem)
ProblemRegistry.register("gaussian_wave1d", GaussianWave1DProblem)
ProblemRegistry.register("shu_osher1d", ShuOsher1DProblem)
ProblemRegistry.register("rti", RTIProblem)
ProblemRegistry.register("hse_1d", HSE1DProblem)
ProblemRegistry.register("rti_1d", RTI1DProblem)
ProblemRegistry.register("uniform_2d", Uniform2DProblem)
ProblemRegistry.register("gaussian_wave2d", GaussianWave2DProblem)
ProblemRegistry.register("gaussian_wave2d_y", GaussianWave2DYProblem)
ProblemRegistry.register("alfven1d", Alfven1DProblem)
ProblemRegistry.register("brio_wu1d", BrioWu1DProblem)
ProblemRegistry.register("adsi2d", ADSI2DProblem)
ProblemRegistry.register("orszag_tang2d", OrszagTang2DProblem)
ProblemRegistry.register("orszag_tang3d", OrszagTang3DProblem)
ProblemRegistry.register("alfven3d", Alfven3DProblem)
ProblemRegistry.register("mhd_turb3d", MHDTurb3DProblem)
ProblemRegistry.register("hall_mri1d", HallMRI1DProblem)
ProblemRegistry.register("adsi_shear", ADSIShearProblem)

ProblemRegistry.register("dusty_box", DustyBoxProblem)
ProblemRegistry.register("dustywave1d", DustyWave1DProblem)
ProblemRegistry.register("dust_beams1d", DustBeams1DProblem)
ProblemRegistry.register("sgs_acoustic1d", SGSAcoustic1DProblem)
ProblemRegistry.register("dusty_turb2d", DustyTurb2DProblem)
ProblemRegistry.register("dusty_turb3d", DustyTurb3DProblem)
ProblemRegistry.register("charged_dust_box", ChargedDustBoxProblem)
ProblemRegistry.register("dusty_alfven1d", DustyAlfven1DProblem)
ProblemRegistry.register("dusty_orszag_tang2d", DustyOrszagTang2DProblem)
