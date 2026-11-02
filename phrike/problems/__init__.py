"""Problem registry and base classes for PHRIKE."""

from .base import BaseProblem
from .dusty_base import DustyProblemBase
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
from .problem_list.hse_1d import HSE1DProblem
from .problem_list.rti_1d import RTI1DProblem
from .problem_list.uniform2d import Uniform2DProblem
from .problem_list.orszag_tang3d import OrszagTang3DProblem
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

__all__ = [
    "DustyProblemBase",
    "DustyBoxProblem",
    "DustyWave1DProblem",
    "DustBeams1DProblem",
    "SGSAcoustic1DProblem",
    "DustyTurb2DProblem",
    "DustyTurb3DProblem",
    "ChargedDustBoxProblem",
    "DustyAlfven1DProblem",
    "DustyOrszagTang2DProblem",
    "BaseProblem", 
    "ProblemRegistry",
    "SodProblem",
    "KHI2DProblem", 
    "TGV3DProblem",
    "Turb3DProblem",
    "Acoustic1DProblem",
    "ShuOsher1DProblem",
    "GaussianWave1DProblem",
    "RTIProblem",
    "HSE1DProblem",
    "RTI1DProblem",
    "Uniform2DProblem",
    "OrszagTang3DProblem",
    "HallMRI1DProblem",
    "ADSIShearProblem",
]
