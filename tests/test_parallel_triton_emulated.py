"""The Triton tier of the distributed solvers, executed on a NumPy emulator.

Triton needs CUDA; ``_triton_emulator`` interprets the kernels on CPU tensors
so that the launch code, strides, index maps and buffer reuse of the GPU tier
are compared with the Torch tier and with the reference solver here.  The
compiler itself is exercised by ``tests/test_parallel_cuda.py`` and
``python -m phrike.parallel.selftest`` on a CUDA machine.
"""

import os

import numpy as np
import pytest

from _parallel_util import COMM, DISSIPATION, fits, gmax, shared_dir, torch
from _triton_emulator import installed
from phrike.equations import EulerEquations3D, MHDEquations3D
from phrike.grid import Grid3D
from phrike.parallel.grid import SlabGrid3D
from phrike.problems.problem_list.orszag_tang3d import orszag_tang_3d
from phrike.solver import SpectralSolver3D, SpectralSolverMHD3D

SHAPE = (16, 12, 10)
TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.skipif(not fits(SHAPE), reason="grid too small for this rank count")


@pytest.fixture(scope="module", autouse=True)
def emulator():
    with installed():
        yield


def _report():
    from phrike.parallel.selftest import Report

    return Report(COMM)


@pytest.mark.parametrize("precision,tight", [("double", 1e-12), ("single", 2e-5)])
def test_new_kernels_match_their_torch_counterparts(precision, tight):
    from phrike.parallel.selftest import check_triton_kernels

    report = _report()
    grid = SlabGrid3D(16, 16, 16, TWO_PI, TWO_PI, TWO_PI, comm=COMM, device="cpu", precision=precision)
    check_triton_kernels(report, grid, tight)
    assert report.failures == 0 and report.checks >= 30


@pytest.mark.parametrize("precision,tight,loose", [("double", 1e-12, 1e-11), ("single", 2e-5, 2e-4)])
def test_triton_tier_matches_torch_tier(precision, tight, loose):
    from phrike.parallel.selftest import check_tiers

    report = _report()
    grid = SlabGrid3D(16, 16, 16, TWO_PI, TWO_PI, TWO_PI, comm=COMM, device="cpu", precision=precision)
    check_tiers(report, grid, tight, loose, triton="force")
    assert report.failures == 0 and report.checks == 10


def test_spherical_truncation_evolves_the_single_precision_boundary_mode():
    from phrike.parallel.incompressible import IncompressibleSlabStepper

    # The (5, 0, 0) mode is exactly on the spherical cutoff on this anisotropic box.
    # Computing kx**2 in float32 rounds it outside the float64-defined cutoff.
    grid = SlabGrid3D(16, 16, 16, 5.0, 4.0, 4.0, comm=COMM, device="cpu", precision="single")
    X, _, _ = grid.xyz_mesh()
    V = torch.stack((torch.ones_like(X), torch.cos(TWO_PI * X), torch.zeros_like(X)))
    ref = IncompressibleSlabStepper(grid, 0.0, mhd=False, spherical_truncation=True, triton=False)
    dev = IncompressibleSlabStepper(grid, 0.0, mhd=False, spherical_truncation=True, triton="force")
    ref.seed(V)
    dev.seed(V)
    start = ref.uhat.clone()
    ref.step(dt=0.05)
    dev.step(dt=0.05)
    assert dev._triton is not None
    assert gmax((ref.uhat - start).abs().max().item()) > 0.1
    assert gmax((dev.uhat - ref.uhat).abs().max().item()) < 2e-6


def _reference(system, dissipation, eta=0.0, nu=0.0, scheme="rk4"):
    Nx, Ny, Nz = SHAPE
    L = 1.0 if system == "mhd" else TWO_PI
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L, dealias=True,
                  filter_params=DISSIPATION if dissipation else None,
                  backend="torch", torch_device="cpu", precision="double")
    if system == "mhd":
        grid.mhd_config = {"resistivity": eta, "viscosity": nu, "divergence_clean": "projection"}
        eqs = MHDEquations3D(gamma=5.0 / 3.0)
        U0 = orszag_tang_3d(*grid.xyz_mesh(), gamma=eqs.gamma, epsilon=0.2)
        g = torch.Generator().manual_seed(0)
        U0[0] = U0[0] * (1.0 + 1e-2 * torch.randn(U0[0].shape, generator=g, dtype=U0.dtype))
        solver = SpectralSolverMHD3D(grid, eqs, spectral_fast_path=False, scheme=scheme)
        start = grid.project_dealiased(solver._project_B(U0))
    else:
        eqs = EulerEquations3D(gamma=1.4)
        z, y, x = np.meshgrid(*(np.arange(n) * TWO_PI / n for n in (Nz, Ny, Nx)), indexing="ij")
        rho = 1.0 + 0.1 * np.cos(x) * np.cos(z)
        ux, uy, uz = np.sin(x) * np.cos(y) * np.cos(z), -np.cos(x) * np.sin(y) * np.cos(z), 0.3 * np.sin(z)
        U0 = torch.as_tensor(eqs.conservative(rho, ux, uy, uz, 10.0 + 0.0 * x))
        solver = SpectralSolver3D(grid, eqs, spectral_fast_path=False, scheme=scheme)
        start = grid.project_dealiased(U0)
    return grid, eqs, solver, U0, start


def _stepper(system, eqs, dissipation, eta=0.0, nu=0.0, triton="force"):
    from phrike.parallel.compressible import CompressibleSlabStepper

    Nx, Ny, Nz = SHAPE
    L = 1.0 if system == "mhd" else TWO_PI
    grid = SlabGrid3D(Nx, Ny, Nz, L, L, L, comm=COMM, device="cpu", precision="double",
                      filter_params=DISSIPATION if dissipation else None)
    st = CompressibleSlabStepper(grid, system, eqs.gamma, 1e-10, 1e-12, eta=eta, nu=nu, triton=triton)
    assert (st._triton is not None) == bool(triton)
    return grid, st


@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("system,dissipation,eta,nu", [
    ("mhd", True, 0.0, 0.0), ("mhd", False, 0.0, 0.0), ("mhd", True, 2e-3, 1e-3),
    ("euler", True, 0.0, 0.0), ("euler", False, 0.0, 0.0),
])
def test_compressible_triton_tier_matches_reference(scheme, system, dissipation, eta, nu):
    """The reused ``spectral3d_triton`` stage kernels on compact slab spectra."""
    ref_grid, eqs, ref, U0, Ur = _reference(system, dissipation, eta, nu, scheme)
    grid, st = _stepper(system, eqs, dissipation, eta, nu)
    d = grid.decomp
    st.seed(U0[:, d.z0:d.z0 + d.nz].clone())
    for _ in range(4):
        Ur = ref.step(Ur, 2e-3)
        st.step(2e-3, scheme)
    assert st._triton is not None and st._stats_kernel is not None  # no silent fallback
    error = gmax((st.physical() - Ur[:, d.z0:d.z0 + d.nz]).abs().max().item()) / float(Ur.abs().max())
    assert error < 1e-11
    assert abs(st.speed - float(eqs.max_wave_speed(Ur))) < 1e-10 * st.speed


@pytest.mark.parametrize("system,dissipation", [("mhd", True), ("mhd", False), ("euler", True), ("euler", False)])
def test_embedded_triton_tier_matches_torch_tier(system, dissipation):
    from phrike.adaptive import create_adaptive_method

    _, eqs, _, U0, _ = _reference(system, dissipation)
    method = create_adaptive_method("rk45")
    results = []
    for triton in ("force", False):
        grid, st = _stepper(system, eqs, dissipation, triton=triton)
        d = grid.decomp
        st.seed(U0[:, d.z0:d.z0 + d.nz].clone())
        error, scale = st.attempt(4e-3, method.a, method.b_high, method.b_low)
        st.commit()
        second, _ = st.attempt(2e-3, method.a, method.b_high, method.b_low)
        st.reject()
        results.append((error, scale, second, st.physical().clone(), st.uhat.clone()))
    (e1, s1, r1, U1, S1), (e2, s2, r2, U2, S2) = results
    assert e1 == pytest.approx(e2, rel=1e-9, abs=1e-15) and s1 == pytest.approx(s2, rel=1e-13)
    assert r1 == pytest.approx(r2, rel=1e-9, abs=1e-15)
    assert gmax((U1 - U2).abs().max().item()) < 1e-12 * gmax(float(U2.abs().max()))
    assert gmax((S1 - S2).abs().max().item()) < 1e-12 * gmax(float(S2.abs().max()))


def test_limiter_fallback_from_the_triton_tier():
    """A floor violation counted by the Triton flux kernel hands the step to the limiter."""
    import warnings

    ref_grid, eqs, ref, U0, _ = _reference("mhd", False)
    U0 = U0.clone()
    U0[4, :3, :3, :3] = -1.0
    U_in = ref_grid.project_dealiased(ref._project_B(U0))
    grid, st = _stepper("mhd", eqs, False)
    d = grid.decomp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st.set_spectrum(grid.forward(U_in[:, d.z0:d.z0 + d.nz].clone(), normalized=True))
        Ur = ref.step(U_in.clone(), 1e-4)
        st.step(1e-4, "rk4")
    assert st._triton is not None and st.events == ref.positivity.events >= 1
    assert gmax((st.physical() - Ur[:, d.z0:d.z0 + d.nz]).abs().max().item()) < 1e-11 * float(Ur.abs().max())


@pytest.mark.parametrize("problem", ["imhd_turb", "mhd_turb3d"])
def test_driver_runs_on_the_triton_tier(tmp_path, problem):
    from phrike.parallel.driver import Simulation

    outdir = shared_dir(tmp_path)
    finals = []
    for tier in ("force", False):
        config = {
            "grid": {"Nx": 16, "Ny": 12, "Nz": 10, "Lx": TWO_PI, "Ly": TWO_PI, "Lz": TWO_PI},
            "physics": {"gamma": 5.0 / 3.0, "viscosity": 0.02, "resistivity": 0.03},
            "integration": {"t_end": 0.05, "cfl": 0.3, "scheme": "rk4", "output_interval": 1.0},
            "initial_conditions": {"vrms": 0.3, "kmin": 1.0, "kmax": 3.0, "b_rms": 0.1, "b_kmax": 3.0},
            "forcing": {"type": "ou", "kmax": 2.0, "amplitude": 1.0, "correlation_time": 0.1},
            "io": {"outdir": outdir, "snapshots": False, "slices": False, "spectra": False},
            "monitoring": {"enabled": False}, "parallel": {"triton": tier},
        }
        sim = Simulation(problem, config, comm=COMM, backend="cpu", quiet=True)
        sim.run()
        assert (sim.stepper._triton is not None) == bool(tier)
        finals.append(sim.stepper.uhat.clone())
    assert gmax((finals[0] - finals[1]).abs().max().item()) < 1e-12 * gmax(float(finals[1].abs().max()))


def test_triton_compiler_guard(monkeypatch):
    """NVHPC modules set CC=nvc, which cannot build Triton's helper."""
    import shutil

    from phrike._triton_cc import ensure_triton_compiler

    usable = shutil.which("gcc") or shutil.which("clang")
    monkeypatch.delenv("PHRIKE_TRITON_CC", raising=False)
    monkeypatch.setenv("CC", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvc")
    ensure_triton_compiler()
    if usable:
        assert os.environ["CC"] == usable
    monkeypatch.setenv("CC", "/usr/bin/gcc-13")
    ensure_triton_compiler()
    assert os.environ["CC"] == "/usr/bin/gcc-13"  # a usable compiler is left alone
    monkeypatch.delenv("CC")
    ensure_triton_compiler()
    assert "CC" not in os.environ
    monkeypatch.setenv("PHRIKE_TRITON_CC", "my-cc")
    ensure_triton_compiler()
    assert os.environ["CC"] == "my-cc"
