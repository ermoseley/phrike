"""The slab-decomposed compressible step must be the reference solver's step.

With one rank this compares the compact-spectrum stepper with
``SpectralSolverMHD3D`` / ``SpectralSolver3D`` (reference path); under MPI each
rank compares its slab with the same single-process reference.
"""

import warnings

import numpy as np
import pytest

from _parallel_util import COMM, DISSIPATION, fits, gmax, shared_dir, torch
from phrike.equations import EulerEquations3D, MHDEquations3D
from phrike.grid import Grid3D
from phrike.parallel.compressible import CompressibleSlabStepper
from phrike.parallel.driver import Simulation
from phrike.parallel.grid import SlabGrid3D
from phrike.problems.problem_list.orszag_tang3d import orszag_tang_3d
from phrike.solver import SpectralSolver3D, SpectralSolverMHD3D

SHAPES = [(16, 12, 10), (15, 9, 11), (20, 18, 14)]


def _euler_state(shape, noise=1e-2):
    Nx, Ny, Nz = shape
    z, y, x = np.meshgrid(*(np.arange(n) * 2 * np.pi / n for n in (Nz, Ny, Nx)), indexing="ij")
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.1 * np.cos(x) * np.cos(z) + noise * rng.standard_normal(x.shape)
    ux = np.sin(x) * np.cos(y) * np.cos(z)
    uy = -np.cos(x) * np.sin(y) * np.cos(z)
    uz = 0.3 * np.sin(z) * np.cos(y)
    p = 10.0 + noise * rng.standard_normal(x.shape)
    return torch.as_tensor(EulerEquations3D(gamma=1.4).conservative(rho, ux, uy, uz, p))


def _reference(system, shape, dissipation, eta=0.0, nu=0.0, noise=1e-2, **solver_kwargs):
    """(reference grid, equations, solver, initial state) on the whole grid."""
    Nx, Ny, Nz = shape
    L = 1.0 if system == "mhd" else 2 * np.pi
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L, dealias=True,
                  filter_params=DISSIPATION if dissipation else None,
                  backend="torch", torch_device="cpu", precision="double")
    if system == "mhd":
        grid.mhd_config = {"resistivity": eta, "viscosity": nu, "divergence_clean": "projection"}
        eqs = MHDEquations3D(gamma=5.0 / 3.0)
        U0 = orszag_tang_3d(*grid.xyz_mesh(), gamma=eqs.gamma, epsilon=0.2)
        if noise:
            g = torch.Generator().manual_seed(0)
            U0[0] = U0[0] * (1.0 + noise * torch.randn(U0[0].shape, generator=g, dtype=U0.dtype))
        solver = SpectralSolverMHD3D(grid, eqs, spectral_fast_path=False, **solver_kwargs)
    else:
        eqs = EulerEquations3D(gamma=1.4)
        U0 = _euler_state(shape, noise)
        solver = SpectralSolver3D(grid, eqs, spectral_fast_path=False, **solver_kwargs)
    return grid, eqs, solver, U0


def _distributed(system, shape, eqs, dissipation, eta=0.0, nu=0.0, **kwargs):
    Nx, Ny, Nz = shape
    L = 1.0 if system == "mhd" else 2 * np.pi
    grid = SlabGrid3D(Nx, Ny, Nz, L, L, L, comm=COMM, device="cpu", precision="double",
                      filter_params=DISSIPATION if dissipation else None, fft_batch=3)
    return grid, CompressibleSlabStepper(grid, system, eqs.gamma, 1e-10, 1e-12, eta=eta, nu=nu, **kwargs)


def _local(U, grid):
    d = grid.decomp
    return U[:, d.z0:d.z0 + d.nz]


def _rel(A, B):
    return gmax((A - B).abs().max().item()) / float(B.abs().max().item())


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("system,dissipation,eta,nu", [
    ("mhd", True, 0.0, 0.0), ("mhd", False, 0.0, 0.0), ("mhd", True, 2e-3, 1e-3),
    ("euler", True, 0.0, 0.0), ("euler", False, 0.0, 0.0),
])
def test_fixed_step_matches_reference(shape, scheme, system, dissipation, eta, nu):
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    ref_grid, eqs, ref, U0 = _reference(system, shape, dissipation, eta, nu, scheme=scheme)
    Ur = ref_grid.project_dealiased(ref._project_B(U0) if system == "mhd" else U0)
    grid, st = _distributed(system, shape, eqs, dissipation, eta, nu)
    assert _rel(st.seed(_local(U0, grid).clone()), _local(Ur, grid)) < 1e-12
    for _ in range(4):
        Ur = ref.step(Ur, 2e-3)
        st.step(2e-3, scheme)
    assert _rel(st.physical(), _local(Ur, grid)) < 1e-11
    assert st.band_limited and st.events == 0
    assert abs(st.speed - float(eqs.max_wave_speed(Ur))) < 1e-10 * st.speed
    if system == "mhd":
        assert st.max_div_b() < 1e-11


@pytest.mark.parametrize("system", ["mhd", "euler"])
def test_strict_clamp_changes_nothing_without_violations(system):
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    _, eqs, _, U0 = _reference(system, shape, True)
    states = []
    for strict in (False, True):
        grid, st = _distributed(system, shape, eqs, True, strict_clamp=strict)
        st.seed(_local(U0, grid).clone())
        st.step(2e-3, "rk4")
        states.append(st.physical())
    assert torch.equal(states[0], states[1])


@pytest.mark.parametrize("system", ["mhd", "euler"])
def test_positivity_limiter_matches_reference(system):
    """A state that needs the limiter takes the reference's limited step."""
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    ref_grid, eqs, ref, U0 = _reference(system, shape, False, noise=0.0)
    U0 = U0.clone()
    U0[4, :3, :3, :3] = -1.0  # negative thermal pressure: the limiter must act
    U_in = ref_grid.project_dealiased(ref._project_B(U0) if system == "mhd" else U0)
    grid, st = _distributed(system, shape, eqs, False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st.set_spectrum(grid.forward(_local(U_in, grid).clone(), normalized=True))
        Ur = ref.step(U_in.clone(), 1e-4)
        st.step(1e-4, "rk4")
    assert _rel(st.physical(), _local(Ur, grid)) < 1e-11
    assert st.events == ref.positivity.events >= 1
    assert st.cells == ref.positivity.cells


def _adaptive_config(tmp_path, problem, shape, cfl, extra_ic=None):
    Nx, Ny, Nz = shape
    L = 1.0 if problem == "orszag_tang3d" else 2 * np.pi
    return {
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "Lx": L, "Ly": L, "Lz": L},
        "physics": {"gamma": 5.0 / 3.0 if problem == "orszag_tang3d" else 1.4},
        "integration": {"t0": 0.0, "t_end": 0.03, "cfl": cfl, "scheme": "rk4", "output_interval": 1.0},
        "adaptive": {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12},
        "initial_conditions": dict({"epsilon": 0.2}, **(extra_ic or {})),
        "io": {"outdir": shared_dir(tmp_path), "snapshots": False, "slices": False, "spectra": False},
        "monitoring": {"enabled": False},
    }


def _record(stepper):
    flags = []
    step = stepper.step

    def recording(*args, **kwargs):
        result = step(*args, **kwargs)
        flags.append(result.accepted)
        return result

    stepper.step = recording
    return flags


@pytest.mark.parametrize("dissipation", [False, True])
def test_adaptive_mhd_run_matches_reference(tmp_path, dissipation):
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    adaptive = {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12}
    ref_grid, eqs, ref, U0 = _reference("mhd", shape, dissipation, noise=0.0, adaptive_config=adaptive)
    ref_flags = _record(ref.adaptive_stepper)
    ref.run(U0, t0=0.0, t_end=0.03, output_interval=1.0)

    config = _adaptive_config(tmp_path, "orszag_tang3d", shape, cfl=0.3)
    if dissipation:
        config["integration"]["spectral_dissipation"] = DISSIPATION
    sim = Simulation("orszag_tang3d", config, comm=COMM, backend="cpu", quiet=True)
    flags = _record(sim.adaptive)
    sim.run()
    assert flags == ref_flags and len(flags) > 1
    assert sim.t == pytest.approx(ref.t, rel=1e-12)
    assert _rel(sim.stepper.physical(), _local(ref.U, sim.grid)) < 1e-9
    assert sim.stepper.max_div_b() < 1e-11


def test_adaptive_euler_run_matches_reference(tmp_path):
    shape = (12, 12, 12)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    from phrike.problems.problem_list.tgv3d import taylor_green_vortex_3d

    adaptive = {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12}
    L = 2 * np.pi
    ref_grid = Grid3D(Nx=12, Ny=12, Nz=12, Lx=L, Ly=L, Lz=L, dealias=True, backend="torch",
                      torch_device="cpu", precision="double")
    eqs = EulerEquations3D(gamma=1.4)
    p0 = 1.0 / (1.4 * 0.3 ** 2)
    U0 = taylor_green_vortex_3d(*ref_grid.xyz_mesh(), rho0=1.0, p0=p0, U0=1.0, k=1, gamma=1.4)
    ref = SpectralSolver3D(ref_grid, eqs, spectral_fast_path=False, adaptive_config=adaptive, cfl=0.25)
    ref_flags = _record(ref.adaptive_stepper)
    ref.run(U0, t0=0.0, t_end=0.03, output_interval=1.0)

    config = _adaptive_config(tmp_path, "tgv3d", (12, 12, 12), cfl=0.25, extra_ic={"mach": 0.3})
    sim = Simulation("tgv3d", config, comm=COMM, backend="cpu", quiet=True)
    flags = _record(sim.adaptive)
    sim.run()
    assert flags == ref_flags and len(flags) > 1
    assert sim.t == pytest.approx(ref.t, rel=1e-12)
    assert _rel(sim.stepper.physical(), _local(ref.U, sim.grid)) < 1e-9


def test_rejected_attempt_restores_the_state():
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    from phrike.adaptive import create_adaptive_method

    _, eqs, _, U0 = _reference("mhd", shape, True)
    grid, st = _distributed("mhd", shape, eqs, True)
    st.seed(_local(U0, grid).clone())
    before, U_before = st.uhat.clone(), st.physical().clone()
    method = create_adaptive_method("rk45")
    error, scale = st.attempt(5e-3, method.a, method.b_high, method.b_low)
    assert error > 0.0 and scale > 0.0
    st.reject()
    assert torch.equal(st.uhat, before)
    assert _rel(st.physical(), U_before) < 1e-13
    st.attempt(5e-3, method.a, method.b_high, method.b_low)
    st.commit()
    assert not torch.equal(st.uhat, before)


def test_kick_adds_the_kinetic_energy_change():
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    _, eqs, _, U0 = _reference("mhd", shape, False, noise=0.0)
    grid, st = _distributed("mhd", shape, eqs, False)
    st.seed(_local(U0, grid).clone())
    before = st.diagnostics()
    X, Y, Z = grid.xyz_mesh()
    dv = 0.05 * torch.stack([torch.sin(2 * np.pi * Y), torch.sin(2 * np.pi * Z), torch.sin(2 * np.pi * X)])
    U = st.physical()
    # Mean energy added: <m . dv + rho dv^2 / 2> (the dealias projection keeps the mean).
    expected = COMM.sum(float(((U[1:4] * dv).sum(dim=0) + 0.5 * U[0] * (dv * dv).sum(dim=0)).sum())) / grid.n_points
    st.kick(dv)
    after = st.diagnostics()
    assert after["mass"] == pytest.approx(before["mass"], rel=1e-13)
    assert after["magnetic_energy"] == pytest.approx(before["magnetic_energy"], rel=1e-13)
    assert after["energy"] - before["energy"] == pytest.approx(expected, rel=1e-11)
    assert (after["kinetic_energy"] - before["kinetic_energy"]) == pytest.approx(expected, rel=1e-3)
    # The state stays band-limited and equal to its stored spectrum.
    assert _rel(grid.inverse(st.uhat), st.physical()) < 1e-13
