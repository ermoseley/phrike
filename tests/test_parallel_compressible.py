"""The slab-decomposed compressible step must be the reference solver's step.

With one rank this compares the compact-spectrum stepper with
``SpectralSolverMHD3D`` / ``SpectralSolver3D`` (reference path); under MPI each
rank compares its slab with the same single-process reference.
"""

import os
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
from phrike.transport import Transport

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


def _reference(system, shape, dissipation, transport=Transport(), noise=1e-2, **solver_kwargs):
    """(reference grid, equations, solver, initial state) on the whole grid."""
    Nx, Ny, Nz = shape
    L = 1.0 if system == "mhd" else 2 * np.pi
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L, dealias=True,
                  filter_params=DISSIPATION if dissipation else None,
                  backend="torch", torch_device="cpu", precision="double")
    if system == "mhd":
        grid.mhd_config = dict(transport.as_dict(), divergence_clean="projection")
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


def _distributed(system, shape, eqs, dissipation, transport=Transport(), **kwargs):
    Nx, Ny, Nz = shape
    L = 1.0 if system == "mhd" else 2 * np.pi
    grid = SlabGrid3D(Nx, Ny, Nz, L, L, L, comm=COMM, device="cpu", precision="double",
                      filter_params=DISSIPATION if dissipation else None, fft_batch=3)
    return grid, CompressibleSlabStepper(grid, system, eqs.gamma, 1e-10, 1e-12, transport=transport, **kwargs)


def _local(U, grid):
    d = grid.decomp
    return U[:, d.z0:d.z0 + d.nz]


def _rel(A, B):
    return gmax((A - B).abs().max().item()) / float(B.abs().max().item())


TRANSPORTS = [Transport(nu=5e-3, eta=4e-3), Transport(nu=5e-3), Transport(eta=4e-3),
              Transport(nu=5e-3, nu_bulk=3e-3, eta=4e-3, kappa=6e-3), Transport(nu_bulk=3e-3),
              Transport(kappa=6e-3)]
FULL = TRANSPORTS[3]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("system,dissipation,transport", [
    ("mhd", True, Transport()), ("mhd", False, Transport()), ("euler", True, Transport()),
    ("euler", False, Transport())] + [("mhd", True, tr) for tr in TRANSPORTS],
    ids=lambda v: v.describe() if isinstance(v, Transport) else str(v))
def test_fixed_step_matches_reference(shape, scheme, system, dissipation, transport):
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    ref_grid, eqs, ref, U0 = _reference(system, shape, dissipation, transport, scheme=scheme)
    Ur = ref_grid.project_dealiased(ref._project_B(U0) if system == "mhd" else U0)
    grid, st = _distributed(system, shape, eqs, dissipation, transport)
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


@pytest.mark.parametrize("system,transport", [("mhd", Transport()), ("mhd", FULL), ("euler", Transport())])
def test_positivity_limiter_matches_reference(system, transport):
    """A state that needs the limiter takes the reference's limited step."""
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    ref_grid, eqs, ref, U0 = _reference(system, shape, False, transport, noise=0.0)
    U0 = U0.clone()
    U0[4, :3, :3, :3] = -1.0  # negative thermal pressure: the limiter must act
    U_in = ref_grid.project_dealiased(ref._project_B(U0) if system == "mhd" else U0)
    grid, st = _distributed(system, shape, eqs, False, transport)
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


@pytest.mark.parametrize("dissipation,transport", [(False, Transport()), (True, Transport()), (True, FULL)])
def test_adaptive_mhd_run_matches_reference(tmp_path, dissipation, transport):
    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    adaptive = {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12}
    ref_grid, eqs, ref, U0 = _reference("mhd", shape, dissipation, transport, noise=0.0,
                                        adaptive_config=adaptive)
    ref_flags = _record(ref.adaptive_stepper)
    ref.run(U0, t0=0.0, t_end=0.03, output_interval=1.0)

    config = _adaptive_config(tmp_path, "orszag_tang3d", shape, cfl=0.3)
    if dissipation:
        config["integration"]["spectral_dissipation"] = DISSIPATION
    if transport.active:
        config["physics"]["mhd"] = transport.as_dict()
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


# --- the state after an accepted-state clamp ------------------------------------


def _clamped_start(dissipation, scheme, patch=0.0, extent=3, transport=Transport()):
    """Reference solver, its start state and a stepper seeded with it; a hot-spot of low pressure
    makes the limiter act."""
    shape = (16, 12, 10)
    ref_grid, eqs, ref, U0 = _reference("mhd", shape, dissipation, transport, noise=0.0, scheme=scheme)
    U0 = U0.clone()
    p = (eqs.gamma - 1.0) * (U0[4] - 0.5 * (U0[1:4] ** 2).sum(0) / U0[0] - 0.5 * (U0[5:8] ** 2).sum(0))
    e = extent
    U0[4, :e, :e, :e] -= (p[:e, :e, :e] - patch) / (eqs.gamma - 1.0)
    U_in = ref_grid.project_dealiased(ref._project_B(U0))
    grid, st = _distributed("mhd", shape, eqs, dissipation, transport)
    st.set_spectrum(grid.forward(_local(U_in, grid).clone(), normalized=True))
    return ref_grid, eqs, ref, U_in, grid, st


@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("dissipation", [False, True])
@pytest.mark.parametrize("transport", [Transport(), FULL], ids=["ideal", "transport"])
def test_steps_after_an_accepted_state_clamp_match_reference(scheme, dissipation, transport):
    """The clamped state keeps its out-of-band part for one more step, as in the reference."""
    if not fits((16, 12, 10)):
        pytest.skip("grid too small for this rank count")
    _, _, ref, Ur, grid, st = _clamped_start(dissipation, scheme, transport=transport)
    worst, residual_steps = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(10):
            Ur = ref.step(Ur, 3e-3)
            st.step(3e-3, scheme)
            worst = max(worst, _rel(st.physical(), _local(Ur, grid)))
            residual_steps += not st.band_limited
    assert residual_steps > 0 and worst < 1e-12
    assert st.events == ref.positivity.events and st.cells == ref.positivity.cells
    if not st.band_limited:  # the residual is exactly what the retained modes cannot hold
        assert _rel(grid.inverse(st.uhat) + torch.nn.functional.pad(st.residual, (0, 0, 0, 0, 0, 0, 0, 3)),
                    st.physical()) < 1e-13


def test_residual_is_dropped_once_the_limiter_stops_acting():
    if not fits((16, 12, 10)):
        pytest.skip("grid too small for this rank count")
    _, _, _, _, grid, st = _clamped_start(True, "rk4")
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(10):
            st.step(3e-3, "rk4")
            seen.append(st.band_limited)
    assert not seen[0] and seen[-1]  # clamps at first, recovers (dissipation smooths the hot-spot)
    assert grid._full is None  # the full-spectrum transform is freed with the residual


def _reference_embedded(ref, U, dt, method):
    """The reference's slow-path adaptive attempt at a fixed dt (``_adaptive_attempt``)."""
    h = 0.5 * dt
    Us = ref.grid.apply_spectral_dissipation(U, h)
    ks = []
    for i in range(len(method.b_high)):
        S = Us + sum(dt * float(method.a[i][j]) * ks[j] for j in range(i) if method.a[i][j] != 0.0) if i else Us
        ks.append(ref._compute_rhs(S))
    high = Us + sum(dt * float(method.b_high[i]) * ks[i] for i in range(len(ks)) if method.b_high[i] != 0.0)
    Un = ref._project_B(ref._clamp(high))
    return ref._limit_accepted(ref.grid.apply_accepted_state_stabilization(Un, h))


@pytest.mark.parametrize("dissipation", [False, True])
@pytest.mark.parametrize("transport", [Transport(), FULL], ids=["ideal", "transport"])
def test_embedded_steps_and_rejects_after_a_clamp_match_reference(dissipation, transport):
    if not fits((16, 12, 10)):
        pytest.skip("grid too small for this rank count")
    from phrike.adaptive import create_adaptive_method

    method = create_adaptive_method("rk45")
    _, _, ref, Ur, grid, st = _clamped_start(dissipation, "rk4", transport=transport)
    worst, residual_steps = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for n in range(8):
            if n >= 1 and not st.band_limited:  # reject an attempt made on a state that has a residual
                before, U_before, R_before = st.uhat.clone(), st.physical().clone(), st.residual.clone()
                st.attempt(7e-3, method.a, method.b_high, method.b_low)
                st.reject()
                assert torch.equal(st.uhat, before) and torch.equal(st.residual, R_before)
                assert _rel(st.physical(), U_before) < 1e-13
            Ur = _reference_embedded(ref, Ur, 3e-3, method)
            st.attempt(3e-3, method.a, method.b_high, method.b_low)
            st.commit()
            worst = max(worst, _rel(st.physical(), _local(Ur, grid)))
            residual_steps += not st.band_limited
    assert residual_steps > 0 and worst < 1e-12


def test_forcing_kick_carries_the_residual():
    if not fits((16, 12, 10)):
        pytest.skip("grid too small for this rank count")
    _, _, _, _, grid, st = _clamped_start(False, "rk4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st.step(3e-3, "rk4")
    assert not st.band_limited
    R = st.residual.clone()
    X, Y, Z = grid.xyz_mesh()
    st.kick(0.01 * torch.stack([torch.sin(2 * np.pi * Y), torch.sin(2 * np.pi * Z), torch.sin(2 * np.pi * X)]))
    assert torch.equal(st.residual, R)
    full = grid.inverse(st.uhat)
    full[:5] += R
    assert _rel(st.physical()[:5], full[:5]) < 1e-13


def test_restart_after_a_clamp_reproduces_the_run(tmp_path):
    """A snapshot with the residual restarts exactly on the same rank count; otherwise it is dropped."""
    from phrike.parallel.io import read_meta, read_residual, read_spectrum, write_snapshot

    if not fits((16, 12, 10)):
        pytest.skip("grid too small for this rank count")
    _, eqs, _, _, grid, st = _clamped_start(True, "rk4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st.step(3e-3, "rk4")
        assert not st.band_limited
        path = os.path.join(shared_dir(tmp_path), "snap")
        write_snapshot(path, grid, "mhd", st.uhat, 0.003, 1, residual=st.residual)
        meta = COMM.bcast(read_meta(path) if COMM.is_root else None)  # rank 0 writes meta.json last
        grid2, st2 = _distributed("mhd", (16, 12, 10), eqs, True)
        st2.set_spectrum(read_spectrum(path, grid2, "mhd", meta))
        st2.set_residual(read_residual(path, grid2, "mhd", meta))
        assert not st2.band_limited and st2.speed == pytest.approx(st.speed, rel=1e-13)
        for _ in range(3):
            st.step(3e-3, "rk4")
            st2.step(3e-3, "rk4")
        assert _rel(st2.physical(), st.physical()) < 1e-12
        assert gmax((st2.uhat - st.uhat).abs().max().item()) < 1e-12 * gmax(float(st.uhat.abs().max()))
        # Another rank count (or grid): the residual is dropped with a warning.
        changed = dict(meta, residual=dict(meta["residual"], nprocs=COMM.size + 1))
        with pytest.warns(RuntimeWarning, match="dropped") if COMM.is_root else warnings.catch_warnings():
            assert read_residual(path, grid2, "mhd", changed) is None
        # A snapshot written without a residual has none to read.
        write_snapshot(path, grid, "mhd", st.uhat, 0.01, 2)
        COMM.barrier()  # rank 0 writes meta.json after the others have returned
        assert read_residual(path, grid2, "mhd") is None
        assert not os.path.exists(os.path.join(path, f"resid.{COMM.rank:05d}.npy"))


def test_strict_clamp_is_the_default():
    shape = (16, 12, 10)
    _, eqs, _, _ = _reference("mhd", shape, True)
    _, st = _distributed("mhd", shape, eqs, True)
    assert st.strict_clamp is True


# --- a Triton failure in the middle of a step must not hang the job -------------


def _mpi_launcher():
    import shutil
    import subprocess
    import sys

    from phrike.parallel.comm import launched_under_mpi

    if launched_under_mpi():
        return None
    exe = shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None:
        return None
    try:
        version = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout
        base = [exe] + (["--oversubscribe"] if "Open MPI" in version or "OpenRTE" in version else [])
        probe = subprocess.run(base + ["-n", "2", sys.executable, "-c",
                                       "from mpi4py import MPI; assert MPI.COMM_WORLD.Get_size() == 2"],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return base if probe.returncode == 0 else None


_FAILING_KERNEL_JOB = '''
import sys, types
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
import numpy as np, torch
torch.set_num_threads(1)
from _triton_emulator import installed
from phrike.parallel import Comm
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.compressible import CompressibleSlabStepper
comm = Comm.auto()
comm.install_abort_hook()
with installed():
    g = SlabGrid3D(12, 12, 12, 2 * np.pi, 2 * np.pi, 2 * np.pi, comm=comm)
    st = CompressibleSlabStepper(g, "mhd", 5 / 3, 1e-10, 1e-12, triton="force")
    X, Y, Z = g.xyz_mesh()
    rho = 1 + 0.1 * torch.cos(X); B = [0.1 * torch.cos(Y), 0.1 * torch.cos(Z), 0.1 * torch.cos(X)]
    E = 1 / (2 / 3) + 0.5 * sum(b * b for b in B)
    st.seed(torch.stack([rho, 0 * X, 0 * X, 0 * X, E] + B))
    assert st.warm_up() and st._strict_kernels
    orig = st._triton.mhd_stage
    calls = [0]
    def flaky(*a, **k):
        calls[0] += 1
        if comm.rank == 1 and calls[0] == 2:
            raise RuntimeError("simulated kernel failure on rank 1, stage 2")
        return orig(*a, **k)
    mod = types.SimpleNamespace(**{{n: getattr(st._triton, n) for n in dir(st._triton) if not n.startswith("__")}})
    mod.mhd_stage = flaky
    st._triton = mod
    st.step(1e-3, "rk4")
    print("STEP RETURNED", flush=True)
'''


def test_kernel_failure_after_warm_up_aborts_instead_of_hanging(tmp_path):
    import subprocess
    import sys

    launcher = _mpi_launcher() if COMM.size == 1 else None
    if launcher is None:
        pytest.skip("needs mpirun and mpi4py, outside an MPI launch")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "failing_kernel_job.py"
    script.write_text(_FAILING_KERNEL_JOB.format(root=root, tests=os.path.join(root, "tests")))
    env = dict(os.environ, PHRIKE_MPI="1", OMP_NUM_THREADS="1")
    done = subprocess.run(launcher + ["-n", "2", sys.executable, str(script)], env=env, cwd=root,
                          capture_output=True, text=True, timeout=180)  # a hang raises TimeoutExpired
    assert done.returncode != 0
    assert "simulated kernel failure" in done.stderr and "STEP RETURNED" not in done.stdout


def test_warm_up_runs_on_the_real_shapes_and_makes_kernels_strict():
    """Warm-up leaves the stepper's state alone and later kernel failures are not swallowed."""
    from _triton_emulator import installed

    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    with installed():
        _, eqs, _, U0 = _reference("mhd", shape, True)
        grid, st = _distributed("mhd", shape, eqs, True, triton="force")
        assert not st._strict_kernels
        st.seed(_local(U0, grid).clone())
        before, U_before = st.uhat.clone(), st.physical().clone()
        events, counters = st.events, (grid.fft.forward_fields, grid.fft.inverse_fields)
        assert st.warm_up(embedded=True)
        assert st._strict_kernels
        assert torch.equal(st.uhat, before) and torch.equal(st.physical(), U_before)
        assert st.events == events and st._acc is None and st._kbuf is None
        assert (grid.fft.forward_fields, grid.fft.inverse_fields) == counters
        if COMM.size == 1:  # with more ranks a kernel failure aborts the job (not testable here)
            boom = RuntimeError("kernel failed")
            with pytest.raises(RuntimeError, match="kernel failed"):
                st._disable_triton(boom)
            assert st._triton is not None


def test_circular_modes_decay_exactly_with_transport():
    """Parallel circular u and B modes along z are an exact decaying solution with uniform heating."""
    shape = (8, 8, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    tr = Transport(nu=0.03, nu_bulk=0.02, eta=0.05, kappa=0.04)
    gamma, U0, B0, p0 = 5.0 / 3.0, 0.2, 0.3, 1.0
    grid = SlabGrid3D(*shape, 2 * np.pi, 2 * np.pi, 2 * np.pi, comm=COMM, device="cpu", precision="double")
    st = CompressibleSlabStepper(grid, "mhd", gamma, 1e-10, 1e-12, transport=tr)
    _, _, Z = grid.xyz_mesh()
    one = torch.ones_like(Z)
    eqs = MHDEquations3D(gamma=gamma)
    st.seed(eqs.conservative(one, U0 * torch.cos(Z), U0 * torch.sin(Z), 0 * one, p0 * one,
                             B0 * torch.cos(Z), B0 * torch.sin(Z), 0 * one))
    for _ in range(50):
        st.step(0.01, "rk4")
    fu, fb = np.exp(-tr.nu * 0.5), np.exp(-tr.eta * 0.5)
    p = p0 + (gamma - 1.0) * (0.5 * U0 ** 2 * (1.0 - fu * fu) + 0.5 * B0 ** 2 * (1.0 - fb * fb))
    rho, ux, uy, uz, pn, Bx, By, Bz = eqs.primitive(st.physical())
    worst = max(float((rho - 1.0).abs().max()), float((ux - U0 * fu * torch.cos(Z)).abs().max()),
                float((uy - U0 * fu * torch.sin(Z)).abs().max()), float((Bx - B0 * fb * torch.cos(Z)).abs().max()),
                float((By - B0 * fb * torch.sin(Z)).abs().max()), float((pn - p).abs().max()))
    assert gmax(worst) < 1e-10
    rates = st.dissipation_rates()  # collective
    assert rates["viscous_dissipation"] == pytest.approx(U0 ** 2 * fu * fu * tr.nu, rel=1e-9)
    assert rates["resistive_dissipation"] == pytest.approx(B0 ** 2 * fb * fb * tr.eta, rel=1e-9)
