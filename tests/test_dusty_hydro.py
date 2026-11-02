"""Coupled solver contracts, conservation and small problem runs."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from phrike.dust_moments import GaussianClosure, HyqmomClosure
from phrike.dust_sources import epstein_nu
from phrike.dusty_hydro import (
    DustyHydroSolver, SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D,
)
from phrike.equations import EulerEquations1D, EulerEquations2D, EulerEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.io import load_config
from phrike.problems.problem_list.dusty_box import dust_state
from phrike.problems.problem_list.dusty_turb2d import turbulent_velocity_2d
from phrike.problems.registry import ProblemRegistry
from phrike.sgs import sgs_coefficients
from phrike.solver import SpectralSolver1D, SpectralSolver2D, SpectralSolver3D
import phrike.problems.register  # noqa: F401


BASE = (EulerEquations1D, EulerEquations2D, EulerEquations3D)
SGS = (SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D)
SOLVER = (SpectralSolver1D, SpectralSolver2D, SpectralSolver3D)


def grid_for(ndim, filtering=True):
    options = dict(backend="numpy", fft_workers=1, filter_params={"spectral_dissipation":
                   dict(enabled=filtering, p=2, rate=4.0, onset_fraction=0.0)})
    return (Grid1D(12, 1.0, **options) if ndim == 1 else
            Grid2D(12, 10, 1.0, 1.2, **options) if ndim == 2 else
            Grid3D(8, 6, 4, 1.0, 1.2, 0.9, **options))


def gas_state(grid, ndim):
    shape = (grid.N,) if ndim == 1 else (grid.Ny, grid.Nx) if ndim == 2 else (grid.Nz, grid.Ny, grid.Nx)
    phases = np.indices(shape)/np.array(shape).reshape((ndim,) + (1,)*ndim)
    rho = 1 + 0.03*np.cos(2*np.pi*phases[-1])
    p = np.full(shape, 2.0)
    u = [0.1 + 0.05*np.sin(2*np.pi*phases[-1-d]) for d in range(ndim)]
    return BASE[ndim-1](1.4).conservative(rho, *u, p)


def coupled(ndim=2, closure="gaussian", sgs_on=True, dust_on=True):
    grid = grid_for(ndim)
    solver = DustyHydroSolver(grid, SGS[ndim-1](1.4), cfl=0.3,
        dust_config=dict(enabled=dust_on, closure=closure, stopping={"model":"constant", "t_s":0.2},
                         eddy_diffusion={"enabled": True, "c_kappa":0.2}),
        sgs_config=dict(enabled=sgs_on, diffusion={"enabled":True,"c_kappa":0.1}))
    solver.U_gas = gas_state(grid, ndim)
    if dust_on:
        rho = np.full(solver.shape, 0.15)
        u = np.zeros((3, *solver.shape))
        u[0] = 0.2
        solver.U_dust = dust_state(solver.closure, rho, u, 0.1)
    if sgs_on:
        solver.e_s[:] = 0.1
    return solver


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_sgs_equations(ndim):
    grid = grid_for(ndim)
    U = gas_state(grid, ndim)
    original = U.copy()
    base, eq = BASE[ndim-1](1.4), SGS[ndim-1](1.4)
    bf, sf = base.flux(U), eq.flux(U)
    for f, g in zip((bf,) if ndim == 1 else bf, (sf,) if ndim == 1 else sf):
        np.testing.assert_array_equal(f, g)
    assert base.max_wave_speed(U) == eq.max_wave_speed(U)
    eq.p_sgs = 0.1 + 0.01*np.cos(2*np.pi*np.arange(U.shape[-1])/U.shape[-1])
    sf = eq.flux(U)
    for d, (f, g) in enumerate(zip((bf,) if ndim == 1 else bf, (sf,) if ndim == 1 else sf)):
        expected = f.copy()
        expected[1+d] += eq.p_sgs
        expected[-1] += eq.p_sgs*U[1+d]/U[0]
        np.testing.assert_array_equal(g, expected)
    pressure = base.primitive(U)[2 if ndim == 1 else -1]
    expected = np.max(np.max(np.abs(U[1:1+ndim]/U[0]), axis=0)
                      + np.sqrt((1.4*pressure + (5/3)*eq.p_sgs)/U[0]))
    assert eq.max_wave_speed(U) == expected
    np.testing.assert_array_equal(U, original)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("filtering", [False, True])
def test_disabled_bitwise(ndim, scheme, filtering):
    grid = grid_for(ndim, filtering)
    U = gas_state(grid, ndim)
    plain = SOLVER[ndim-1](grid, BASE[ndim-1](1.4), scheme=scheme, cfl=0.3)
    dusty = DustyHydroSolver(grid, SGS[ndim-1](1.4), scheme=scheme, cfl=0.3,
                            dust_config={"enabled":False}, sgs_config={"enabled":False})
    dusty.U_gas = U.copy()
    assert dusty.compute_dt() == plain._stability_limit(U)
    for _ in range(4):
        dt = 0.5*dusty.compute_dt()
        U = plain._fixed_step(U, dt)
        dusty.step(dt)
        np.testing.assert_array_equal(dusty.U_gas, U)
    assert dusty.positivity.cells == plain.positivity.cells == 0


def turbulence_solver():
    s = coupled()
    u, v = turbulent_velocity_2d(*s.grid.xy_mesh(), vrms=0.25, kmin=1, kmax=4, seed=3)
    s.U_gas = s.equations.conservative(np.ones(s.shape), u+0.15, v-0.08, np.full(s.shape, 2.0))
    s.e_s[:] = 0.0
    s.dust_config["stopping"] = {"model":"epstein", "grain_rho_a":0.3}
    return s


def test_all_on_turbulence_conserves_and_is_realizable():
    s = turbulence_solver()
    initial = s.diagnostics()
    np.testing.assert_allclose(s._nu(), epstein_nu(s.U_gas, 1.4, 0.3, 2))
    for _ in range(50):
        s.step(min(0.002, s.compute_dt()))
        assert s.closure.realizable(s.U_dust).all()
        assert np.min(s.e_s) >= 0
        state = s.diagnostics()
        assert abs(state["E_total"]/initial["E_total"] - 1) < 1e-11
        for axis in "xy":
            np.testing.assert_allclose(state[f"momentum_{axis}"], initial[f"momentum_{axis}"],
                                       rtol=1e-12, atol=1e-14)
    assert s.positivity.cells == 0
    assert state["production"] > 0 and state["dust_work"] > 0 and state["sgs_decay"] > 0
    assert state["diffusion_substeps"] > 0
    for key in ("mass_gas", "mass_dust"):
        np.testing.assert_allclose(state[key], initial[key], rtol=1e-13)
    np.testing.assert_allclose(state["E_sgs"], initial["E_sgs"] + state["production"]
        - state["sgs_decay"] - state["dust_work"] + state["compression_to_sgs"], rtol=1e-12)
    # The cap is inactive: all of the measured filter loss reaches the reservoir.
    np.testing.assert_allclose(state["production"], state["filter_kinetic_drop"], rtol=1e-12)


@pytest.mark.parametrize("ndim,closure", [(1,"gaussian"), (1,"hyqmom"), (2,"gaussian"), (3,"gaussian")])
def test_uniform_box_against_coupled_ode(ndim, closure):
    errors = []
    for dt in (0.001, 0.0005):
        s = coupled(ndim, closure)
        s.sgs_config.update(production="none", c_eps=0.4)
        s.U_gas[:] = 0
        s.U_gas[0], s.U_gas[-1] = 1.0, 5.0
        s.e_s[:] = 0.2
        u = np.zeros((3, *s.shape))
        u[0] = 0.3
        s.U_dust = dust_state(s.closure, np.full(s.shape, 0.15), u, 0.1,
                              S3=0.2 if closure == "hyqmom" else 0.0)
        start = s.diagnostics()
        nu, end = 5.0, 0.04

        def rhs(t, y):
            es, variance = y
            D = float(sgs_coefficients(es, 1.0, nu, s.Delta)["D"])
            return [-3*0.15*D - 0.4*es**1.5/s.Delta, -2*nu*variance+2*D]

        ref = solve_ivp(rhs, (0, end), [0.2, 0.01], rtol=1e-11, atol=1e-13)
        assert ref.success
        for _ in range(round(end/dt)):
            s.step(dt)
        p = s.closure.primitives(s.U_dust)
        var = p["C"][0, 0] if closure == "gaussian" else p["v"]
        result = s.diagnostics()
        exact_delta = 0.3*np.exp(-nu*(1+0.15)*end)
        np.testing.assert_allclose(p["u"][0]-s.U_gas[1]/s.U_gas[0], exact_delta, rtol=3e-13)
        np.testing.assert_allclose(var, result["dust_var_mean"], atol=3e-15)
        error = np.max(np.abs(np.array([np.mean(s.e_s), np.mean(var)]) - ref.y[:, -1]))
        errors.append(error)
        assert error < 2e-4
        assert s.closure.realizable(s.U_dust).all()
        assert s.positivity.cells == 0
        np.testing.assert_allclose(result["E_total"], start["E_total"], rtol=1e-13)
        np.testing.assert_allclose(result["E_sgs"], start["E_sgs"] - result["sgs_decay"] - result["dust_work"], atol=1e-14)
        np.testing.assert_allclose(result["E_dust_random"] - start["E_dust_random"],
                                   result["dust_work"]-result["drag_heat_random"], atol=2e-14)
        np.testing.assert_allclose(result["E_gas"] - start["E_gas"],
            result["sgs_decay"] + result["drag_heat_mean"] + result["drag_heat_random"]
            + s._integral(s._kinetic(s.U_gas)), atol=3e-13)
        if closure == "hyqmom":
            np.testing.assert_allclose(p["q"], 0.2*0.1**3*np.exp(-3*nu*end), rtol=2e-11)
    # The specified local budget freezes nonlinear rates per half-step.
    assert errors[1] < 0.52*errors[0]


@pytest.mark.parametrize("dust,sgs_on", [(True,False), (False,True)])
@pytest.mark.parametrize("ndim", [1,2,3])
def test_single_reservoir_disabled(ndim, dust, sgs_on):
    s = coupled(ndim, dust_on=dust, sgs_on=sgs_on)
    initial = s.diagnostics()["E_total"]
    s.step(0.001)
    assert s.diagnostics()["E_total"] == pytest.approx(initial, rel=1e-13)
    if not sgs_on:
        assert not s.e_s.any() and s.ledger["production"] == 0
    else:
        assert not s.U_dust.any() and s.ledger["dust_work"] == 0


def test_production_cap_and_compression_ledgers():
    s = turbulence_solver()
    s.sgs_config["f_cap"] = 1e-8
    initial = s.diagnostics()
    s.step(0.002)
    final = s.diagnostics()
    assert 0 < final["production"] < final["filter_kinetic_drop"]
    assert abs(final["compression_to_sgs"]) > 0
    np.testing.assert_allclose(final["E_sgs"], final["production"] - final["dust_work"]
                               - final["sgs_decay"] + final["compression_to_sgs"], atol=2e-15)
    np.testing.assert_allclose(final["E_total"], initial["E_total"], rtol=1e-14)


def test_time_history_callbacks_and_output_boundaries():
    s = coupled(1)
    outputs, steps = [], []
    h = s.run(1.0, 1.025, 0.01, on_output=lambda t,U: outputs.append(t),
              on_step=lambda n,dt,U: steps.append((n,dt)))
    np.testing.assert_allclose(outputs, [1,1.01,1.02,1.025], atol=1e-15)
    assert len(h["time"]) == len(steps)+1
    np.testing.assert_allclose(np.diff(h["time"]), h["dt"][1:], atol=1e-16)
    assert all(len(v) == len(h["time"]) for v in h.values())
    s2 = coupled(1)
    h2 = s2.run(0.0, 1.0, 0.1, max_steps=2)
    assert len(h2["time"]) == 3 and s2.t < 1.0


PROBLEMS = ("dusty_box", "dustywave1d", "dust_beams1d", "sgs_acoustic1d", "dusty_turb2d", "dusty_turb3d")


@pytest.mark.parametrize("name", PROBLEMS)
def test_problem_configs_run_and_write_snapshots(name, tmp_path):
    config = load_config(f"configs/{name}.yaml")
    config["io"]["outdir"] = str(tmp_path)
    # Exercise the supplied resolutions for several short accepted steps.
    config["integration"].update(t_end=0.002, output_interval=0.001)
    problem = ProblemRegistry.create_problem(name, config=config)
    s, h = problem.run()
    assert s.t == 0.002 and len(h["time"]) >= 3
    assert s.positivity.cells == 0
    assert s.closure.realizable(s.U_dust).all()
    paths = sorted(tmp_path.glob("snapshot_*.npz"))
    assert len(paths) == 3
    with np.load(paths[-1]) as snap:
        assert snap["t"] == s.t and snap["closure"] == s.closure.name
        np.testing.assert_array_equal(snap["gas"], s.U_gas)
        np.testing.assert_array_equal(snap["dust"], s.U_dust)
        np.testing.assert_array_equal(snap["e_s"], s.e_s)
        assert tuple(snap["field_names"]) == s.closure.field_names
    with np.load(tmp_path/"history.npz") as saved:
        np.testing.assert_array_equal(saved["E_total"], h["E_total"])
    from matplotlib.image import imread
    image = imread(tmp_path/"diagnostics.png")
    assert image.shape[0] > 100 and np.ptp(image[:, :, :3]) > 0.5


@pytest.mark.parametrize("closure", ["gaussian", "hyqmom"])
@pytest.mark.parametrize("profile", ["two_streams", "gaussian"])
def test_beam_profiles(closure, profile, tmp_path):
    cfg = load_config("configs/dust_beams1d.yaml")
    cfg["io"]["outdir"] = str(tmp_path)
    cfg["dust"]["closure"] = closure
    cfg["initial_conditions"]["profile"] = profile
    p = ProblemRegistry.create_problem("dust_beams1d", config=cfg)
    g = p.create_grid()
    c = GaussianClosure(1) if closure == "gaussian" else HyqmomClosure()
    U = p.create_dust_initial_conditions(g, p.create_initial_conditions(g), c)
    assert c.realizable(U).all()
    if profile == "two_streams":
        # Raw moments represent both streams, including their overlap variance.
        np.testing.assert_allclose(c.kinetic_energy(U), 0.5*U[0]*0.5**2, rtol=1e-14)


def test_2d_turbulence_generator():
    grid = grid_for(2)
    u,v = turbulent_velocity_2d(*grid.xy_mesh(),vrms=0.3,kmin=1,kmax=4,seed=3)
    u2,v2 = turbulent_velocity_2d(*grid.xy_mesh(),vrms=0.3,kmin=1,kmax=4,seed=3)
    np.testing.assert_array_equal(u,u2)
    np.testing.assert_array_equal(v,v2)
    assert np.sqrt(np.mean(u*u+v*v)) == pytest.approx(0.3)
    np.testing.assert_allclose(grid.dx1(u)+grid.dy1(v), 0, atol=2e-14)


def test_rejected_modes_and_states(tmp_path):
    g = grid_for(1)
    with pytest.raises(ValueError, match="adaptive"):
        DustyHydroSolver(g, SGS[0](), cfl=0.3, scheme="rk45", dust_config={}, sgs_config={})
    polynomial = Grid1D(8,1.0,basis="legendre",backend="numpy")
    with pytest.raises(ValueError, match="Fourier"):
        DustyHydroSolver(polynomial, SGS[0](), cfl=0.3, dust_config={}, sgs_config={})
    with pytest.raises(ValueError, match="ndim=1"):
        DustyHydroSolver(grid_for(2), SGS[1](), cfl=0.3, dust_config={"closure":"hyqmom"}, sgs_config={})
    s = coupled(1, sgs_on=False)
    s.e_s[:] = 1.0
    with pytest.raises(ValueError, match="zero when SGS"):
        s.step(0.001)
    cfg = load_config("configs/dusty_box.yaml")
    cfg["io"]["outdir"] = str(tmp_path)
    p = ProblemRegistry.create_problem("dusty_box", config=cfg)
    with pytest.raises(ValueError, match="torch"):
        p.run(backend="torch")
    torch = pytest.importorskip("torch")
    s = coupled(1)
    s.U_gas = torch.from_numpy(s.U_gas)
    with pytest.raises(TypeError, match="NumPy"):
        s.step(0.001)
