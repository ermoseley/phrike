"""Exact charged sources and small coupled MHD conservation regressions."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from phrike.dust_moments import GaussianClosure, HyqmomClosure
from phrike.dust_sources import epstein_nu, local_coupling_step
from phrike.dusty_hydro import DustyHydroSolver, SGSEulerEquations1D
from phrike.dusty_mhd import (
    DustyMHDSolver, SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D,
)
from phrike.equations import MHDEquations1D, MHDEquations2D, MHDEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.io import load_config
from phrike.problems import ProblemRegistry
from phrike.problems.problem_list.dusty_box import dust_state
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.sgs import sgs_coefficients
from phrike.solver import SpectralSolverMHD
import phrike.problems.register  # noqa: F401


SGS = (SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D)
BASE = (MHDEquations1D, MHDEquations2D, MHDEquations3D)
BHAT = np.array([2., -1., 3.])/np.sqrt(14.)


def box(closure, *, eps=0.2, field=BHAT):
    shape = (1,)*closure.ndim
    rho, zero = np.ones(shape), np.zeros(shape)
    ug = np.array([0.2, -0.1, 0.3]).reshape((3, *shape))
    ud = np.array([0.7, 0.4, -0.2]).reshape((3, *shape))
    B = np.asarray(field).reshape((3, *shape)) + zero
    gas = BASE[closure.ndim-1](1.4).conservative(rho, *ug, 2*rho, *B)
    dust = dust_state(closure, eps*rho, ud, 0.1)
    return gas, dust, zero


def source(g, d, e, h, closure, nu, omega, **kwargs):
    return local_coupling_step(g, d, e, h, closure=closure, gamma=1.4,
        nu=nu, Delta=0.25, ndim=closure.ndim, charge_to_mass=omega,
        d_model=kwargs.pop("d_model", "none"), sgs_decay=False, **kwargs)


@pytest.mark.parametrize("backreaction", [False, True])
def test_two_body_map_against_ode_and_stiff_limit(backreaction):
    c = GaussianClosure(1)
    g, d, e = box(c)
    ug, ud = g[1:4, 0], c.primitives(d)["u"][:, 0]
    eps = d[0, 0]/g[0, 0]
    # Independent six-variable ODE, including both corners of the stiffness plane.
    cross = np.array([[0, BHAT[2], -BHAT[1]], [-BHAT[2], 0, BHAT[0]],
                      [BHAT[1], -BHAT[0], 0]])
    for nu, omega in ((1e-4, 1e-4), (1e-4, 1e4), (1, -1),
                      (1e4, 1e-4), (1e4, 1e4), (0, 30)):
        L = -nu*np.eye(3) + omega*cross
        ratio = eps if backreaction else 0
        matrix = np.block([[ratio*L, -ratio*L], [-L, L]])
        ref = solve_ivp(lambda t, y: matrix@y, (0, 1), np.r_[ug, ud],
                        method="DOP853", rtol=2e-10, atol=2e-12)
        assert ref.success
        gn, dn, en, ledger = source(g, d, e, 1, c, nu, omega,
                                     backreaction=backreaction)
        un = c.primitives(dn)["u"][:, 0]
        np.testing.assert_allclose(np.r_[gn[1:4, 0], un], ref.y[:, -1], rtol=0, atol=2e-7)
        np.testing.assert_allclose(gn[4] + c.kinetic_energy(dn) + en,
                                   g[4] + c.kinetic_energy(d) + e, rtol=0, atol=3e-15)
        np.testing.assert_array_equal(gn[5:8], g[5:8])
        if backreaction:
            np.testing.assert_allclose(gn[1:4] + dn[1:4], g[1:4] + d[1:4], rtol=0, atol=2e-16)
        else:
            np.testing.assert_array_equal(gn[1:4], g[1:4])
        delta = un - gn[1:4, 0]
        if nu == 0:
            np.testing.assert_allclose(np.linalg.norm(delta), np.linalg.norm(ud - ug), rtol=0, atol=3e-16)
            if backreaction:
                np.testing.assert_array_equal(ledger["Q_mean"], 0)
        if nu == 1e4:
            np.testing.assert_allclose(delta, 0, rtol=0, atol=2e-16)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_covariance_rotation_preserves_spectrum(ndim):
    c = GaussianClosure(ndim)
    g, d, e = box(c)
    p = c.primitives(d)
    covariance = np.array([[0.3, 0.08, -0.02], [0.08, 0.1, 0.01], [-0.02, 0.01, 0.2]])
    C = covariance.reshape((3, 3, *((1,)*ndim)))
    d = c.from_primitives(p["rho"], p["u"], C)
    for angle in (-1e4, -0.3, 0, 2, 1e4):
        _, dn, _, _ = source(g, d, e, 1, c, 0, angle)
        Cn = c.primitives(dn)["C"].reshape(3, 3)
        R = Rotation.from_rotvec(-angle*BHAT).as_matrix()
        np.testing.assert_allclose(Cn, R@covariance@R.T, rtol=0, atol=6e-13)
        np.testing.assert_allclose(np.linalg.eigvalsh(Cn), np.linalg.eigvalsh(covariance), rtol=0, atol=3e-15)
        assert c.realizable(dn).all()


def test_magnetized_coefficients_limits_and_resonance():
    for x, y in ((1e-12, 0), (1, 0), (1e12, 20), (0, 2), (0, 1), (0, 0.5)):
        # sigma=1, Delta=1: tau=eps_D=1, x=nu, y=|omega|.
        with np.errstate(all="raise"):
            coeff = sgs_coefficients(1.5, 1, x, 1, "mt25_magnetized", omega=y)
        dp, dt = coeff["D_par"], coeff["D_perp"]
        assert dp >= 0 and dt >= 0
        if y == 0:
            assert dp == dt
        if x == 1e12:
            np.testing.assert_allclose([dp, dt], np.pi/2, rtol=1e-11)
        if x == 0:
            assert dp == 0
            assert dt == pytest.approx(np.pi/2 if y > 1 else np.pi/4 if y == 1 else 0)
    isotropic = sgs_coefficients(1.5, 1, 0.3, 1, "mt25", omega=20)
    assert isotropic["D_par"] == isotropic["D_perp"]


@pytest.mark.parametrize("name,ndim", [("gaussian", 1), ("gaussian", 2),
                                      ("gaussian", 3), ("hyqmom", 1)])
def test_magnetized_ou_time_dependence(name, ndim):
    c = GaussianClosure(ndim) if name == "gaussian" else HyqmomClosure()
    field = BHAT if name == "gaussian" else np.array([1., 0, 0])
    g, _, e = box(c, field=field)
    d = dust_state(c, np.full(e.shape, 1e-12), np.zeros((3, *e.shape)), 0.1)
    nu, omega = 0.7, 8.0
    es = e + 0.06
    coeff = sgs_coefficients(es, g[0], nu, 0.25, "mt25_magnetized", omega=omega)
    for h in (1e-4, 0.1, 3, 30):
        gn, dn, en, ledger = source(g, d, es, h, c, nu, omega, d_model="mt25_magnetized")
        p = c.primitives(dn)
        if name == "gaussian":
            par = np.einsum("i,ij...,j->...", field, p["C"], field)
            perp = (np.einsum("ii...->...", p["C"]) - par)/2
        else:
            par, perp = p["v"], p["sperp2"]
            np.testing.assert_allclose(p["r"], 3*par**2, rtol=1e-11)
        for value, key in ((par, "D_par"), (perp, "D_perp")):
            exact = 0.01*np.exp(-2*nu*h) + coeff[key]/nu*(-np.expm1(-2*nu*h))
            np.testing.assert_allclose(value, exact, rtol=2e-10)
        np.testing.assert_allclose(gn[4]+en+c.kinetic_energy(dn),
                                   g[4]+es+c.kinetic_energy(d), rtol=0, atol=2e-15)
        np.testing.assert_allclose(ledger["L_dust"],
            ledger["Q_rand"]+c.random_energy(dn)-c.random_energy(d), rtol=0, atol=1e-16)


def test_field_aligned_hyqmom_matches_gaussian_and_rejects_oblique():
    results = []
    for c in (GaussianClosure(1), HyqmomClosure()):
        g, d, e = box(c, field=[-1, 0, 0])
        gn, dn, en, _ = source(g, d, e+0.1, 0.3, c, 2, -7, d_model="mt25_magnetized")
        p = c.primitives(dn)
        variance = np.array([p["C"][0, 0], p["C"][1, 1]]) if c.name == "gaussian" else np.array([p["v"], p["sperp2"]])
        results.append((gn, p["u"], variance, en))
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-15)
    c = HyqmomClosure()
    g, d, e = box(c)
    with pytest.raises(ValueError, match="parallel to x"):
        source(g, d, e, 0.1, c, 1, 1)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_mhd_flux_pressure_and_disabled_matches_reference(ndim):
    opts = dict(backend="numpy", fft_workers=1)
    grid = (Grid1D(8, 1., **opts) if ndim == 1 else Grid2D(8, 8, 1., 1., **opts)
            if ndim == 2 else Grid3D(6, 6, 6, 1., 1., 1., **opts))
    s = DustyMHDSolver(grid, SGS[ndim-1](), cfl=0.3,
                       dust_config={"enabled": False}, sgs_config={"enabled": False})
    rho, zero = np.ones(s.shape), np.zeros(s.shape)
    s.U_gas = s.equations.conservative(rho, 0.2+zero, zero, zero, rho, rho, zero, zero)
    base = BASE[ndim-1]()
    bf = base.flux(s.U_gas)
    s.equations.p_sgs = 0.1*rho
    sf = s.equations.flux(s.U_gas)
    for direction, (f, g) in enumerate(zip((bf,) if ndim == 1 else bf, (sf,) if ndim == 1 else sf)):
        expected = f.copy()
        expected[1+direction] += 0.1*rho
        expected[4] += 0.1*s.U_gas[1+direction]/rho
        np.testing.assert_array_equal(g, expected)
    assert s.equations.max_wave_speed(s.U_gas) >= base.max_wave_speed(s.U_gas)
    s.equations.p_sgs = None
    assert s.equations.max_wave_speed(s.U_gas) == base.max_wave_speed(s.U_gas)
    plain = SpectralSolverMHD(grid, base, cfl=0.3)
    expected = plain._reference_fixed_step(s.U_gas, 0.001)
    s.step(0.001)
    np.testing.assert_array_equal(s.U_gas, expected)
    np.testing.assert_allclose(epstein_nu(s.U_gas, base.gamma, 0.2, ndim),
                               np.sqrt(8/np.pi)/0.2, rtol=2e-15)


def test_hydro_charge_configuration_is_bit_identical():
    states = []
    for charge in (0., 50.):
        s = DustyHydroSolver(Grid1D(8, 1., backend="numpy"), SGSEulerEquations1D(1.4),
            cfl=0.3, dust_config={"charge_to_mass": charge}, sgs_config={})
        rho, zero = np.ones(s.shape), np.zeros(s.shape)
        s.U_gas = s.equations.conservative(rho, 0.2+zero, rho)
        s.U_dust = dust_state(s.closure, 0.1*rho, np.full((3, *s.shape), 0.3), 0.1)
        s.e_s[:] = 0.1
        for _ in range(3):
            s.step(0.01)
        states.append((s.U_gas, s.U_dust, s.e_s))
    for a, b in zip(*states):
        np.testing.assert_array_equal(a, b)


def test_zero_magnetic_field_disables_charge_bitwise():
    c = GaussianClosure(3)
    g, d, e = box(c, field=[0., 0., 0.])
    neutral = source(g, d, e+0.1, 0.3, c, 2, 0, d_model=None)
    charged = source(g, d, e+0.1, 0.3, c, 2, 50, d_model=None)
    for a, b in zip(neutral[:3], charged[:3]):
        np.testing.assert_array_equal(a, b)
    for key in neutral[3]:
        np.testing.assert_array_equal(neutral[3][key], charged[3][key])


@pytest.mark.parametrize("eta", [0., 0.01])
def test_magnetic_filter_loss_becomes_heat_and_not_sgs(eta):
    grid = Grid1D(16, 1., backend="numpy", fft_workers=1,
        filter_params={"spectral_dissipation": dict(enabled=True, p=2, rate=100., onset_fraction=0.)})
    grid.mhd_config = {"resistivity": eta}
    s = DustyMHDSolver(grid, SGSMHDEquations1D(), cfl=0.3,
        dust_config={"enabled": False}, sgs_config={"decay": False, "d_model": "none"})
    one, zero = np.ones(s.shape), np.zeros(s.shape)
    phase = 2*np.pi*grid.x
    s.U_gas = s.equations.conservative(one, zero, zero, zero, one, zero,
                                       0.1*np.cos(phase), 0.1*np.sin(phase))
    initial = s.diagnostics()
    p0 = s.equations.primitive(s.U_gas)[4].copy()
    ref = s.gas_solver._reference_fixed_step(s.U_gas, 0.01)
    s.step(0.01)
    final = s.diagnostics()
    np.testing.assert_array_equal(s.U_gas, ref)
    assert final["E_magnetic"] < initial["E_magnetic"]
    assert np.min(s.equations.primitive(s.U_gas)[4] - p0) > 0
    assert final["production"] < 1e-30
    assert abs(final["E_total"]/initial["E_total"] - 1) < 1e-14


def test_orszag_tang_all_on_energy_and_ledger():
    grid = Grid2D(16, 16, 1., 1., backend="numpy", fft_workers=1,
        filter_params={"spectral_dissipation": dict(enabled=True, p=4, rate=10., onset_fraction=0.)})
    s = DustyMHDSolver(grid, SGSMHDEquations2D(), cfl=0.3,
        dust_config=dict(charge_to_mass=10., stopping={"t_s": 0.1},
                         eddy_diffusion={"enabled": True, "c_kappa": 0.1}),
        sgs_config=dict(diffusion={"enabled": True, "c_kappa": 0.1}))
    s.U_gas = orszag_tang_2d(*grid.xy_mesh())
    s.U_dust = dust_state(s.closure, 0.2*s.U_gas[0], s.U_gas[1:4]/s.U_gas[0], 0.05)
    s.e_s[:] = 0.01
    initial = s.diagnostics()
    for _ in range(50):
        s.step(min(0.001, s.compute_dt()))
        assert s.closure.realizable(s.U_dust).all()
    final = s.diagnostics()
    assert abs(final["E_total"]/initial["E_total"]-1) < 1e-11
    assert final["repaired_cells"] == final["positivity_cells"] == 0
    assert final["div_b_max"] < 1e-12
    assert final["production"] > 0 and final["dust_work"] > 0 and final["sgs_decay"] > 0
    np.testing.assert_allclose(final["E_sgs"], initial["E_sgs"] + final["production"]
        - final["dust_work"] - final["sgs_decay"] + final["compression_to_sgs"], rtol=0, atol=1e-14)
    for axis in "xyz":
        assert final[f"momentum_{axis}"] == pytest.approx(initial[f"momentum_{axis}"], abs=2e-16)


@pytest.mark.parametrize("name", ["hall", "ambipolar", "shearing_sheet"])
def test_unsupported_mhd_sources_raise(name):
    grid = Grid1D(8, 1., backend="numpy")
    grid.mhd_config = {name: {"enabled": True}}
    with pytest.raises(ValueError, match=name):
        DustyMHDSolver(grid, SGS[0](), cfl=0.3, dust_config={}, sgs_config={})


@pytest.mark.parametrize("name", ["charged_dust_box", "dusty_alfven1d", "dusty_orszag_tang2d"])
def test_registered_configs_write_mhd_snapshots(name, tmp_path):
    cfg = load_config(f"configs/{name}.yaml")
    cfg["io"]["outdir"] = str(tmp_path)
    cfg["integration"].update(t_end=0.001, output_interval=0.001)
    for key in ("N", "Nx", "Ny", "Nz"):
        if key in cfg["grid"]:
            cfg["grid"][key] = 12
    s, history = ProblemRegistry.create_problem(name, config=cfg).run()
    assert s.t == 0.001 and s.positivity.cells == 0
    assert s.closure.realizable(s.U_dust).all()
    with np.load(tmp_path/"snapshot_00001.npz") as snap:
        assert tuple(snap["gas_field_names"]) == ("rho", "rho_ux", "rho_uy", "rho_uz", "E", "Bx", "By", "Bz")
        np.testing.assert_array_equal(snap["gas"], s.U_gas)
    assert abs(history["E_total"][-1]/history["E_total"][0]-1) < 1e-11
