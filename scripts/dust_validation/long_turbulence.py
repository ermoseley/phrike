"""Decaying turbulence from an empty SGS reservoir: the three-way exchange over time.

The resolved cascade is the only energy source.  The spectral dissipation drains
resolved kinetic energy into the SGS reservoir, the reservoir heats the dust
(MT25) and decays to gas heat, and drag couples dust and gas both ways.  This
complements turb_energy_ledger.py, whose windows are short and whose reservoirs
start full.

PYTHONPATH=. python scripts/dust_validation/long_turbulence.py [--quick]
"""
import sys

from common import Report, np, plt
from phrike.dusty_hydro import DustyHydroSolver, SGSEulerEquations2D, SGSEulerEquations3D
from phrike.grid import Grid2D, Grid3D
from phrike.problems.problem_list.dusty_box import dust_state
from phrike.problems.problem_list.dusty_turb2d import turbulent_velocity_2d
from phrike.problems.problem_list.turb3d import turbulent_velocity_3d
from phrike.sgs import sgs_coefficients

QUICK = "--quick" in sys.argv
VRMS, GAMMA, T_S = 0.3, 1.4, 0.1


def build(ndim, n, eps):
    opts = dict(backend="numpy", precision="double", fft_workers=1,
                filter_params={"spectral_dissipation": dict(enabled=True, p=8, rate=30.0)})
    if ndim == 2:
        grid, eqs = Grid2D(n, n, 1.0, 1.0, **opts), SGSEulerEquations2D(GAMMA)
        X, Y = grid.xy_mesh()
        vel = turbulent_velocity_2d(X, Y, VRMS, 2, 8, seed=3)
    else:
        grid, eqs = Grid3D(n, n, n, 1.0, 1.0, 1.0, **opts), SGSEulerEquations3D(GAMMA)
        X, Y, Z = grid.xyz_mesh()
        U3 = turbulent_velocity_3d(X, Y, Z, 1.0, 1.0, VRMS, 2, 8, 0.0, "parabolic", -5/3, 3, GAMMA)
        vel = tuple(U3[1:4]/U3[0])
    s = DustyHydroSolver(grid, eqs, scheme="rk4", cfl=0.3,
                         dust_config=dict(closure="gaussian", backreaction=True,
                                          stopping=dict(model="constant", t_s=T_S),
                                          eddy_diffusion=dict(enabled=True, c_kappa=1.0)),
                         sgs_config=dict(production="filter_drain", pressure=True, decay=True,
                                         d_model="mt25", c_eps=1.0))
    rho = np.ones_like(vel[0])
    s.U_gas = eqs.conservative(rho, *vel, np.ones_like(rho))
    # Dust starts co-moving with the gas, so the run measures turbulence
    # modulation rather than the impulsive acceleration of dust at rest.
    u_d = np.zeros((3, *rho.shape))
    u_d[:ndim] = np.asarray(vel)
    s.U_dust = dust_state(s.closure, eps*rho, u_d, 0.0)
    s.e_s = np.zeros_like(rho)
    return s


def sample(s):
    d = s.diagnostics()
    rho_d = s.U_dust[0]
    D = sgs_coefficients(s.e_s, s.U_gas[0], s._nu(), s.Delta)["D"]
    d["D_mass"] = float(np.sum(rho_d*D)/np.sum(rho_d))
    d["E_resolved"] = s._integral(s._kinetic(s.U_gas))
    return d


def main():
    report = Report("long_turbulence")
    cases = [(2, 64 if QUICK else 128), (3, 16 if QUICK else 24)]
    t_end = 0.4 if QUICK else 1.0
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 7), constrained_layout=True)
    late = {}
    for row, (ndim, n) in enumerate(cases):
        for eps, style in ((0.01, "-"), (1.0, "--")):
            s = build(ndim, n, eps)
            rows = [sample(s)]
            prediction = [rows[0]["dust_var_mean"]]
            outputs = np.linspace(0, t_end, 41)[1:]
            for target in outputs:
                while s.t < target - 1e-12:
                    s.step(min(s.compute_dt(), target - s.t))
                now = sample(s)
                h = now["time"] - rows[-1]["time"]
                a = np.exp(-2*h/T_S)
                Dm = 0.5*(rows[-1]["D_mass"] + now["D_mass"])
                prediction.append(a*prediction[-1] + Dm*T_S*(1 - a))
                rows.append(now)
            h = {k: np.array([r[k] for r in rows]) for k in rows[0]}
            name = f"{ndim}d_n{n}_eps{eps:g}"
            drift = float(np.max(np.abs(h["E_total"]/h["E_total"][0] - 1)))
            ledger = h["production"] - h["sgs_decay"] - h["dust_work"] + h["compression_to_sgs"]
            ledger_err = float(np.max(np.abs(ledger - h["E_sgs"]))/max(np.max(h["E_sgs"]), 1e-300))
            # Periodic conservative transport plus pointwise exchanges: round-off only.
            report.check(name+"/energy_drift", drift, 1e-10)
            report.check(name+"/sgs_ledger_closure", ledger_err, 1e-10)
            report.check(name+"/repaired_cells", h["repaired_cells"][-1], 0)
            report.check(name+"/positivity_floor_cells", h["positivity_cells"][-1], 0)
            # The reservoir must be filled by the cascade (it starts at zero).
            report.check(name+"/sgs_filled_by_cascade", h["production"][-1], 0.0, op=">")
            late[ndim, eps] = h
            t = h["time"]
            ax = axes[row]
            ax[0].plot(t, h["E_resolved"], style, color="C0", label=f"resolved, eps={eps:g}")
            ax[0].plot(t, h["E_sgs"], style, color="C1", label=f"SGS, eps={eps:g}")
            ax[0].plot(t, h["E_dust_random"], style, color="C2", label=f"dust random, eps={eps:g}")
            for key, color in (("production", "C3"), ("sgs_decay", "C4"), ("dust_work", "C5"),
                               ("drag_heat_mean", "C6")):
                ax[1].plot(t, h[key], style, color=color, label=f"{key}, eps={eps:g}")
            # Dust random-energy budget: SGS work in, drag damping out, and the
            # dispersion created when transport mixes the resolved dust velocity.
            transport = h["E_dust_random"] - h["E_dust_random"][0] - h["dust_work"] + h["drag_heat_random"]
            ax[2].plot(t, h["dust_work"]/eps, style, color="C3", label=f"SGS work, eps={eps:g}")
            ax[2].plot(t, -h["drag_heat_random"]/eps, style, color="C4", label=f"-drag damping, eps={eps:g}")
            ax[2].plot(t, transport/eps, style, color="C2", label=f"transport mixing, eps={eps:g}")
            report.runs.append(dict(case=name, steps=int(h["substeps"][-1]), energy_drift=drift,
                                    ledger_error=ledger_err, final=dict((k, float(v[-1])) for k, v in h.items()),
                                    ou_prediction_final=float(prediction[-1]),
                                    roundoff_projections=int(h["roundoff_projections"][-1]),
                                    mood_cells=int(h.get("mood_cells", np.zeros(1))[-1])))
        for col, (title, ylab) in enumerate((("reservoir energies", "energy / (rho0 L^d v_unit^2)"),
                                             ("cumulative exchanges", "energy / (rho0 L^d v_unit^2)"),
                                             ("dust random-energy budget / eps", "energy / (eps rho0 L^d v_unit^2)"))):
            axes[row, col].set(title=f"{ndim}D {n}^{ndim}: {title}", xlabel="t / (L / c_s)", ylabel=ylab)
            axes[row, col].legend(fontsize=6)
        # Heavy loading drains both the resolved flow (mean drag) and the SGS
        # reservoir (dust work), so both must end lower than with light loading.
        report.check(f"{ndim}d/heavy_loading_damps_resolved",
                     late[ndim, 1.0]["E_resolved"][-1]/late[ndim, 0.01]["E_resolved"][-1], 1.0, op="<")
        report.check(f"{ndim}d/heavy_loading_damps_sgs",
                     late[ndim, 1.0]["E_sgs"][-1]/late[ndim, 0.01]["E_sgs"][-1], 1.0, op="<")
    report.figure(fig, "long_turbulence.png")
    report.notes.append(f"vrms={VRMS}, t_s={T_S}, spectral dissipation p=8 rate=30, MT25 coupling, "
                        "SGS pressure, decay, drag back-reaction, dust eddy diffusion; e_s(0)=0.")
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
