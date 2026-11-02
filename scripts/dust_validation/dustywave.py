"""Linear dusty acoustic wave with drag back-reaction against the exact mode solution.

Gas (adiabatic, spectral) and cold dust (realizable moments, finite volume) start
co-moving in a single Fourier mode.  For amplitudes (drho_g, du_g, drho_d, du_d)
the linear system y' = M y with
    M = [[0, -ik rho_g0, 0, 0], [-ik c^2/rho_g0, -eps nu, 0, eps nu],
         [0, 0, 0, -ik rho_d0], [0, nu, 0, -nu]],  c^2 = gamma p0/rho_g0,
is integrated exactly with scipy.linalg.expm (Laibe & Price 2011 setup, adiabatic).
Amplitude 1e-6 keeps nonlinear departures (~amplitude) below the discretisation
errors measured here.  Cases: dust-to-gas eps in {0.01, 1}, stopping time
t_s/T in {1e-3, 0.3, 30}, T the acoustic period; N in {32, 64, 128, 256}.

PYTHONPATH=. python scripts/dust_validation/dustywave.py
"""
from common import Report, solver, np, plt
from scipy.linalg import expm
from phrike.problems.problem_list.dusty_box import dust_state

GAMMA, AMP = 1.4, 1e-6
C, K = np.sqrt(GAMMA), 2*np.pi
PERIOD = 1/C
FIELDS = ("rho_g", "u_g", "rho_d", "u_d")


def exact_modes(eps, nu, t):
    M = np.array([[0, -1j*K, 0, 0], [-1j*K*C*C, -eps*nu, 0, eps*nu],
                  [0, 0, 0, -1j*K*eps], [0, nu, 0, -nu]], dtype=complex)
    y0 = AMP*np.array([1.0, C, eps, C])
    return expm(M*t) @ y0, y0


def run(eps, ratio, n):
    nu = 1/(ratio*PERIOD)
    s = solver(n=n, eps=eps, ts=1/nu, cfl=0.8)
    x = s.grid.x
    phase = np.cos(K*x)
    rho, vel = 1 + AMP*phase, AMP*C*phase
    s.U_gas = s.equations.conservative(rho, vel, 1 + C*C*AMP*phase)
    u = np.zeros((3, n))
    u[0] = vel
    s.U_dust = dust_state(s.closure, eps*rho, u)
    E0 = s.diagnostics()["E_total"]
    while s.t < PERIOD - 1e-14:
        s.step(min(s.gas_solver._stability_limit(s.U_gas), PERIOD - s.t))
    p = s.closure.primitives(s.U_dust)
    numerical = np.array([s.U_gas[0] - 1, s.U_gas[1]/s.U_gas[0], p["rho"] - eps, p["u"][0]])
    amp, y0 = exact_modes(eps, nu, PERIOD)
    reference = np.real(amp[:, None]*np.exp(1j*K*x))
    l1 = np.mean(np.abs(numerical - reference), axis=1)/np.abs(y0)
    drift = abs(s.diagnostics()["E_total"]/E0 - 1)
    return x, numerical, reference, l1, drift, s


def main():
    r = Report("dustywave")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for ie, eps in enumerate((0.01, 1.0)):
        for it, ratio in enumerate((1e-3, 0.3, 30.0)):
            # Tight coupling needs nu*dt <~ 1 to leave the stiff splitting regime.
            sizes = (64, 128, 256, 512) if ratio < 0.01 else (32, 64, 128, 256)
            errors = []
            for n in sizes:
                x, num, ref, l1, drift, s = run(eps, ratio, n)
                errors.append(l1)
                label = f"eps={eps:g}, ts/T={ratio:g}, N={n}"
                r.check(label+" energy drift", drift, 1e-12)
                r.check(label+" repaired_cells", s.transport_stats["repaired_cells"], 0, op="==")
                r.check(label+" unrealizable cells", np.count_nonzero(~s.closure.realizable(s.U_dust)), 0, op="==")
                r.runs.append(dict(case=label, l1_over_initial_amplitude=dict(zip(FIELDS, l1)),
                                   energy_drift=drift, transport=dict(s.transport_stats)))
            errors = np.array(errors)
            orders = np.log2(errors[:-1]/errors[1:])
            r.runs.append(dict(case=f"eps={eps:g}, ts/T={ratio:g} L1 orders", orders=orders))
            # A field already at the amplitude-1e-6 nonlinear floor (~2e-6 of the
            # mode amplitude: spectral gas when weakly coupled, dust velocity when
            # drag is negligible) passes on its error, not its order.  Otherwise
            # fields must converge like a limited second-order scheme on the
            # finest pair (minmod caps smooth extrema near 1.5).  Under stiff drag
            # (nu*dt >> 1) the separate gas/dust transports drift apart by O(dt^2)
            # per step and the scheme is first order (docs §6); tight coupling is
            # therefore refined to N=512, where nu*dt ~ 2, and must have recovered.
            for j, field in enumerate(FIELDS):
                name = f"eps={eps:g}, ts/T={ratio:g} {field}"
                if errors[-1, j] <= 1e-5:
                    r.check(name+f" L1 error at N={sizes[-1]} (floor)", errors[-1, j], 1e-5)
                else:
                    r.check(name+f" L1 order {sizes[-2]}-{sizes[-1]}", orders[-1, j], 1.5, op=">=")
            ax = axes[ie, it]
            for j, field in enumerate(FIELDS):
                ax.loglog(sizes, errors[:, j], "o-", label=field)
            lo, hi = sizes[0], sizes[-1]
            ax.loglog([lo, hi], errors[0, 2]*np.array([1, (lo/hi)**2]), "k--", lw=0.8, label="slope 2")
            ax.loglog([lo, hi], errors[0, 2]*np.array([1, lo/hi]), "k:", lw=0.8, label="slope 1")
            ax.set(xlabel="cells N", ylabel="L1 / initial mode amplitude", title=f"eps={eps:g}, t_s/T={ratio:g}")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    r.figure(fig, "convergence.png")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    x, num, ref, l1, drift, s = run(1.0, 0.3, 64)
    xf = np.linspace(0, 1, 400)
    amp, y0 = exact_modes(1.0, 1/(0.3*PERIOD), PERIOD)
    for j, field in enumerate(FIELDS):
        axes[j].plot(xf, np.real(amp[j]*np.exp(1j*K*xf))/abs(y0[j]), "k-", lw=1, label="exact")
        axes[j].plot(x, num[j]/abs(y0[j]), "o", ms=3, label="PHRIKE N=64")
        axes[j].set(xlabel="x / L", ylabel=f"{field} / initial amplitude", title=field)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dusty wave after one acoustic period, eps=1, t_s/T=0.3")
    r.figure(fig, "profiles.png")
    r.notes.append("Cold dust uses primitive (rho, u) faces with sub-cell-level cone projection "
                   "(docs §3.3); gas is pseudo-spectral; drag is exact with back-reaction.")
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
