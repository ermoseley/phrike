"""MT25 dust heating with a held SGS reservoir: growth rate, saturation, Gaussianisation.

The claim under test (Moseley & Teyssier 2025; MT25 production contract §3): unresolved
turbulence of one-component dispersion sigma_D drives grain velocities as an
Ornstein-Uhlenbeck process with diffusion D = eps_D arctan(x), eps_D = sigma_D^3/Delta,
x = nu tau_Delta.  The dispersion therefore grows from zero at rate 2D and saturates at
D/nu, with sigma^2(t) = (D/nu)(1 - e^{-2 nu t}), and any initial distribution relaxes to a
Gaussian (kurtosis 3).

To isolate that claim the reservoir is held fixed: SGS decay is off and the dust loading
(1e-9) drains a negligible fraction of it, so D is constant in time.  The full coupled
solver (DustyHydroSolver) is stepped in a uniform periodic box; transport is identity.

PYTHONPATH=. python scripts/dust_validation/mt25_gaussianisation.py
"""
from common import Report, solver, np, plt
from phrike.problems.problem_list.dusty_box import dust_state

X_VALUES = np.logspace(-2, 2, 13)
E_S, RHO, N = 0.06, 1.0, 16


def eps_D(delta):
    sigma = np.sqrt(2*E_S/(3*RHO))
    return sigma**3/delta, delta/sigma


def run(x, closure, ndim, steps=400, nu_t_end=4.0, S3=0.0, S4=3.0):
    """Return times, per-component dispersion of x velocity, and kurtosis history."""
    s = solver(ndim=ndim, n=N, closure=closure, eps=1e-9, ts=1.0, sgs_on=True)
    eD, tau = eps_D(s.Delta)
    nu = x/tau
    s.dust_config["stopping"] = dict(model="constant", t_s=1/nu)
    s.sgs_config.update(production="none", decay=False, pressure=False, d_model="mt25")
    rho = np.ones(s.shape)
    s.U_dust = dust_state(s.closure, 1e-9*rho, np.zeros((3, *s.shape)),
                          0.05*np.sqrt(eD/nu), S3, S4)
    s.e_s = np.full(s.shape, E_S)
    dt = nu_t_end/nu/steps
    t, var, kurt = [0.0], [], []

    def record():
        p = s.closure.primitives(s.U_dust)
        v = p["C"][0, 0] if closure == "gaussian" else p["v"]
        var.append(float(np.mean(v)))
        kurt.append(float(np.mean(p["r"]/p["v"]**2)) if closure == "hyqmom" else np.nan)

    record()
    for _ in range(steps):
        s._local_step(dt)
        t.append(t[-1] + dt)
        record()
    return np.array(t), np.array(var), np.array(kurt), eD*np.arctan(x), nu


def main():
    r = Report("mt25_gaussianisation")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    eD, tau = eps_D(1.0/N)
    for closure, ndims in (("gaussian", (1, 2, 3)), ("hyqmom", (1,))):
        for ndim in ndims:
            rates, plateaus, worst = [], [], 0.0
            for x in X_VALUES:
                t, var, _, D, nu = run(x, closure, ndim)
                exact = var[0]*np.exp(-2*nu*t) + (D/nu)*(1 - np.exp(-2*nu*t))
                worst = max(worst, float(np.max(np.abs(var - exact))/(D/nu)))
                rates.append((var[1] - var[0]*np.exp(-2*nu*t[1]))/(1 - np.exp(-2*nu*t[1]))*nu/eD)
                late = var[-1] - var[0]*np.exp(-2*nu*t[-1])
                plateaus.append(late*nu/eD/(1 - np.exp(-2*nu*t[-1])))
            rates, plateaus = np.array(rates), np.array(plateaus)
            label = f"{closure}/{ndim}D"
            # The source map is exact for constant D.  The 1e-9 loading still drains
            # the reservoir by 3 rho_d D t / e_s <~ 1e-8 over the longest run, which
            # shifts D by that much; 1e-7 allows it with margin, nothing more.
            r.check(label+"/max_dispersion_error_vs_exact_OU", worst, 1e-7)
            r.check(label+"/growth_rate_vs_MT25_max_rel", float(np.max(np.abs(rates/np.arctan(X_VALUES)-1))), 1e-9)
            r.check(label+"/plateau_vs_D_over_nu_max_rel", float(np.max(np.abs(plateaus/np.arctan(X_VALUES)-1))), 1e-7)
            if ndim == 1:
                mk = "o" if closure == "gaussian" else "s"
                ax[0].semilogx(X_VALUES, rates, mk, mfc="none", label=f"measured, {closure}")
    xs = np.logspace(-2.2, 2.2, 200)
    ax[0].semilogx(xs, np.arctan(xs), "k-", lw=1, label=r"MT25: $D/\varepsilon_\Delta=\arctan x$")
    ax[0].axhline(np.pi/2, color="0.6", ls=":", lw=1)
    ax[0].set(xlabel=r"$x=\nu\tau_\Delta$", ylabel=r"$(d\sigma^2/dt)_0\,/\,2\varepsilon_\Delta$",
              title="Heating rate of initially cold dust")
    ax[0].legend(fontsize=8)
    for x, color in zip((0.03, 1.0, 30.0), ("C0", "C1", "C2")):
        t, var, _, D, nu = run(x, "gaussian", 1)
        ax[1].plot(nu*t, var/(D/nu), color=color, label=f"x={x:g}")
        ax[1].plot(nu*t, var[0]*np.exp(-2*nu*t)/(D/nu) + 1 - np.exp(-2*nu*t), "k:", lw=1)
    ax[1].set(xlabel=r"$\nu t$", ylabel=r"$\sigma^2/(D/\nu)$",
              title="Growth and saturation (dotted: exact OU)")
    ax[1].legend(fontsize=8)
    for S3, S4, color in ((1.0, 5.0, "C3"), (0.0, 1.5, "C4"), (-1.5, 6.0, "C5")):
        t, var, kurt, D, nu = run(1.0, "hyqmom", 1, S3=S3, S4=S4)
        ax[2].plot(nu*t, kurt, color=color, label=f"start S3={S3:g}, S4={S4:g}")
        r.check(f"hyqmom/gaussianisation S3={S3:g},S4={S4:g}/final_kurtosis", abs(kurt[-1]-3.0), 1e-3)
    ax[2].axhline(3.0, color="k", ls=":", lw=1)
    ax[2].set(xlabel=r"$\nu t$", ylabel=r"kurtosis $S_4=r/v^2$", title="Relaxation to a Gaussian (HyQMOM)")
    ax[2].legend(fontsize=8)
    r.figure(fig, "mt25_gaussianisation.png")
    r.notes.append(f"Held reservoir e_s={E_S}, rho=1, Delta=1/{N}: sigma_D={np.sqrt(2*E_S/3):.4f}, "
                   f"eps_D={eD:.4g}, tau_Delta={tau:.4g}; x swept over {X_VALUES[0]:g}..{X_VALUES[-1]:g}.")
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
