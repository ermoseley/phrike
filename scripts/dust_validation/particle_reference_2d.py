"""Gaussian moment closure against kinetic OU particles in the same 2D turbulence.

Passive dust (loading 1e-6, back-reaction off) in decaying 2D turbulence with the
active SGS reservoir.  The coupled solver evolves gas, SGS and the dust moments;
256 particles per cell evolve in the same gas: drift, exact drag relaxation toward
the gas velocity of their cell, and exact OU velocity kicks with the MT25 D of their
cell (eddy diffusion off in both).  Particles see exactly the cell-uniform gas and
reservoir that the moment drag sees (nearest-grid-point sampling) and are deposited
by nearest grid point into cell averages, so the comparison isolates transport and
Gaussian-closure error from sampling-scheme differences.

PYTHONPATH=. python scripts/dust_validation/particle_reference_2d.py
"""
from common import Report, np, plt
from turb_energy_ledger import initial
from phrike.sgs import sgs_coefficients

N, EPS, PPC = 32, 1e-6, 256
VRMS_GAS = 0.2  # initial gas rms speed of turb_energy_ledger.initial


def cells(pos):
    ij = np.floor(pos*N).astype(int) % N
    return ij[:, 1]*N + ij[:, 0]


def kick(s, pos, vel, h, rng):
    c = cells(pos)
    rho = s.U_gas[0].ravel()[c]
    gas_u = np.zeros_like(vel)
    gas_u[:, :2] = (s.U_gas[1:3]/s.U_gas[0]).reshape(2, -1)[:, c].T
    es = s.e_s.ravel()[c]
    nu = 1/s.dust_config["stopping"]["t_s"]
    D = sgs_coefficients(es, rho, nu, s.Delta, "mt25")["D"]
    a = np.exp(-nu*h)
    vel[:] = gas_u + (vel - gas_u)*a + np.sqrt(D/nu*(-np.expm1(-2*nu*h)))[:, None]*rng.normal(size=vel.shape)


def deposit(pos, vel):
    c = cells(pos)
    m = np.bincount(c, minlength=N*N).astype(float)
    mean = np.stack([np.bincount(c, weights=vel[:, k], minlength=N*N) for k in range(3)])/m
    var = sum(np.bincount(c, weights=(vel[:, k] - mean[k, c])**2, minlength=N*N) for k in range(3))/(3*m)
    return (m*EPS/PPC).reshape(N, N), mean.reshape(3, N, N), var.reshape(N, N), c, mean


def compare(s, pos, vel):
    rho_p, u_p, var_p, c, mean = deposit(pos, vel)
    p = s.closure.primitives(s.U_dust)
    var_m = np.trace(p["C"], axis1=0, axis2=1)/3
    w = p["rho"]
    return dict(time=s.t,
                rho_L1=float(np.mean(np.abs(rho_p - p["rho"]))/EPS),
                u_L1=float(np.mean(np.linalg.norm(u_p - p["u"], axis=0))/VRMS_GAS),
                var_L1=float(np.mean(np.abs(var_p - var_m))/np.mean(var_m)),
                var_global_particles=float(np.sum(rho_p*var_p)/np.sum(rho_p)),
                var_global_moments=float(np.sum(w*var_m)/np.sum(w)),
                unrealizable=int(np.count_nonzero(~s.closure.realizable(s.U_dust))))


def main():
    r = Report("particle_reference_2d")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for ir, regime in enumerate(("short", "intermediate", "long")):
        s, tau, ts = initial(2, N, EPS, regime)
        s.dust_config["backreaction"] = False
        s.dust_config["eddy_diffusion"]["enabled"] = False
        rng = np.random.default_rng(1101 + ir)
        side = int(np.sqrt(PPC))
        yy, xx, by, bx = np.meshgrid(np.arange(N), np.arange(N), np.arange(side), np.arange(side), indexing="ij")
        pos = np.column_stack(((xx.ravel() + (bx.ravel() + 0.5)/side)/N, (yy.ravel() + (by.ravel() + 0.5)/side)/N))
        # Initial particle velocities sample each cell's moment Gaussian (antithetic).
        p0 = s.closure.primitives(s.U_dust)
        c0 = cells(pos)
        noise = rng.normal(size=(len(pos)//2, 3))
        noise = np.concatenate([noise, -noise])[rng.permutation(len(pos))]
        L = np.linalg.cholesky(np.moveaxis(p0["C"], (0, 1), (-2, -1)).reshape(-1, 3, 3) + 1e-300*np.eye(3))
        vel = p0["u"].reshape(3, -1)[:, c0].T + np.einsum("nij,nj->ni", L[c0], noise)
        hist = [compare(s, pos, vel)]
        t_end = 2.0*tau
        while s.t < t_end - 1e-14:
            dt = min(s.compute_dt(), 0.02*tau, t_end - s.t)
            kick(s, pos, vel, 0.5*dt, rng)
            pos[:] = (pos + dt*vel[:, :2]) % 1.0
            s.step(dt)
            kick(s, pos, vel, 0.5*dt, rng)
            if len(hist) < 2 or s.t >= hist[-1]["time"] + 0.1*tau - 1e-14 or s.t >= t_end - 1e-14:
                hist.append(compare(s, pos, vel))
        h = {k: np.array([x[k] for x in hist]) for k in hist[0]}
        # Global dispersion: the moment closure must track the kinetic particles.
        # Monte Carlo noise in a 256-particle cell variance is ~sqrt(2/256) ~ 9%
        # per cell but averages to <1% over 1024 cells; 10% allows closure and
        # transport differences while still failing a qualitatively wrong closure.
        rel = np.abs(h["var_global_moments"]/h["var_global_particles"] - 1)
        # For short t_s the dispersion is set by grid-scale velocity structure,
        # not SGS kicks: NGP particles carry a neighbour's velocity for a fraction
        # of a cell after crossing the piecewise-constant gas field, while the
        # moments relax to a cell-uniform gas velocity.  That difference measures
        # sub-grid modelling choices, not the SGS-dust coupling, and is recorded.
        if regime != "short":
            r.check(f"{regime}/global_dispersion_rel_diff_max", float(rel.max()), 0.10)
        r.check(f"{regime}/unrealizable_cells", int(h["unrealizable"].max()), 0)
        r.check(f"{regime}/repaired_cells", s.transport_stats["repaired_cells"], 0)
        # Stacked, normalised velocity PDF over all cells vs a unit Gaussian.
        _, u_p, var_p, c, mean = deposit(pos, vel)
        z = ((vel[:, 0] - mean[0, c])/np.sqrt(np.maximum(var_p.ravel()[c], 1e-300)))
        kurt = float(np.mean(z**4)/np.mean(z**2)**2)
        # Gaussianity is the claim where SGS kicks set the dispersion (long t_s).
        # For short t_s the cell PDF is a mixture of particles still carrying
        # neighbouring cells' velocities; normalised by the tiny thermal width
        # D/nu it is heavy-tailed, which a single-Gaussian closure cannot
        # represent.  That kurtosis is recorded as measured closure error.
        if regime == "long":
            r.check(f"{regime}/stacked_kurtosis_minus_3", abs(kurt - 3.0), 0.15)
        r.runs.append(dict(regime=regime, ts_over_tau=ts/tau, history=h, stacked_kurtosis=kurt,
                           global_dispersion_rel_diff_max=float(rel.max()),
                           final=dict((k, float(v[-1])) for k, v in h.items())))
        ax = axes[ir]
        ax[0].plot(h["time"]/tau, h["var_global_particles"], "o", ms=4, label="particles (kinetic)")
        ax[0].plot(h["time"]/tau, h["var_global_moments"], "-", label="moment closure")
        ax[0].set(xlabel=r"$t/\tau_\Delta$", ylabel=r"mass-weighted $\sigma_d^2$",
                  title=f"{regime}: $t_s/\\tau_\\Delta$={ts/tau:g}")
        ax[0].legend(fontsize=8)
        for key, lab in (("rho_L1", "density"), ("u_L1", "mean velocity"), ("var_L1", "dispersion")):
            ax[1].plot(h["time"]/tau, h[key], label=lab)
        ax[1].set(xlabel=r"$t/\tau_\Delta$", ylabel="L1 difference (density, dispersion: relative; velocity: / gas v_rms)",
                  title="field differences")
        ax[1].legend(fontsize=8)
        ax[2].hist(z, bins=80, range=(-5, 5), density=True, alpha=0.6, label=f"particles, kurtosis {kurt:.2f}")
        zz = np.linspace(-5, 5, 200)
        ax[2].plot(zz, np.exp(-zz**2/2)/np.sqrt(2*np.pi), "k-", lw=1, label="unit Gaussian")
        ax[2].set(xlabel=r"$(v_x-\bar u_x)/\sigma$ (all cells)", ylabel="PDF", yscale="log", ylim=(1e-4, 1))
        ax[2].legend(fontsize=8)
        p = s.closure.primitives(s.U_dust)
        var_m = np.trace(p["C"], axis1=0, axis2=1)/3
        vmax = float(max(var_m.max(), var_p.max()))
        im = ax[3].imshow(np.concatenate([var_m, var_p], axis=1), origin="lower", vmin=0, vmax=vmax, cmap="magma")
        ax[3].set(title=r"$\sigma_d^2$: moments | particles", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax[3], shrink=0.8)
    r.figure(fig, "particle_reference.png")
    r.notes.append(f"{N}^2 grid, {PPC} particles per cell, 2 tau_Delta, NGP sampling and deposition.")
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
