"""Diagnostic validation of the 1-D circularly polarized Alfvén wave.

Checks:
  - max|div B| at round-off (1D: dBx/dx, Bx = const),
  - phase speed = c_A = B0/sqrt(rho0) (from the fundamental-mode phase slope),
  - exactness after one full period (L2 error vs analytic),
  - spatial error at a fixed small dt (CPU / double),
  - RK4 temporal order at fixed N (CPU / double).

The spatial and temporal studies are separate because dt scaled with dx
would fold the RK4 time error into a "spatial" table. The wave is a single
Fourier mode, so its spatial error is zero once the grid retains the mode and
the fixed-dt table shows the round-off floor, not a finite convergence order.

The CPU/double run is the default. ``--mps`` adds an optional Metal
(float32) run when PyTorch reports MPS as available.
"""

import argparse

import numpy as np

from phrike.grid import Grid1D
from phrike.equations import MHDEquations1D
from phrike.solver import SpectralSolverMHD1D
from phrike.problems.problem_list.alfven1d import circularly_polarized_alfven_1d

GAMMA = 5.0 / 3.0
RHO0, P0, B0, B1, M, LX = 1.0, 0.1, 1.0, 0.1, 1, 1.0
C_A = B0 / np.sqrt(RHO0)
PERIOD = LX / (M * C_A)


def make_grid(N, backend, device, precision, filt=False):
    fp = (
        {
            "enabled": True,
            "order": 8,
            "e_folding_time_at_cutoff": 1.0 / 36.0,
        }
        if filt
        else {"enabled": False}
    )
    return Grid1D(N=N, Lx=LX, dealias=True, filter_params=fp,
                  backend=backend, torch_device=device, precision=precision)


def to_np(a):
    return a.detach().cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)


def b_perp_error(grid, eqs, U, t):
    """L2 error of (By,Bz) vs analytic at time t (numpy)."""
    x = to_np(grid.x)
    _, _, _, _, _, _, By, Bz = eqs.primitive(U)
    By, Bz = to_np(By), to_np(Bz)
    Ua = circularly_polarized_alfven_1d(x, t=t, rho0=RHO0, p0=P0, B0=B0, b1=B1,
                                        m=M, Lx=LX, gamma=GAMMA)
    eqn = MHDEquations1D(gamma=GAMMA)
    _, _, _, _, _, _, Bya, Bza = eqn.primitive(Ua)
    return float(np.sqrt(np.mean((By - Bya) ** 2 + (Bz - Bza) ** 2)))


def run(N, t_end, backend, device, precision, cfl=0.4, scheme="rk4", filt=False):
    grid = make_grid(N, backend, device, precision, filt)
    eqs = MHDEquations1D(gamma=GAMMA)
    U0 = circularly_polarized_alfven_1d(grid.x, t=0.0, rho0=RHO0, p0=P0, B0=B0,
                                        b1=B1, m=M, Lx=LX, gamma=GAMMA)
    solver = SpectralSolverMHD1D(grid, eqs, scheme=scheme, cfl=cfl)
    hist = solver.run(U0, 0.0, t_end, output_interval=t_end)
    return grid, eqs, solver, hist


def run_fixed_dt(N, t_end, dt, backend="numpy", device=None, precision="double",
                 scheme="rk4", filt=False):
    """Advance to ``t_end`` with a step independent of N.

    ``dt`` is rounded down to ``t_end / ceil(t_end / dt)`` so that the final
    time is hit exactly. Returns ``(grid, eqs, U, nsteps)``.
    """
    grid = make_grid(N, backend, device, precision, filt)
    eqs = MHDEquations1D(gamma=GAMMA)
    U = circularly_polarized_alfven_1d(grid.x, t=0.0, rho0=RHO0, p0=P0, B0=B0,
                                       b1=B1, m=M, Lx=LX, gamma=GAMMA)
    solver = SpectralSolverMHD1D(grid, eqs, scheme=scheme, cfl=0.4)
    nsteps = max(1, int(np.ceil(t_end / dt - 1e-12)))
    h = t_end / nsteps
    U = grid.project_dealiased(U)
    for _ in range(nsteps):
        U = solver.step(U, h)
    return grid, eqs, U, nsteps


def spatial_convergence(Ns=(4, 6, 8, 12, 16, 24, 32, 64), t_end=0.25, dt=2e-4,
                        backend="numpy", device=None, precision="double"):
    """L2(B_perp) error versus N at one fixed dt. Returns [(N, error), ...]."""
    rows = []
    for N in Ns:
        grid, eqs, U, _ = run_fixed_dt(N, t_end, dt, backend, device, precision)
        rows.append((N, b_perp_error(grid, eqs, U, t_end)))
    return rows


def temporal_convergence(N=32, t_end=0.25, dts=(8e-3, 4e-3, 2e-3, 1e-3),
                         backend="numpy", device=None, precision="double"):
    """L2(B_perp) error versus dt at one fixed N. Returns [(dt, error), ...]."""
    rows = []
    for dt in dts:
        grid, eqs, U, nsteps = run_fixed_dt(N, t_end, dt, backend, device, precision)
        rows.append((t_end / nsteps, b_perp_error(grid, eqs, U, t_end)))
    return rows


def measure_phase_speed(N=256, backend="numpy", device=None, precision="double"):
    """Fit c_A from the phase of the fundamental +k Fourier mode of By(t)."""
    grid = make_grid(N, backend, device, precision)
    eqs = MHDEquations1D(gamma=GAMMA)
    U = circularly_polarized_alfven_1d(grid.x, t=0.0, rho0=RHO0, p0=P0, B0=B0,
                                       b1=B1, m=M, Lx=LX, gamma=GAMMA)
    solver = SpectralSolverMHD1D(grid, eqs, scheme="rk4", cfl=0.3)
    solver.U = to_np(U) if backend == "numpy" else U
    x = to_np(grid.x)
    k = 2.0 * np.pi * M / LX
    times = np.linspace(0.0, 0.5 * PERIOD, 11)
    phases = []
    solver.t = 0.0
    Ucur = U
    for i in range(len(times)):
        if i > 0:
            # integrate from times[i-1] to times[i]
            tloc, tend = times[i - 1], times[i]
            dt0 = solver.compute_dt(Ucur)
            tt = tloc
            while tt < tend - 1e-12:
                dt = min(dt0, tend - tt)
                Ucur = solver.step(Ucur, dt)
                tt += dt
                dt0 = solver.compute_dt(Ucur)
        By = to_np(eqs.primitive(Ucur)[6])
        mode = np.sum(By * np.exp(-1j * k * x))   # +k coefficient
        phases.append(np.angle(mode))
    phases = np.unwrap(phases)
    slope = np.polyfit(times, phases, 1)[0]   # d(phase)/dt = -omega
    omega = -slope
    return omega / k   # c_A


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnostic validation of the 1-D circularly polarized "
                    "Alfven wave (div B, phase speed, one-period exactness, "
                    "fixed-dt spatial error, fixed-N temporal order).",
    )
    parser.add_argument("--mps", action="store_true",
                        help="also run the primary N=256 case on Metal (float32) "
                             "if PyTorch reports MPS as available")
    parser.add_argument("--n", type=int, default=256, metavar="N",
                        help="resolution of the one-period runs (default: 256)")
    parser.add_argument("--spatial-dt", type=float, default=2e-4, metavar="DT",
                        help="fixed step for the spatial table (default: 2e-4)")
    parser.add_argument("--skip-convergence", action="store_true",
                        help="skip the spatial and temporal tables")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    N0 = args.n
    print("=" * 78)
    print("1D CIRCULARLY POLARIZED ALFVEN WAVE — VALIDATION")
    print(f"  rho0={RHO0} p0={P0} B0={B0} b1={B1}  c_A={C_A:.6f}  period={PERIOD:.6f}")
    print("=" * 78)

    # --- Optional Metal (MPS, float32) run ---
    if args.mps:
        try:
            import torch
            mps_ok = torch.backends.mps.is_available()
        except Exception:
            mps_ok = False
        if mps_ok:
            print(f"\n[Metal / MPS, float32]  N={N0}, one period")
            grid, eqs, solver, hist = run(N0, PERIOD, "torch", "mps", "single", cfl=0.4)
            e = b_perp_error(grid, eqs, solver.U, solver.t)
            dE = abs(hist["energy"][-1] - hist["energy"][0]) / abs(hist["energy"][0])
            print(f"  max|divB|        = {hist['div_b_max'][-1]:.3e}")
            print(f"  L2(B_perp) error = {e:.3e}   (after 1 period, vs analytic)")
            print(f"  energy drift     = {dE:.3e}")
            print(f"  mag-energy drift = {abs(hist['magnetic_energy'][-1]-hist['magnetic_energy'][0]):.3e}")
            ca = measure_phase_speed(N0, "torch", "mps", "single")
            print(f"  measured c_A     = {ca:.6f}  (exact {C_A:.6f}, rel err {abs(ca-C_A)/C_A:.2e})")
        else:
            print("\n[Metal / MPS] requested but not available — skipping")

    # --- CPU / double: machine-precision div(B) + one-period exactness ---
    print(f"\n[CPU / double]  N={N0}, one period")
    grid, eqs, solver, hist = run(N0, PERIOD, "numpy", None, "double", cfl=0.4)
    e = b_perp_error(grid, eqs, solver.U, solver.t)
    dE = abs(hist["energy"][-1] - hist["energy"][0]) / abs(hist["energy"][0])
    print(f"  max|divB|        = {hist['div_b_max'][-1]:.3e}")
    print(f"  L2(B_perp) error = {e:.3e}")
    print(f"  energy drift     = {dE:.3e}")
    ca = measure_phase_speed(N0, "numpy", None, "double")
    print(f"  measured c_A     = {ca:.8f}  (exact {C_A:.8f}, rel err {abs(ca-C_A)/C_A:.2e})")

    if args.skip_convergence:
        print("\nDone.")
        return

    # --- Spatial error at fixed dt (CPU/double) ---
    print(f"\n[CPU / double]  spatial error at t=0.25, fixed dt={args.spatial_dt:g} (rk4)")
    for N, err in spatial_convergence(dt=args.spatial_dt):
        print(f"  N={N:4d}  L2 error = {err:.3e}")
    print("  The wave is one Fourier mode: once N retains it, the spatial error is zero\n"
          "  and the values above are the round-off (plus residual time-error) floor,\n"
          "  so no finite convergence order is expected or reported.")

    # --- Temporal (RK4) convergence at fixed N ---
    print("\n[CPU / double]  temporal RK4 convergence at t=0.25, N=32 (fixed dt, filter off)")
    prev = None
    for dt, err in temporal_convergence():
        msg = f"  dt={dt:9.3e}  L2 error = {err:.3e}"
        if prev is not None and err > 1e-13 and prev[1] > 1e-13:
            rate = np.log(prev[1] / err) / np.log(prev[0] / dt)
            msg += f"   (order ~ {rate:.1f})"
        print(msg)
        prev = (dt, err)

    print("\nDone.")


if __name__ == "__main__":
    main()
