"""Validation of the 1D circularly polarized Alfven wave (ideal MHD).

Checks, per mhd_plan.md section 7:
  - max|div B| at round-off (1D: dBx/dx, Bx = const),
  - phase speed = c_A = B0/sqrt(rho0) (from the fundamental-mode phase slope),
  - exactness after one full period (L2 error vs analytic),
  - spectral spatial convergence and RK4 temporal order (CPU / double).

Primary run is on Metal (device='mps', float32); a CPU/double run confirms the
machine-precision claims that single precision cannot show.
"""

import numpy as np

from phrike.grid import Grid1D
from phrike.equations import MHDEquations1D
from phrike.solver import SpectralSolverMHD1D
from phrike.problems.problem_list.alfven1d import circularly_polarized_alfven_1d

GAMMA = 5.0 / 3.0
RHO0, P0, B0, B1, M, LX = 1.0, 0.1, 1.0, 0.1, 1, 1.0
C_A = B0 / np.sqrt(RHO0)
PERIOD = LX / (M * C_A)


def make_grid(N, backend, device, precision, filt=True):
    fp = {"enabled": True, "p": 8, "alpha": 36.0} if filt else {"enabled": False}
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


def run(N, t_end, backend, device, precision, cfl=0.4, scheme="rk4", filt=True):
    grid = make_grid(N, backend, device, precision, filt)
    eqs = MHDEquations1D(gamma=GAMMA)
    U0 = circularly_polarized_alfven_1d(grid.x, t=0.0, rho0=RHO0, p0=P0, B0=B0,
                                        b1=B1, m=M, Lx=LX, gamma=GAMMA)
    solver = SpectralSolverMHD1D(grid, eqs, scheme=scheme, cfl=cfl)
    hist = solver.run(U0, 0.0, t_end, output_interval=t_end)
    return grid, eqs, solver, hist


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


def main():
    print("=" * 78)
    print("1D CIRCULARLY POLARIZED ALFVEN WAVE — VALIDATION")
    print(f"  rho0={RHO0} p0={P0} B0={B0} b1={B1}  c_A={C_A:.6f}  period={PERIOD:.6f}")
    print("=" * 78)

    # --- Primary run on Metal (MPS, float32) ---
    try:
        import torch
        mps_ok = torch.backends.mps.is_available()
    except Exception:
        mps_ok = False

    if mps_ok:
        print("\n[Metal / MPS, float32]  N=256, one period")
        grid, eqs, solver, hist = run(256, PERIOD, "torch", "mps", "single", cfl=0.4)
        e = b_perp_error(grid, eqs, solver.U, solver.t)
        dE = abs(hist["energy"][-1] - hist["energy"][0]) / abs(hist["energy"][0])
        print(f"  max|divB|        = {hist['div_b_max'][-1]:.3e}")
        print(f"  L2(B_perp) error = {e:.3e}   (after 1 period, vs analytic)")
        print(f"  energy drift     = {dE:.3e}")
        print(f"  mag-energy drift = {abs(hist['magnetic_energy'][-1]-hist['magnetic_energy'][0]):.3e}")
        ca = measure_phase_speed(256, "torch", "mps", "single")
        print(f"  measured c_A     = {ca:.6f}  (exact {C_A:.6f}, rel err {abs(ca-C_A)/C_A:.2e})")
    else:
        print("\n[Metal / MPS] not available on this machine — skipping")

    # --- CPU / double: machine-precision div(B) + one-period exactness ---
    print("\n[CPU / double]  N=256, one period")
    grid, eqs, solver, hist = run(256, PERIOD, "numpy", None, "double", cfl=0.4)
    e = b_perp_error(grid, eqs, solver.U, solver.t)
    dE = abs(hist["energy"][-1] - hist["energy"][0]) / abs(hist["energy"][0])
    print(f"  max|divB|        = {hist['div_b_max'][-1]:.3e}")
    print(f"  L2(B_perp) error = {e:.3e}")
    print(f"  energy drift     = {dE:.3e}")
    ca = measure_phase_speed(256, "numpy", None, "double")
    print(f"  measured c_A     = {ca:.8f}  (exact {C_A:.8f}, rel err {abs(ca-C_A)/C_A:.2e})")

    # --- Spatial convergence (CPU/double), short time, tiny dt so error ~ spatial ---
    print("\n[CPU / double]  spatial convergence at t=0.25 (cfl=0.1, rk4)")
    prev = None
    for N in (16, 24, 32, 48, 64):
        grid, eqs, solver, hist = run(N, 0.25, "numpy", None, "double", cfl=0.1)
        e = b_perp_error(grid, eqs, solver.U, solver.t)
        msg = f"  N={N:4d}  L2 error = {e:.3e}"
        if prev is not None and e > 0 and prev[1] > 0:
            rate = np.log(prev[1] / e) / np.log(N / prev[0])
            msg += f"   (order ~ {rate:.1f})"
        print(msg)
        prev = (N, e)

    # --- Temporal (RK4) convergence: fixed large N, vary dt via cfl ---
    print("\n[CPU / double]  temporal RK4 convergence at t=0.25, N=128 (filter off)")
    prev = None
    for cfl in (0.4, 0.2, 0.1, 0.05):
        grid, eqs, solver, hist = run(128, 0.25, "numpy", None, "double",
                                      cfl=cfl, scheme="rk4", filt=False)
        e = b_perp_error(grid, eqs, solver.U, solver.t)
        msg = f"  cfl={cfl:5.3f}  L2 error = {e:.3e}"
        if prev is not None and e > 0 and prev[1] > 0:
            rate = np.log(prev[1] / e) / np.log(prev[0] / cfl)
            msg += f"   (order ~ {rate:.1f})"
        print(msg)
        prev = (cfl, e)

    print("\nDone.")


if __name__ == "__main__":
    main()
