"""Throughput benchmark for the 3D hydro solver (Taylor-Green vortex).

Reports zone-updates per second per *accepted* step (one zone update = one
cell advanced by one completed time step), rejected attempts (adaptive
schemes), and simulated time per wall-clock second, for the same loop body the
solver's ``run`` uses: dt capped by the CFL limit, a step, and the bookkeeping
after an accepted step.  For large grids set
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

Examples:
    python scripts/bench_hydro3d.py --N 256 --device cuda --precision double
    python scripts/bench_hydro3d.py --N 128 --device mps --precision single
    python scripts/bench_hydro3d.py --N 64 --steps 20 --save ref.npz
    python scripts/bench_hydro3d.py --N 64 --steps 20 --compare ref.npz
"""

from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from phrike.equations import EulerEquations3D
from phrike.grid import Grid3D
from phrike.problems.problem_list.tgv3d import taylor_green_vortex_3d
from phrike import solver as solver_mod
from phrike.solver import SpectralSolver3D


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@contextlib.contextmanager
def _nvtx(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def build(args):
    filter_params = None if args.no_filter else {"enabled": True, "p": 8, "rate": args.dissipation_rate}
    Nx, Ny, Nz = (int(v) for v in args.shape.split(",")) if args.shape else (args.N,) * 3
    grid = Grid3D(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        Lz=2 * np.pi,
        dealias=True,
        filter_params=filter_params,
        backend="torch" if args.device != "numpy" else "numpy",
        torch_device=None if args.device == "numpy" else args.device,
        precision=args.precision,
        fft_workers=args.fft_workers,
    )
    eqs = EulerEquations3D(gamma=1.4)
    X, Y, Z = grid.xyz_mesh()
    U0 = taylor_green_vortex_3d(X, Y, Z, rho0=1.0, p0=100.0, U0=1.0, k=1, gamma=1.4)
    if isinstance(U0, torch.Tensor):
        U0 = U0.to(dtype=grid.kx.dtype)
    if args.noise > 0:
        # White-noise density/energy perturbation: content above the dealias cutoff.
        rng = np.random.default_rng(1234)
        pert = args.noise * rng.standard_normal((2,) + tuple(U0.shape[1:]))
        if isinstance(U0, torch.Tensor):
            pert = torch.as_tensor(pert, dtype=U0.dtype, device=U0.device)
        U0[0] = U0[0] * (1.0 + pert[0])
        U0[4] = U0[4] * (1.0 + pert[1])
    adaptive = (
        {"enabled": True, "scheme": args.scheme, "rtol": args.rtol, "atol": args.atol}
        if args.scheme in ("rk23", "rk45", "rk78") else None
    )
    solver = SpectralSolver3D(
        grid=grid, equations=eqs, scheme=args.scheme, cfl=args.cfl, adaptive_config=adaptive,
        spectral_fast_path=not args.legacy,
    )
    return grid, eqs, solver, U0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--shape", default=None, help="Nx,Ny,Nz (overrides --N)")
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu | numpy")
    ap.add_argument("--precision", default="double")
    ap.add_argument("--scheme", default="rk4")
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--fixed-dt", type=float, default=None,
                    help="use a fixed dt (skips CFL); needed for --save/--compare")
    ap.add_argument("--no-filter", action="store_true", help="disable spectral dissipation")
    ap.add_argument("--dissipation-rate", type=float, default=40.0)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--noise", type=float, default=0.0, help="relative white-noise IC perturbation")
    ap.add_argument("--legacy", action="store_true", help="original complex-FFT path")
    ap.add_argument("--fft-workers", type=int, default=1)
    ap.add_argument("--nvtx", action="store_true")
    ap.add_argument("--save", default=None, help="save final state to .npz")
    ap.add_argument("--compare", default=None, help="compare final state to .npz")
    args = ap.parse_args()

    dev = args.device if args.device != "numpy" else "cpu"
    grid, eqs, solver, U0 = build(args)
    # As SpectralSolver3D.run: start in the retained (dealiased) subspace.
    solver.U = grid.project_dealiased(U0)
    solver_mod._begin_run(solver, 0.0, 1e9)
    nvtx = args.nvtx and dev == "cuda"

    def one_step(dt):
        """One attempt; returns (next dt proposal, accepted)."""
        t_before = solver.t
        if args.fixed_dt is not None:
            solver.U = solver.step(solver.U, args.fixed_dt)
            solver.t += args.fixed_dt
            return args.fixed_dt, True
        dt = min(dt, solver._stability_limit(solver.U))  # as _cap_proposal
        if solver.adaptive_enabled:
            dt = solver._adaptive_run_step(dt, 0, None)
        else:
            dt = solver._fixed_run_step(dt, 0, None)
        return dt, solver.t > t_before

    dt = args.fixed_dt if args.fixed_dt is not None else float("inf")
    for i in range(args.warmup):
        with _nvtx(f"warmup{i}", nvtx):
            dt, _ = one_step(dt)
    _sync(dev)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    accepted = attempts = 0
    t_start = solver.t
    t0 = time.perf_counter()
    while accepted < args.steps:
        with _nvtx(f"step{attempts}", nvtx):
            dt, ok = one_step(dt)
        attempts += 1
        accepted += int(ok)
    _sync(dev)
    elapsed = time.perf_counter() - t0

    ncell = int(np.prod(solver.U.shape[1:]))
    zups = ncell * accepted / elapsed
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    print(
        f"N={args.shape or str(args.N) + '^3'} device={args.device} precision={args.precision} scheme={args.scheme} "
        f"path={'legacy' if args.legacy else 'fast'} "
        f"accepted={accepted} rejected={attempts - accepted} time={elapsed:.4f}s  "
        f"per-accepted-step={1e3 * elapsed / accepted:.2f} ms  "
        f"throughput={zups / 1e6:.3f} Mzups  sim_time/s={(solver.t - t_start) / elapsed:.4g}  "
        f"peak_mem={mem:.2f} GiB  t={solver.t:.6f}"
    )

    U = solver.U
    U_np = U.detach().cpu().numpy() if isinstance(U, torch.Tensor) else np.asarray(U)
    if not np.all(np.isfinite(U_np)):
        print("WARNING: non-finite values in final state")
    if args.save:
        np.savez(args.save, U=U_np, t=solver.t)
        print(f"saved {args.save}")
    if args.compare:
        ref = np.load(args.compare)
        Ur = ref["U"]
        err = np.abs(U_np - Ur).max(axis=(1, 2, 3))
        scale = np.abs(Ur).max(axis=(1, 2, 3))
        print("max |dU| per component:", " ".join(f"{e:.3e}" for e in err))
        print("relative (to max|U|):  ", " ".join(f"{e:.3e}" for e in err / np.maximum(scale, 1e-300)))
        print(f"t ref={float(ref['t']):.17g} now={solver.t:.17g}")


if __name__ == "__main__":
    main()
