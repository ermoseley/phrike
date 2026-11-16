"""Self-test of the distributed solvers on the machine they will run on.

    python -m phrike.parallel.selftest                # one process, one device
    srun -n 4 python -m phrike.parallel.selftest      # one rank per GPU

It reports the environment (Torch, CUDA, Triton, MPI, device buffers or host
staging) and checks, on the device the solvers will use:

* the distributed FFT against a round trip and Parseval's identity;
* every Triton kernel against its Torch counterpart (CUDA only);
* one compressible and one incompressible step, Triton tier against Torch tier;
* an exact solution of the incompressible equations across all ranks.

Each check prints PASS or FAIL with the measured error; the exit status is the
number of failures.  A Triton kernel that fails to build is reported here
rather than silently replaced by the Torch fallback, which the solvers would
otherwise do at run time.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import torch

from .comm import Comm
from .compressible import CompressibleSlabStepper
from .fields import random_vector_spectrum
from .grid import SlabGrid3D
from .incompressible import IncompressibleSlabStepper, lawson_coefficients, LAWSON_TABLES

TWO_PI = 2.0 * np.pi


class Report:
    def __init__(self, comm: Comm):
        self.comm = comm
        self.failures = 0
        self.checks = 0

    def say(self, text: str) -> None:
        if self.comm.is_root:
            print(text, flush=True)

    def check(self, name: str, error: float, tolerance: float) -> None:
        """``error`` is this rank's value; the worst over all ranks is judged."""
        error = self.comm.max(float(error) if np.isfinite(error) else float("inf"))
        ok = error <= tolerance
        self.checks += 1
        self.failures += 0 if ok else 1
        self.say(f"  {'PASS' if ok else 'FAIL'}  {name}: {error:.3e} (tolerance {tolerance:.1e})")

    def attempt(self, name: str, fn) -> None:
        """Run a group of checks; an exception on any rank is one failure."""
        failed = 0.0
        try:
            fn()
        except Exception:  # noqa: BLE001 - report and continue with the other groups
            failed = 1.0
            sys.stderr.write(f"[rank {self.comm.rank}] {name} raised:\n{traceback.format_exc()}\n")
        if self.comm.sum(failed) > 0:
            self.checks += 1
            self.failures += 1
            self.say(f"  FAIL  {name}: raised an exception (traceback on stderr)")


def _tolerances(precision: str):
    """(tight, loose): round-off comparisons and accumulated round-off."""
    return (1e-11, 1e-9) if precision == "double" else (2e-4, 2e-3)


def _grid(comm, device, precision, n, **kwargs) -> SlabGrid3D:
    return SlabGrid3D(n, n, n, TWO_PI, TWO_PI, TWO_PI, comm=comm, device=device,
                      precision=precision, **kwargs)


def _compressible_state(grid: SlabGrid3D, system: str, gamma: float) -> torch.Tensor:
    X, Y, Z = grid.xyz_mesh()
    rho = 1.0 + 0.2 * torch.cos(X) * torch.cos(Z)
    u = [torch.sin(X) * torch.cos(Y) * torch.cos(Z), -torch.cos(X) * torch.sin(Y) * torch.cos(Z),
         0.3 * torch.sin(Z) * torch.cos(Y)]
    p = 2.0 + 0.3 * torch.sin(Y)
    kinetic = 0.5 * rho * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    if system == "euler":
        return torch.stack([rho, rho * u[0], rho * u[1], rho * u[2], p / (gamma - 1.0) + kinetic])
    B = [torch.sin(Z) + 0.4 * torch.cos(Y), 0.7 * torch.sin(X) + torch.cos(Z), 0.4 * torch.sin(Y) + 0.7 * torch.cos(X)]
    magnetic = 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    return torch.stack([rho, rho * u[0], rho * u[1], rho * u[2], p / (gamma - 1.0) + kinetic + magnetic] + B)


def check_fft(report: Report, grid: SlabGrid3D, tight: float) -> None:
    comm = grid.comm
    g = torch.Generator().manual_seed(1234 + comm.rank)
    f = torch.randn((3,) + grid.decomp.physical_shape, generator=g, dtype=torch.float64).to(
        dtype=grid.rdtype, device=grid.device)
    S = grid.forward(f, normalized=True)
    smooth = grid.inverse(S)  # the dealiased part of f
    report.check("fft: forward(inverse(S)) == S", (grid.forward(smooth, normalized=True) - S).abs().max().item()
                 / max(comm.max(S.abs().max().item()), 1e-300), tight)
    parseval = grid.spectral_sum((S.real ** 2 + S.imag ** 2).sum(dim=0))
    mean_square = comm.sum(float((smooth.double() ** 2).sum().item())) / grid.n_points
    report.check("fft: Parseval", abs(parseval - mean_square) / mean_square, tight)


def check_triton_kernels(report: Report, grid: SlabGrid3D, tight: float) -> None:
    """Each new Triton kernel against the Torch code path it replaces."""
    from . import kernels_triton as kt

    comm = grid.comm
    for system, gamma in (("euler", 1.4), ("mhd", 5.0 / 3.0)):
        # Floors inside the range of the test state, so that some values count.
        st = CompressibleSlabStepper(grid, system, gamma, 1.05, 2.1, triton=False)
        U = _compressible_state(grid, system, gamma)
        got = kt.state_stats(st.kind, U, gamma, st.rho_min, st.p_min)
        want = st._local_stats_torch(U)
        report.check(f"kernel state_stats[{system}]: the test counts something", 0.0 if want[0] > 0 else 1.0, 0.0)
        report.check(f"kernel state_stats[{system}]: floor count", abs(got[0] - want[0]), 0.0)
        report.check(f"kernel state_stats[{system}]: max wave speed", abs(got[3] - want[3]) / want[3], tight)
        report.check(f"kernel state_stats[{system}]: finite flags", got[1] + got[2] + want[1] + want[2], 0.0)
        bad = U.clone()
        bad[4, 0, 0, 0] = float("nan")
        report.check(f"kernel state_stats[{system}]: sees a NaN", 0.0 if kt.state_stats(st.kind, bad, gamma, 0.0, 0.0)[1] > 0 else 1.0, 0.0)

    for mhd in (True, False):
        st = IncompressibleSlabStepper(grid, 0.03, 0.05, mhd=mhd, triton=False)
        g = torch.Generator().manual_seed(77 + comm.rank)
        V = torch.randn((st.ncomp,) + grid.decomp.physical_shape, generator=g, dtype=torch.float64).to(
            dtype=grid.rdtype, device=grid.device)
        G_ref, rate_ref, bad_ref = st._products_torch(V, True)
        G, rate, bad = kt.imhd_products(V, st.inv_spacing, mhd, True)
        scale = G_ref.abs().max().item()
        label = "imhd" if mhd else "ins"
        report.check(f"kernel products[{label}]: fields", (G - G_ref).abs().max().item() / scale, tight)
        report.check(f"kernel products[{label}]: CFL rate", abs(rate.item() - rate_ref.item()) / rate_ref.item(), tight)
        report.check(f"kernel products[{label}]: finite flag", float(bad.item() + bad_ref.item()), 0.0)
        G2 = kt.imhd_products(V, st.inv_spacing, mhd, False)
        report.check(f"kernel products[{label}]: without statistics", (G2 - G_ref).abs().max().item() / scale, tight)

        # Stage kernel: every (first, final) variant, with and without spherical truncation.
        for spherical in (False, True):
            ref = IncompressibleSlabStepper(grid, 0.03, 0.05, mhd=mhd, triton=False, spherical_truncation=spherical)
            dev = IncompressibleSlabStepper(grid, 0.03, 0.05, mhd=mhd, triton=False, spherical_truncation=spherical)
            spec = lambda seed, rms: random_vector_spectrum(grid, seed, 1.0, 6.0, rms, spectrum="flat")
            parts = [spec(1, 1.0)] + ([spec(2, 0.5)] if mhd else [])
            for s in (ref, dev):
                s.set_spectrum(torch.cat(parts))
                s.acc.copy_(torch.cat([spec(3, 0.2)] + ([spec(4, 0.1)] if mhd else [])))
                s.shat.zero_()
            Gh = grid.forward(G_ref, normalized=False)
            coefs = lawson_coefficients(LAWSON_TABLES["rk4"], 0.01)
            for first, final, coef in ((True, False, coefs[0]), (False, False, coefs[1]), (False, True, coefs[3])):
                ref._stage_torch(Gh * grid.inv_n, coef, first, final)
                huge = 1e300 if grid.rdtype == torch.float64 else 3e38
                wa, wn, t0, tn, ta = coef
                params = torch.tensor([grid.inv_n, 0.03, 0.05 if mhd else 0.0, min(dev.kcut2, huge),
                                       wa, wn, t0, tn, ta], dtype=grid.rdtype, device=grid.device)
                kt.imhd_stage(Gh, dev.uhat, dev.uhat if final else dev.shat, dev.acc,
                              (grid.kx, grid.ky, grid.kz), params, mhd, first, final)
                name = f"kernel stage[{label}{', spherical' if spherical else ''}; first={first}, final={final}]"
                scale = max(comm.max(ref.uhat.abs().max().item()), 1e-300)
                error = max((dev.uhat - ref.uhat).abs().max().item(), (dev.shat - ref.shat).abs().max().item(),
                            (dev.acc - ref.acc).abs().max().item()) / scale
                report.check(name, error, tight)


def check_tiers(report: Report, grid: SlabGrid3D, tight: float, loose: float, triton=True) -> None:
    """Whole steps: the Triton tier against the Torch tier on the same state."""
    comm = grid.comm
    for system, gamma, eta, nu in (("euler", 1.4, 0.0, 0.0), ("mhd", 5.0 / 3.0, 2e-3, 1e-3)):
        states = []
        for tier in (triton, False):
            st = CompressibleSlabStepper(grid, system, gamma, 1e-10, 1e-12, eta=eta, nu=nu, triton=tier)
            if tier and st._triton is None:
                report.check(f"step[{system}]: Triton tier available", 1.0, 0.0)
                return
            st.seed(_compressible_state(grid, system, gamma))
            for _ in range(3):
                st.step(1e-3, "rk4")
            if tier:
                report.check(f"step[{system}]: Triton tier stayed active",
                             0.0 if st._triton is not None and st._stats_kernel is not None else 1.0, 0.0)
            states.append(st.physical())
        scale = comm.max(states[1].abs().max().item())
        report.check(f"step[{system}]: Triton vs Torch, 3 RK4 steps", (states[0] - states[1]).abs().max().item() / scale, loose)

    for mhd in (True, False):
        spectra = []
        for tier in (triton, False):
            st = IncompressibleSlabStepper(grid, 0.02, 0.03, mhd=mhd, triton=tier)
            if tier and st._triton is None:
                report.check("step[incompressible]: Triton tier available", 1.0, 0.0)
                return
            parts = [random_vector_spectrum(grid, 1, 1.0, 4.0, 1.0)]
            if mhd:
                parts.append(random_vector_spectrum(grid, 2, 1.0, 4.0, 0.3))
            st.set_spectrum(torch.cat(parts))
            for _ in range(3):
                st.step(dt=2e-3)
            if tier:
                report.check(f"step[{'imhd' if mhd else 'ins'}]: Triton tier stayed active",
                             0.0 if st._triton is not None else 1.0, 0.0)
            spectra.append(st.uhat.clone())
        scale = comm.max(spectra[1].abs().max().item())
        report.check(f"step[{'imhd' if mhd else 'ins'}]: Triton vs Torch, 3 RK4 steps",
                     (spectra[0] - spectra[1]).abs().max().item() / scale, loose)


def check_exact_solution(report: Report, grid: SlabGrid3D, loose: float) -> None:
    """ABC flow with a parallel field decays as exp(-nu t): exact for the whole pipeline."""
    nu = 0.05
    st = IncompressibleSlabStepper(grid, nu, nu, mhd=True, scheme="rk4", cfl=0.5)
    X, Y, Z = grid.xyz_mesh()
    abc = torch.stack([torch.sin(Z) + 0.4 * torch.cos(Y), 0.7 * torch.sin(X) + torch.cos(Z),
                       0.4 * torch.sin(Y) + 0.7 * torch.cos(X)])
    st.seed(torch.cat([abc, 0.5 * abc]))
    for _ in range(10):
        st.step()
    exact = torch.cat([abc, 0.5 * abc]) * float(np.exp(-nu * st.t))
    report.check("incompressible MHD: exact Beltrami decay, 10 steps",
                 (st.physical() - exact).abs().max().item(), loose)
    d = st.diagnostics()
    report.check("incompressible MHD: div u, div b", max(d["max_div_u_hat"], d["max_div_b_hat"]), loose)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phrike.parallel.selftest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--size", type=int, default=48, help="grid points per side (default 48)")
    parser.add_argument("--precision", choices=["single", "double", "both"], default="both")
    args = parser.parse_args(argv)

    comm = Comm.auto()
    comm.install_abort_hook()
    device = comm.pick_device(args.backend)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    report = Report(comm)
    report.say(f"phrike.parallel self-test: torch {torch.__version__}, device {device}")
    report.say(f"  MPI: {comm.describe()}")
    cuda = device.startswith("cuda")
    triton_ok = False
    if cuda:
        report.say(f"  GPU: {torch.cuda.get_device_name(torch.device(device))}")
        try:
            import triton

            triton_ok = True
            report.say(f"  Triton {triton.__version__}")
        except Exception as exc:  # noqa: BLE001
            report.say(f"  Triton is not importable ({exc}); the solvers will use torch ops")
    else:
        report.say("  no CUDA device: the Triton kernels are not exercised (torch ops only)")

    start = time.perf_counter()
    for precision in (["single", "double"] if args.precision == "both" else [args.precision]):
        tight, loose = _tolerances(precision)
        report.say(f"[{precision} precision, {args.size}^3]")
        grid = _grid(comm, device, precision, args.size)
        report.attempt("fft", lambda: check_fft(report, grid, tight))
        if cuda and triton_ok:
            report.attempt("Triton kernels", lambda: check_triton_kernels(report, grid, tight))
            report.attempt("Triton vs Torch steps", lambda: check_tiers(report, grid, tight, loose))
        report.attempt("exact solution", lambda: check_exact_solution(report, grid, loose))
    report.say(f"{report.checks - report.failures} of {report.checks} checks passed "
               f"in {time.perf_counter() - start:.1f} s")
    return report.failures


if __name__ == "__main__":
    sys.exit(main())
