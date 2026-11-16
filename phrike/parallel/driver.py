"""Run loop of the distributed solvers: configuration, time stepping, output, restart.

``Simulation`` reads the same YAML layout as the serial problems (``grid``,
``physics``, ``integration``, ``initial_conditions``, ``io``, ``monitoring``)
plus ``forcing`` and ``parallel``.  The system comes from the problem: the
compressible ones (``euler``, ``mhd``) step with ``CompressibleSlabStepper``,
the incompressible ones (``ins``, ``imhd``) with ``IncompressibleSlabStepper``.

Everything rank 0 prints or writes is a reduction over all ranks; state goes
to disk as one file per rank (``parallel.io``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..adaptive import create_adaptive_stepper
from .comm import Comm, Timers
from .compressible import CompressibleSlabStepper
from .forcing import BandForcing
from .grid import SlabGrid3D
from .incompressible import LAWSON_TABLES, IncompressibleSlabStepper
from .io import History, field_names, read_meta, read_spectrum, write_slice, write_snapshot
from .problems import COMPRESSIBLE, get_problem

_ALLOWED = {
    "grid": {"Nx", "Ny", "Nz", "Lx", "Ly", "Lz", "dealias", "precision", "fft_workers"},
    "physics": {"gamma", "mhd", "viscosity", "resistivity", "magnetic_prandtl", "spherical_truncation"},
    "equations": {"gamma"},
    "integration": {"t0", "t_end", "cfl", "scheme", "output_interval", "checkpoint_interval",
                    "spectral_dissipation", "modal_filter", "sbp_dissipation", "adaptive",
                    "max_steps", "max_wall_minutes", "dt", "dt_max"},
    "io": {"outdir", "snapshots", "slices", "spectra", "keep_checkpoints"},
    "monitoring": {"enabled", "step_interval", "output_file", "include_conservation", "include_timestep",
                   "include_time", "include_velocity_stats", "include_density_stats"},
    "forcing": {"type", "kmin", "kmax", "amplitude", "power", "correlation_time",
                "compressive_fraction", "seed", "profile", "enabled"},
    "parallel": {"gpu_aware_mpi", "transpose", "fft_batch", "triton", "profile", "strict_clamp"},
    "adaptive": {"enabled", "scheme", "rtol", "atol", "safety_factor", "min_dt_factor",
                 "max_dt_factor", "max_rejections", "fallback_scheme", "dt_min"},
}
_MHD_ALLOWED = {"resistivity", "viscosity", "project_each_substage", "divergence_clean", "rho_min",
                "p_min", "resistivity_max", "cfl_diff", "ambipolar", "hall", "shearing_sheet"}
_TOP_LEVEL = set(_ALLOWED) | {"problem", "initial_conditions", "runtime", "video", "tracers",
                              "artificial_viscosity", "gravity", "initial_conditions_smoothing"}


def _warn_unknown_keys(config: Dict[str, Any]) -> None:
    def note(key: str) -> None:
        warnings.warn(f"Unknown configuration key '{key}' is ignored", UserWarning, stacklevel=4)

    for key in sorted(set(config) - _TOP_LEVEL):
        note(key)
    for section, allowed in _ALLOWED.items():
        block = config.get(section)
        if isinstance(block, dict):
            for key in sorted(set(block) - allowed):
                note(f"{section}.{key}")
    mhd = (config.get("physics") or {}).get("mhd")
    if isinstance(mhd, dict):
        for key in sorted(set(mhd) - _MHD_ALLOWED):
            note(f"physics.mhd.{key}")
    nested = (config.get("integration") or {}).get("adaptive")
    if isinstance(nested, dict):
        for key in sorted(set(nested) - _ALLOWED["adaptive"]):
            note(f"integration.adaptive.{key}")


def _enabled(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("enabled", False))


class Simulation:
    """One distributed run of a problem from ``parallel.problems``."""

    def __init__(self, problem: str, config: Dict[str, Any], comm: Optional[Comm] = None,
                 backend: str = "auto", restart: Optional[str] = None, outdir: Optional[str] = None,
                 precision: Optional[str] = None, max_steps: Optional[int] = None,
                 max_wall_minutes: Optional[float] = None, profile: Optional[bool] = None,
                 quiet: bool = False):
        self.config = config
        self.problem = get_problem(problem)
        self.system = self.problem.system
        self.compressible = self.system in COMPRESSIBLE
        par = config.get("parallel") or {}
        if comm is None:
            comm = Comm.auto(gpu_aware=par.get("gpu_aware_mpi"), transpose=par.get("transpose"))
        self.comm = comm
        self.quiet = bool(quiet)
        if comm.is_root:
            _warn_unknown_keys(config)

        device = comm.pick_device(backend)
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        elif comm.size > 1 and "OMP_NUM_THREADS" not in os.environ:
            torch.set_num_threads(1)  # one core per rank unless told otherwise
        profile = bool(par.get("profile", False)) if profile is None else bool(profile)
        self.timers = Timers(enabled=True, sync=profile)

        gcfg = config.get("grid") or {}
        if not bool(gcfg.get("dealias", True)):
            raise ValueError("The distributed solvers always dealias (2/3 rule); remove grid.dealias: false")
        integ = config.get("integration") or {}
        filter_params = {"spectral_dissipation": integ.get("spectral_dissipation", {"enabled": False})}
        if _enabled(filter_params["spectral_dissipation"]) and not self.compressible:
            raise ValueError("integration.spectral_dissipation applies to the compressible solvers; "
                             "the incompressible solver has physical viscosity and resistivity")
        fft_batch = par.get("fft_batch", 4)
        self.grid = SlabGrid3D(
            int(gcfg["Nx"]), int(gcfg["Ny"]), int(gcfg["Nz"]),
            float(gcfg.get("Lx", 2.0 * np.pi)), float(gcfg.get("Ly", 2.0 * np.pi)),
            float(gcfg.get("Lz", 2.0 * np.pi)), comm=comm, device=device,
            precision=str(precision or gcfg.get("precision", "double")),
            filter_params=filter_params, fft_batch=fft_batch, timers=self.timers,
        )

        self.t0 = float(integ.get("t0", 0.0))
        self.t_end = float(integ["t_end"])
        self.cfl = float(integ.get("cfl", 0.3 if self.compressible else 0.5))
        self.scheme = str(integ.get("scheme", "rk4")).lower()
        self.output_interval = float(integ.get("output_interval", 0.0))
        self.checkpoint_interval = float(integ.get("checkpoint_interval", 0.0))
        self.max_steps = max_steps if max_steps is not None else integ.get("max_steps")
        wall = max_wall_minutes if max_wall_minutes is not None else integ.get("max_wall_minutes")
        self.max_wall = None if wall is None else 60.0 * float(wall)
        self.dt_fixed = integ.get("dt")
        self.dt_max = float(integ.get("dt_max", float("inf")))

        iocfg = config.get("io") or {}
        self.outdir = os.path.abspath(outdir or iocfg.get("outdir") or f"outputs/{problem}_parallel")
        self.write_snapshots = bool(iocfg.get("snapshots", True))
        self.write_slices = bool(iocfg.get("slices", True))
        self.write_spectra = bool(iocfg.get("spectra", True))
        self.keep_checkpoints = int(iocfg.get("keep_checkpoints", 2))
        mon = config.get("monitoring") or {}
        self.monitor_enabled = bool(mon.get("enabled", True))
        self.monitor_interval = max(1, int(mon.get("step_interval", 10)))

        fcfg = config.get("forcing") or {}
        self.forcing = None
        if fcfg and fcfg.get("enabled", True) and str(fcfg.get("type", "none")).lower() != "none":
            self.forcing = BandForcing(self.grid, fcfg, incompressible=not self.compressible)

        triton = par.get("triton")
        if self.compressible:
            self._setup_compressible(config, integ, par, triton)
        else:
            self._setup_incompressible(config, triton)

        self.step_count = 0
        self._last_output_time = None
        self.out_index = 0
        self.chk_index = 0
        self.t = self.t0
        if comm.is_root:
            os.makedirs(self.outdir, exist_ok=True)
        comm.barrier()
        self.restarted = bool(restart)
        if restart:
            self._restart(restart)
        else:
            self._initialize()
        self.history = History(os.path.join(self.outdir, "history.csv"),
                               enabled=comm.is_root, append=bool(restart))

    # --- setup --------------------------------------------------------------

    def log(self, message: str) -> None:
        if self.comm.is_root and not self.quiet:
            print(message, flush=True)

    def _gamma(self, config: Dict[str, Any]) -> float:
        for section in ("equations", "physics"):
            block = config.get(section) or {}
            if "gamma" in block:
                return float(block["gamma"])
        raise KeyError("Configuration must contain either 'equations.gamma' or 'physics.gamma'")

    def _setup_compressible(self, config, integ, par, triton) -> None:
        self.gamma = self._gamma(config)
        mhd = (config.get("physics") or {}).get("mhd") or {}
        eta = nu = 0.0
        rho_min, p_min = 1e-10, 1e-12
        self.cfl_diff = 1.0
        if self.system == "mhd":
            for name in ("hall", "ambipolar", "shearing_sheet"):
                if _enabled(mhd.get(name)) or (name == "shearing_sheet" and mhd.get(name)):
                    raise NotImplementedError(f"physics.mhd.{name} is not available on the distributed path")
            if str(mhd.get("divergence_clean", "projection")) != "projection":
                raise ValueError("The distributed MHD path needs divergence_clean: projection")
            eta = float(mhd.get("resistivity", 0.0))
            if mhd.get("resistivity_max") is not None:
                eta = min(eta, float(mhd["resistivity_max"]))
            nu = float(mhd.get("viscosity", 0.0))
            rho_min, p_min = float(mhd.get("rho_min", 1e-10)), float(mhd.get("p_min", 1e-12))
            self.cfl_diff = float(mhd.get("cfl_diff", 1.0))
        for name in ("artificial_viscosity", "gravity"):
            if _enabled(config.get(name)):
                raise NotImplementedError(f"{name} is not available on the distributed path")
        self.eta, self.nu = eta, nu
        self.stepper = CompressibleSlabStepper(
            self.grid, self.system, self.gamma, rho_min, p_min, eta=eta, nu=nu,
            triton=triton, strict_clamp=bool(par.get("strict_clamp", False)))
        nested, top = integ.get("adaptive"), config.get("adaptive")
        if nested is not None and top is not None:
            raise ValueError("Both integration.adaptive and top-level adaptive are set. Keep only one of them.")
        adaptive = nested if nested is not None else top
        self.adaptive = None
        if _enabled(adaptive):
            method = str(adaptive.get("scheme", self.scheme if self.scheme in ("rk23", "rk45", "rk78") else "rk45"))
            self.adaptive = create_adaptive_stepper(
                method, rtol=float(adaptive.get("rtol", 1e-6)), atol=float(adaptive.get("atol", 1e-8)),
                safety_factor=float(adaptive.get("safety_factor", 0.9)),
                min_dt_factor=float(adaptive.get("min_dt_factor", 0.1)),
                max_dt_factor=float(adaptive.get("max_dt_factor", 5.0)),
                max_rejections=int(adaptive.get("max_rejections", 10)), dt_min=adaptive.get("dt_min"))
            if self.adaptive.controller.dt_min is None:
                span = abs(self.t_end - self.t0)
                self.adaptive.controller.dt_min = 1e-12 * span if span > 0.0 else 1e-14
        elif self.scheme not in ("rk2", "rk4"):
            raise ValueError(f"Integration scheme {self.scheme!r} needs adaptive stepping; choose rk2 or rk4")
        with self.timers("kernel_warm_up"):
            self.stepper.warm_up(embedded=self.adaptive is not None)

    def _setup_incompressible(self, config, triton) -> None:
        phys = config.get("physics") or {}
        mhd_block = phys.get("mhd") or {}
        nu = float(phys.get("viscosity", mhd_block.get("viscosity", 0.0)))
        if "resistivity" in phys or "resistivity" in mhd_block:
            eta = float(phys.get("resistivity", mhd_block.get("resistivity", 0.0)))
        elif "magnetic_prandtl" in phys:
            eta = nu / float(phys["magnetic_prandtl"])
        else:
            eta = 0.0
        if self.scheme not in LAWSON_TABLES:
            raise ValueError(f"Incompressible scheme must be one of {sorted(LAWSON_TABLES)}, got {self.scheme!r}")
        if _enabled((config.get("integration") or {}).get("adaptive")) or _enabled(config.get("adaptive")):
            raise NotImplementedError("Adaptive (embedded) stepping is not available for the incompressible solver")
        self.adaptive = None
        self.nu, self.eta = nu, eta
        self.stepper = IncompressibleSlabStepper(
            self.grid, nu, eta, mhd=(self.system == "imhd"), scheme=self.scheme, cfl=self.cfl,
            triton=triton, spherical_truncation=bool(phys.get("spherical_truncation", False)),
            forcing=self.forcing)
        with self.timers("kernel_warm_up"):
            self.stepper.warm_up()

    def _initialize(self) -> None:
        icfg = self.config.get("initial_conditions") or {}
        with self.timers("initial_conditions"):
            if self.compressible:
                self.stepper.seed(self.problem.build(self.grid, icfg, self.gamma))
            else:
                kind, data = self.problem.build(self.grid, icfg, 0.0)
                if kind == "physical":
                    self.stepper.seed(data)
                else:
                    self.stepper.set_spectrum(data)
                self.stepper.t = self.t0

    def _restart(self, path: str) -> None:
        meta = self.comm.bcast(read_meta(path) if self.comm.is_root else None)
        spectrum = read_spectrum(path, self.grid, self.system, meta)
        self.stepper.set_spectrum(spectrum)
        self.t = float(meta["t"])
        self.step_count = int(meta.get("step", 0))
        self.out_index = int(meta.get("output_index", 0))
        self.chk_index = int(meta.get("checkpoint_index", 0))
        if not self.compressible:
            self.stepper.t = self.t
            self.stepper.steps = self.step_count
        if self.forcing is not None:
            state = meta.get("forcing")
            if state is None:
                self.log("[phrike.parallel] restart file has no forcing state; the forcing starts afresh")
            else:
                self.forcing.load_state_dict(state)
        changed = (int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])) != (self.grid.Nx, self.grid.Ny, self.grid.Nz)
        self.log(f"[phrike.parallel] restarted from {path} at t={self.t:.6g}, step {self.step_count}"
                 + (f" (resolution changed from {meta['Nx']}x{meta['Ny']}x{meta['Nz']})" if changed else ""))

    # --- output -------------------------------------------------------------

    def _extra_meta(self) -> Dict[str, Any]:
        extra = {"problem": self.problem.name, "output_index": self.out_index,
                 "checkpoint_index": self.chk_index, "config": self.config,
                 "physics": {"viscosity": self.nu, "resistivity": self.eta}}
        if self.compressible:
            extra["physics"]["gamma"] = self.gamma
        if self.forcing is not None:
            extra["forcing"] = self.forcing.state_dict()
        return extra

    def _diagnostics(self) -> Dict[str, float]:
        with self.timers("diagnostics"):
            diag = self.stepper.diagnostics()
        diag["time"] = self.t
        diag["step"] = float(self.step_count)
        return diag

    def _output(self) -> None:
        """Snapshot, spectra, slice and a history row at an output time."""
        with self.timers("output"):
            index = self.out_index
            self.out_index += 1
            self._last_output_time = self.t
            if self.write_snapshots:
                write_snapshot(os.path.join(self.outdir, "snapshots", f"snap_{index:06d}"), self.grid,
                               self.system, self.stepper.uhat, self.t, self.step_count, self._extra_meta())
            if self.write_spectra:
                spectra = self.stepper.spectra()
                if self.comm.is_root:
                    os.makedirs(os.path.join(self.outdir, "spectra"), exist_ok=True)
                    np.savez(os.path.join(self.outdir, "spectra", f"spectra_{index:06d}.npz"),
                             t=self.t, step=self.step_count, **spectra)
            if self.write_slices:
                write_slice(os.path.join(self.outdir, "slices", f"slice_{index:06d}.npz"), self.grid,
                            self.stepper.physical(), field_names(self.system), self.t)
            diag = self._diagnostics()
            if self.compressible and self.system == "mhd":
                diag["div_b_max"] = self.stepper.max_div_b()
            if self.comm.is_root:
                self.history.write(diag)
            self.log(f"[output {index}] t={self.t:.6g} step={self.step_count}")

    def _checkpoint(self) -> None:
        with self.timers("output"):
            index = self.chk_index
            self.chk_index += 1
            root = os.path.join(self.outdir, "checkpoints")
            write_snapshot(os.path.join(root, f"chk_{index:06d}"), self.grid, self.system,
                           self.stepper.uhat, self.t, self.step_count, self._extra_meta())
            old = index - self.keep_checkpoints
            if self.comm.is_root and self.keep_checkpoints > 0 and old >= 0:
                shutil.rmtree(os.path.join(root, f"chk_{old:06d}"), ignore_errors=True)

    def _monitor(self, dt: float, wall: float) -> None:
        diag = self._diagnostics()
        diag["dt"] = dt
        if self.comm.is_root:
            self.history.write(diag)
        parts = [f"step {self.step_count}", f"t={self.t:.5g}", f"dt={dt:.3e}"]
        for key, label in (("kinetic_energy", "Ek"), ("magnetic_energy", "Em"), ("alfven_mach", "MA"),
                           ("sonic_mach", "Ms"), ("viscous_dissipation", "eps_nu"),
                           ("resistive_dissipation", "eps_eta"), ("forcing_power", "P"),
                           ("kmax_eta", "kmax*eta_K"), ("energy", "E")):
            if key in diag:
                parts.append(f"{label}={diag[key]:.4g}")
        if wall > 0.0:
            rate = self.grid.n_points / self.comm.size / wall / 1e6
            parts.append(f"{wall:.3f} s/step, {rate:.2f} Mzc/s/rank")
        self.log("  ".join(parts))

    # --- time stepping ------------------------------------------------------

    def _next_event(self, interval: float) -> float:
        """First time on the grid ``t0 + n * interval`` after the current time."""
        if not interval > 0.0:
            return float("inf")
        done = math.floor((self.t - self.t0) / interval + 1e-9)
        return self.t0 + (done + 1) * interval

    def _horizon(self, next_output: float, next_checkpoint: float) -> float:
        horizon = self.t_end - self.t
        for event in (next_output, next_checkpoint):
            if math.isfinite(event) and event > self.t:
                horizon = min(horizon, event - self.t)
        return min(horizon, self.dt_max)

    def _compressible_limit(self) -> float:
        speed = self.stepper.speed
        bounds = []
        if speed is None or not math.isfinite(speed):
            raise FloatingPointError(
                f"Non-finite maximum wave speed ({speed}) at step {self.step_count}, t={self.t}")
        if speed > 0.0:
            bounds.append(self.cfl * self.grid.min_spacing / speed)
        rate = max(self.eta, self.nu) * self.grid.k2_max
        if rate > 0.0:
            bounds.append(self.cfl_diff / rate)
        return min(bounds) if bounds else float("inf")

    def _compressible_kick(self, dt: float) -> None:
        f = self.forcing
        if f is None:
            return
        with self.timers("forcing"):
            if f.kind == "ou":
                dv = f.acceleration() * dt
                f.advance(dt)
            else:
                dv = f.acceleration(f._sample() * math.sqrt(dt))
            self.stepper.kick(dv)

    def _advance_compressible(self, proposal: float, horizon: float):
        """One fixed step, or one adaptive attempt.  Returns (dt taken or 0, next proposal)."""
        limit = self._compressible_limit()
        choices = [v for v in (proposal, limit, horizon) if math.isfinite(v)]
        if self.dt_fixed is not None:
            choices = [min(float(self.dt_fixed), horizon)]
        if not choices or min(choices) <= 0.0:
            raise RuntimeError(f"No finite positive time step at step {self.step_count}, t={self.t}")
        dt = min(choices)
        free = [v for v in (proposal, limit) if math.isfinite(v)]
        self._check_stall(min(free) if free else float("inf"))
        st = self.stepper
        if self.adaptive is None:
            st.step(dt, self.scheme)
            self._compressible_kick(dt)
            return dt, float("inf")
        method = self.adaptive.method
        error, scale = st.attempt(dt, method.a, method.b_high, method.b_low)
        result = self.adaptive.step(None, None, dt, scale, embedded=lambda U, h: (None, None, error))
        if result.accepted:
            st.commit()
            self._compressible_kick(dt)
            return dt, result.dt_next
        st.reject()
        return 0.0, result.dt_next

    def _check_stall(self, free_dt: float) -> None:
        reference = getattr(self, "_dt_reference", None)
        if reference is None:
            if math.isfinite(free_dt) and free_dt > 0.0:
                self._dt_reference = free_dt
            return
        if math.isfinite(free_dt) and free_dt < 1e-8 * reference:
            raise RuntimeError(
                f"Time step collapsed to {free_dt:.3e} ({free_dt / reference:.1e} of the initial "
                f"{reference:.3e}) at step {self.step_count}, t={self.t}; the solution has likely "
                "lost resolution.")

    def run(self) -> Dict[str, Any]:
        comm, st = self.comm, self.stepper
        self.log(f"[phrike.parallel] {self.problem.name} ({self.system}): {self.grid.describe()}")
        self.log(f"[phrike.parallel] {comm.describe()}")
        if self.forcing is not None:
            self.log(f"[phrike.parallel] {self.forcing.describe()}")
        tier = "Triton kernels" if getattr(st, "_triton", None) is not None else "torch ops"
        self.log(f"[phrike.parallel] stages: {tier}; scheme {self.scheme}"
                 + (f" adaptive {self.adaptive.method.name}" if self.adaptive is not None else "")
                 + f"; output to {self.outdir}")
        if comm.is_root:
            with open(os.path.join(self.outdir, "run_info.json"), "w", encoding="utf-8") as handle:
                json.dump({"problem": self.problem.name, "system": self.system, "ranks": comm.size,
                           "grid": [self.grid.Nx, self.grid.Ny, self.grid.Nz],
                           "precision": self.grid.precision, "device": str(self.grid.device),
                           "mpi": comm.describe(), "config": self.config}, handle, indent=1, default=str)

        next_output = self._next_event(self.output_interval)
        next_checkpoint = self._next_event(self.checkpoint_interval)
        if not self.restarted:
            self._output()  # a restart starts from a state that is already on disk
        proposal = float("inf")
        self._step_fields = 0
        self._step_bytes = 0
        start = time.perf_counter()
        last_wall, last_step = start, self.step_count
        first_steps = self.step_count
        last_dt = float("nan")
        stopped_early = False
        while self.t < self.t_end - 1e-12:
            if self.max_steps is not None and self.step_count - first_steps >= int(self.max_steps):
                stopped_early = True
                break
            if self.max_wall is not None and (self.step_count - first_steps) % 10 == 0:
                # Rank 0's clock decides for everyone.
                if comm.bcast(time.perf_counter() - start > self.max_wall if comm.is_root else None):
                    self.log(f"[phrike.parallel] wall-clock limit reached at step {self.step_count}")
                    stopped_early = True
                    break
            horizon = self._horizon(next_output, next_checkpoint)
            fft = self.grid.fft
            fields_before, bytes_before = fft.forward_fields + fft.inverse_fields, comm.bytes_sent
            with self.timers("step"):
                if self.compressible:
                    dt, proposal = self._advance_compressible(proposal, horizon)
                    self.t += dt
                else:
                    fixed = None if self.dt_fixed is None else min(float(self.dt_fixed), horizon)
                    dt = st.step(dt_max=horizon, dt=fixed)
                    self.t = st.t
            self._step_fields += fft.forward_fields + fft.inverse_fields - fields_before
            self._step_bytes += comm.bytes_sent - bytes_before
            if dt <= 0.0:
                continue  # a rejected adaptive attempt
            self.step_count += 1
            last_dt = dt
            if self.monitor_enabled and self.step_count % self.monitor_interval == 0:
                now = time.perf_counter()
                self._monitor(dt, (now - last_wall) / max(self.step_count - last_step, 1))
                last_wall, last_step = time.perf_counter(), self.step_count
            if self.t + 1e-12 >= next_output:
                self._output()
                next_output += self.output_interval
            if self.t + 1e-12 >= next_checkpoint:
                self._checkpoint()
                next_checkpoint += self.checkpoint_interval
        wall = time.perf_counter() - start
        steps = self.step_count - first_steps
        if stopped_early:
            # Not an output time: leave the output series alone and save a restart point.
            self._checkpoint()
            self.log(f"[phrike.parallel] stopped at t={self.t:.6g}; restart from "
                     f"{os.path.join(self.outdir, 'checkpoints', f'chk_{self.chk_index - 1:06d}')}")
        elif self._last_output_time is None or abs(self.t - self._last_output_time) > 1e-12:
            self._output()  # the end state, unless the last step already landed on an output
        summary = self._summary(wall, steps, last_dt)
        return summary

    def _summary(self, wall: float, steps: int, last_dt: float) -> Dict[str, Any]:
        comm = self.comm
        names = sorted(self.timers.total)
        totals = np.asarray([self.timers.total[n] for n in names])
        mean = comm.allreduce(totals, "sum") / comm.size if names else totals
        worst = comm.allreduce(totals, "max") if names else totals
        per_rank = self.grid.n_points / comm.size
        step_seconds = self.timers.total.get("step", 0.0)
        step_seconds = comm.allreduce([step_seconds], "max")[0]
        summary = {
            "problem": self.problem.name, "system": self.system, "ranks": comm.size,
            "grid": [self.grid.Nx, self.grid.Ny, self.grid.Nz], "t": self.t, "steps": steps,
            "last_dt": last_dt, "wall_seconds": wall,
            # Stepping only: output, diagnostics and setup are timed separately.
            "seconds_per_step": step_seconds / steps if steps else float("nan"),
            "zone_cycles_per_second_per_rank": (per_rank * steps / step_seconds
                                                if step_seconds > 0 and steps else float("nan")),
            "fft_fields_per_step": self._step_fields / steps if steps else float("nan"),
            "megabytes_sent_per_step_per_rank": (comm.allreduce([self._step_bytes], "sum")[0] / comm.size
                                                 / steps / 1e6 if steps else float("nan")),
            "timers_mean_seconds": {n: float(v) for n, v in zip(names, mean)},
            "timers_max_seconds": {n: float(v) for n, v in zip(names, worst)},
            "outdir": self.outdir,
        }
        if self.compressible:
            summary["positivity_events"] = self.stepper.events
        if comm.is_root:
            with open(os.path.join(self.outdir, "timing.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=1, default=str)
        self.log(f"[phrike.parallel] done: t={self.t:.6g}, {steps} steps in {wall:.2f} s "
                 f"({summary['seconds_per_step']:.4g} s/step, "
                 f"{summary['zone_cycles_per_second_per_rank'] / 1e6:.3g} Mzc/s/rank)")
        return summary


def run(problem: str, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None,
        **kwargs) -> Dict[str, Any]:
    """Run a distributed problem; see ``Simulation`` for the keyword arguments."""
    if config is None:
        if config_path is None:
            raise ValueError("Either config or config_path must be provided")
        from ..io import load_config

        config = load_config(config_path)
    sim = Simulation(problem, config, **kwargs)
    summary = sim.run()
    summary["simulation"] = sim
    return summary
