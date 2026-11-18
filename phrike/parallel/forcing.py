"""Large-scale forcing on a shell of low wavenumbers, identical for any rank count.

The forced band holds a few hundred modes at most, so every rank carries the
whole band: the coefficients are generated from a counter-based stream keyed
by ``(seed, update index)`` and evolve identically everywhere, with no
communication.  A rank then applies the modes it owns (spectral solvers) or
places them on the retained modes of its ky slab and takes one distributed
inverse transform of the three components (physical-space kick of the
compressible solvers), so the work per step is proportional to the data the
rank holds.

Types
-----
``ou``
    Ornstein-Uhlenbeck acceleration with correlation time ``correlation_time``
    and rms ``amplitude``; frozen during a step and advanced exactly between
    steps.
``white``
    Delta-correlated forcing applied as one kick per step.  ``power`` is the
    mean energy injection rate per unit mass (the ``|du|^2 / 2`` part, which is
    exact for every realization; the ``u . du`` part has zero mean).
``constant_power``
    ``f_k = power * u_k / (2 E_band)`` on the band: energy enters at exactly
    ``power``.  Needs the band of the velocity spectrum, so it is available to
    the incompressible solver only.

``kmin`` and ``kmax`` are mode magnitudes (cycles per box), as in the
turbulence initial conditions.  ``profile`` shapes the band: ``parabolic``
(zero at both edges), ``flat`` (equal variance per mode) or ``power_law``,
whose shell-integrated spectrum of the forcing follows ``k^spectral_index``
(default -5/3, a Kolmogorov band), i.e. a variance per mode of
``k^(spectral_index - 2)``.  With a spherical truncation (anisotropic
boxes) only the modes of the band the solver keeps are forced.
``compressive_fraction`` mixes the compressive
and solenoidal projections of the random vector (0 is purely solenoidal).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .grid import SlabGrid3D

_TYPES = ("ou", "white", "constant_power")


def spherical_kcut2(grid: SlabGrid3D) -> float:
    """Squared radius of the spherical truncation: the smallest per-axis retained wavenumber, squared.

    The one definition shared by the incompressible solver (which truncates its
    state there) and the forcing (which must not drive modes beyond it).
    """
    d = grid.decomp
    kcut = min(2.0 * np.pi * d.cx / grid.Lx, 2.0 * np.pi * d.cy / grid.Ly, 2.0 * np.pi * d.cz / grid.Lz)
    return float(kcut * kcut) * (1.0 + 1e-12)


class BandForcing:
    """Forcing of the velocity on ``kmin <= |n| <= kmax``."""

    def __init__(self, grid: SlabGrid3D, config: Dict[str, Any], incompressible: bool,
                 spherical_truncation: bool = False):
        self.grid = grid
        self.kcut2 = spherical_kcut2(grid) if spherical_truncation else float("inf")
        self.kind = str(config.get("type", "ou")).lower()
        if self.kind not in _TYPES:
            raise ValueError(f"forcing.type must be one of {_TYPES}, got {self.kind!r}")
        if self.kind == "constant_power" and not incompressible:
            raise ValueError("forcing.type 'constant_power' needs the incompressible solver")
        self.kmin = float(config.get("kmin", 1.0))
        self.kmax = float(config.get("kmax", 2.5))
        self.seed = int(config.get("seed", 0))
        self.tau = float(config.get("correlation_time", 1.0))
        self.amplitude = float(config.get("amplitude", 1.0))
        self.power = float(config.get("power", 0.0))
        zeta = 0.0 if incompressible else float(config.get("compressive_fraction", 0.0))
        if not 0.0 <= zeta <= 1.0:
            raise ValueError("forcing.compressive_fraction must lie in [0, 1]")
        self.zeta = zeta
        profile = str(config.get("profile", "parabolic")).lower()
        if profile == "kolmogorov":
            profile = "power_law"
        if profile not in ("parabolic", "flat", "power_law"):
            raise ValueError("forcing.profile must be 'parabolic', 'flat' or 'power_law'")
        self.profile = profile
        self.spectral_index = float(config.get("spectral_index", -5.0 / 3.0))
        if self.kind == "ou" and not (np.isfinite(self.tau) and self.tau > 0.0):
            raise ValueError("forcing.correlation_time must be positive")
        if self.kind in ("white", "constant_power") and self.power <= 0.0:
            raise ValueError(f"forcing.type '{self.kind}' needs a positive forcing.power")

        d = grid.decomp
        m = int(np.floor(self.kmax))
        if m < 1 or m > min(d.cx, d.cy, d.cz):
            raise ValueError("forcing.kmax must be at least 1 and inside the retained modes")
        # Independent modes: nx > 0, or the nx = 0 half plane with (ny > 0) or (ny == 0, nz > 0).
        nz, ny, nx = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), np.arange(0, m + 1),
                                 indexing="ij")
        mag = np.sqrt(nz * nz + ny * ny + nx * nx)
        band = (mag >= self.kmin) & (mag <= self.kmax)
        canonical = (nx > 0) | (ny > 0) | ((ny == 0) & (nz > 0))
        # Modes beyond the solver's spherical truncation would be driven but never coupled.
        k2_box = ((2.0 * np.pi * nx / grid.Lx) ** 2 + (2.0 * np.pi * ny / grid.Ly) ** 2
                  + (2.0 * np.pi * nz / grid.Lz) ** 2)
        inside = k2_box <= self.kcut2
        keep = band & canonical & inside
        self.n_truncated = int((band & canonical & ~inside).sum())
        self.n = np.stack([nz[keep], ny[keep], nx[keep]], axis=1).astype(np.int64)  # (M, 3)
        self.M = int(self.n.shape[0])
        if self.M == 0:
            if self.n_truncated:
                raise ValueError(
                    f"the forcing band {self.kmin:g} <= |n| <= {self.kmax:g} contains no modes inside the "
                    f"spherical truncation |k| <= {np.sqrt(self.kcut2):.6g} ({self.n_truncated} lie outside)")
            raise ValueError("the forcing band contains no modes")
        mag = mag[keep]
        self.k = np.stack([2.0 * np.pi * self.n[:, 2] / grid.Lx, 2.0 * np.pi * self.n[:, 1] / grid.Ly,
                           2.0 * np.pi * self.n[:, 0] / grid.Lz], axis=1)  # (M, 3) as (kx, ky, kz)
        self.k2 = (self.k * self.k).sum(axis=1)
        if profile == "parabolic" and self.kmax > self.kmin:
            envelope = np.sqrt(np.clip((mag - self.kmin) * (self.kmax - mag), 0.0, None))
            if not np.any(envelope > 0.0):
                envelope = np.ones_like(mag)
        elif profile == "power_law":
            # Modes per unit |n| grow as |n|^2, so E(k) ~ k^s needs |f_k|^2 ~ k^(s - 2).
            envelope = mag ** (0.5 * (self.spectral_index - 2.0))
        else:
            envelope = np.ones_like(mag)
        # A unit complex Gaussian vector keeps variance 2 in the solenoidal
        # plane and 1 along k, so E|sample|^2 = env^2 ((1 - zeta)^2 2 + zeta^2).
        per_mode = envelope ** 2 * ((1.0 - zeta) ** 2 * 2.0 + zeta ** 2)
        total = float(per_mode.sum())
        # rms^2 of the field = 2 sum_M |f|^2 (each mode has a conjugate partner);
        # white-noise injection = sum_M |du|^2 / dt.
        target = 0.5 * self.amplitude ** 2 if self.kind == "ou" else self.power
        self._envelope = envelope * np.sqrt(target / total)
        self.f = np.zeros((self.M, 3), dtype=np.complex128)  # components (x, y, z)
        self.updates = 0  # index of the next random draw
        self.injected = 0.0  # this rank's part of the energy added by kicks
        if self.kind == "ou":
            self.f = self._sample()

        # Stored modes owned by this rank: every independent mode, plus the
        # conjugate partners on the kx = 0 plane.
        rows = {int(v): j for j, v in enumerate(d.modes_y_local())}
        idx, src, conj, weight = [], [], [], []
        for j in range(self.M):
            mz, my, mx = (int(v) for v in self.n[j])
            for sign in ((1, -1) if mx == 0 else (1,)):
                row = rows.get(sign * my)
                if row is None:
                    continue
                iz = sign * mz if sign * mz >= 0 else sign * mz + d.Kz
                idx.append((iz * d.nky + row) * d.Kx + mx)
                src.append(j)
                conj.append(sign < 0)
                weight.append(1.0 if mx == 0 else 2.0)
        self.n_local = len(idx)
        self._src = np.asarray(src, dtype=np.int64)
        self._conj = np.asarray(conj, dtype=bool)
        dev = grid.device
        self.index = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=dev)
        self.local_k2 = torch.as_tensor(self.k2[self._src] if self.n_local else np.zeros(0),
                                        dtype=grid.rdtype, device=dev)
        self.local_weight = torch.as_tensor(np.asarray(weight), dtype=grid.rdtype, device=dev)

    # --- random band --------------------------------------------------------

    def _sample(self) -> np.ndarray:
        """One draw of the band, (M, 3) complex, with the configured spectrum and mix."""
        rng = np.random.Generator(np.random.Philox(key=np.array([self.seed, self.updates],
                                                                dtype=np.uint64)))
        self.updates += 1
        xi = (rng.standard_normal((self.M, 3)) + 1j * rng.standard_normal((self.M, 3))) / np.sqrt(2.0)
        along = self.k * ((xi * self.k).sum(axis=1) / self.k2)[:, None]
        mixed = (1.0 - self.zeta) * (xi - along) + self.zeta * along
        return mixed * self._envelope[:, None]

    def _to_local(self, band: np.ndarray) -> torch.Tensor:
        """Band values (M, 3) -> this rank's stored modes, (3, n_local) on the device."""
        local = band[self._src]
        local = np.where(self._conj[:, None], np.conj(local), local)
        return torch.as_tensor(np.ascontiguousarray(local.T), dtype=self.grid.cdtype,
                               device=self.grid.device)

    # --- spectral solvers ---------------------------------------------------

    def frozen(self, uhat: torch.Tensor) -> Optional[torch.Tensor]:
        """Acceleration on this rank's band modes for the coming step, (3, n_local)."""
        if self.kind == "white":
            return None
        if self.kind == "constant_power":
            band = self._gather_band(uhat)
            energy = float((np.abs(band) ** 2).sum())  # = E_band, both halves counted
            if not energy > 0.0:
                return None
            self.f = band * (self.power / (2.0 * energy))
        return self._to_local(self.f) if self.n_local else None

    def kick(self, dt: float) -> Optional[torch.Tensor]:
        """Velocity increment of a white-noise step on this rank's band modes."""
        if self.kind != "white":
            return None
        increment = self._sample() * np.sqrt(float(dt))
        return self._to_local(increment) if self.n_local else None

    def advance(self, dt: float) -> None:
        """Advance the Ornstein-Uhlenbeck process by ``dt`` (exact update)."""
        if self.kind != "ou":
            return
        decay = float(np.exp(-float(dt) / self.tau))
        self.f = decay * self.f + np.sqrt(max(1.0 - decay * decay, 0.0)) * self._sample()

    def _gather_band(self, uhat: torch.Tensor) -> np.ndarray:
        """Velocity spectrum on the independent band modes, (M, 3), on every rank."""
        local = np.zeros((self.M, 3, 2))
        if self.n_local:
            own = ~self._conj
            values = uhat.reshape(uhat.shape[0], -1)[:3][:, self.index[torch.as_tensor(
                own, device=self.grid.device)]]
            values = torch.view_as_real(values).to(torch.float64).cpu().numpy()  # (3, n, 2)
            local[self._src[own]] = np.transpose(values, (1, 0, 2))
        total = self.grid.comm.allreduce(local.reshape(-1), "sum").reshape(self.M, 3, 2)
        return total[..., 0] + 1j * total[..., 1]

    def local_power(self, uhat: torch.Tensor, values: Optional[torch.Tensor]) -> float:
        """This rank's part of sum_k Re(conj(u_k) f_k) over the full spectrum."""
        if values is None or not self.n_local:
            return 0.0
        u = uhat.reshape(uhat.shape[0], -1)[:3][:, self.index]
        return float(((u.conj() * values).real.sum(dim=0) * self.local_weight).sum().item())

    # --- physical-space synthesis (compressible solvers) --------------------

    def acceleration(self, band: Optional[np.ndarray] = None) -> torch.Tensor:
        """Real field (3, nz, Ny, Nx) on this rank's slab from band values (default: ``f``).

        The band goes onto the retained modes this rank owns (a small scatter,
        the same placement as the spectral solvers) and one distributed inverse
        transform of the three components returns the physical field; no
        rank synthesizes more than its own slab.
        """
        g = self.grid
        S = g.zeros_spectral(3)
        if self.n_local:
            S.reshape(3, -1)[:, self.index] = self._to_local(self.f if band is None else band)
        return g.inverse(S)

    # --- restart ------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {"f_real": self.f.real.tolist(), "f_imag": self.f.imag.tolist(),
                "updates": int(self.updates), "kind": self.kind, "modes": int(self.M)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("kind") != self.kind or int(state.get("modes", -1)) != self.M:
            raise ValueError("the restart file's forcing does not match forcing.* in the configuration")
        self.f = np.asarray(state["f_real"]) + 1j * np.asarray(state["f_imag"])
        self.updates = int(state["updates"])

    def describe(self) -> str:
        detail = {"ou": f"rms acceleration {self.amplitude:g}, correlation time {self.tau:g}",
                  "white": f"power {self.power:g}",
                  "constant_power": f"power {self.power:g}"}[self.kind]
        shape = f"k^{self.spectral_index:.4g} spectrum" if self.profile == "power_law" else self.profile
        return (f"{self.kind} forcing on {self.kmin:g} <= |n| <= {self.kmax:g}, {shape} "
                f"({self.M} independent modes"
                + (f", {self.n_truncated} beyond the truncation dropped" if self.n_truncated else "")
                + f"), {detail}")
