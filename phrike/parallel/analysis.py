"""Spectral turbulence diagnostics of the distributed solvers.

Two collective entry points, called by every rank at the same point with the
same ``config`` (the ``io.diagnostics`` block; it may be empty) and returning
the same dict on every rank, numpy arrays plus an optional ``"attrs"`` dict of
Python scalars and strings::

    spectral_diagnostics(stepper, config)   # every output
    transfer_diagnostics(stepper, config)   # every few outputs; incompressible only

All sums run over the local retained modes and are reduced with one
``comm.allreduce`` of a small float64 array.  Binning is ``scatter_add_`` (no host synchronisation) on
the device, in chunks of whole kz planes (``chunk_modes`` modes at most, default
2**23), so no per-mode host loop and no field-sized host copy exists.

Conventions
-----------
``uhat`` is the retained spectrum divided by N, so a volume mean is the sum
over the half spectrum with the Parseval weights ``w`` (2 for kx > 0):
``<f g> = sum_k w Re(f_k g_k^*)``.  Shells are those of ``stepper.spectra()``:
unit width in ``|k| L_max / (2 pi)``, centred on integers (shell ``n`` holds
``n - 1/2 <= |k|/dk < n + 1/2``), ``k = arange(nbins) * dk``.  ``b`` is in
Alfven units.  Every spectrum is a shell *sum*: summed over shells it gives
the volume mean quoted below.

Spectra (incompressible, keys of ``spectral_diagnostics``; ``(nbins,)``)
    ``kinetic``/``magnetic``  ``w |u_k|^2 / 2``, sum ``<|u|^2>/2``.
    ``kinetic_helicity``      ``w Re(u_k . w_k^*)``, ``w_k = i k x u_k``, sum ``<u . curl u>``.
    ``cross_helicity``        ``w Re(u_k . b_k^*)``, sum ``<u . b>``.
    ``magnetic_helicity``     ``w Re(a_k . b_k^*)``, ``a_k = i k x b_k / k^2`` (Coulomb gauge,
                              ``k = 0`` omitted), sum ``<a . b>``.
    ``current_helicity``      ``w Re(j_k . b_k^*)``, ``j_k = i k x b_k`` (only with ``"current_helicity": true``).
    ``dissipation_kinetic``   ``w D_u(k^2) |u_k|^2``, with ``D_u = nu k^2 + nu_h k^(2p)`` (the stepper's
                              ``damping``); its sum is ``viscous + hyperviscous_dissipation`` of
                              ``stepper.diagnostics()``.  ``dissipation_magnetic``: the same with
                              ``eta``, ``eta_h`` for ``b`` (sum ``resistive + hyperresistive_dissipation``).
    Without a magnetic field only the kinetic keys appear.

Anisotropy about the axis ``e`` (``config["anisotropy"]``; ``false`` or
``{"enabled": false}`` skips it)
    ``"axis"``: ``"auto"`` (default; the direction of the mean field, the ``k = 0`` mode of ``b``,
    when ``|<b>| > 1e-8 b_rms``, else z), ``"x"``/``"y"``/``"z"`` or a 3-vector.
    ``k_par = |k . e|``, ``k_perp = |k - (k . e) e|``, unit bins of ``2 pi / L_max`` centred on integers.
    ``k_par`` ``(npar,)``, ``k_perp`` ``(nperp,)``; ``kinetic_2d``, ``magnetic_2d`` ``(npar, nperp)``
    (energy ``w |.|^2 / 2``), ``kinetic_par``/``kinetic_perp`` and ``magnetic_par``/``magnetic_perp``
    (the 2D spectra summed over the other index; all sum to the energy).  0-d arrays
    ``u_component_anisotropy = <(u.e)^2> / <|u_perp|^2>``, ``u_gradient_anisotropy =
    <|(e.grad) u|^2> / <|grad_perp u|^2>`` and the same for ``b``.  ``attrs`` carry the axis used
    (``axis`` as text, ``axis_x``, ``axis_y``, ``axis_z``).

Compressible (``euler``/``mhd``): ``kinetic`` (``u = m / rho``, sum ``<|u|^2>/2``),
``kinetic_density_weighted`` (``sqrt(rho) u``, Kida & Orszag; sum ``<rho |u|^2>/2``; both are the
means of the 2/3-truncated fields, as ``m / rho`` itself is not band limited),
``density`` (sum ``<rho^2>/2``, the mean sits in shell 0), ``magnetic`` (``B``, mhd only) and the same
anisotropy outputs for ``u`` and ``B``.  ``u`` and ``sqrt(rho) u`` are formed on the slab and
transformed one component at a time; ``rho`` and ``B`` are read from the retained state spectrum.

Transfer (``transfer_diagnostics``; unit ``<.>`` volume means)
    Solver equations ``du/dt = -(u.grad)u + (b.grad)b - grad P``, ``db/dt = (b.grad)u - (u.grad)b``;
    ``X_K`` is ``X`` filtered to shell ``K``::

        T_uu(K,Q) = -<u_K . (u.grad) u_Q>     T_bb(K,Q) = -<b_K . (u.grad) b_Q>
        T_ub(K,Q) =  <u_K . (b.grad) b_Q>     T_bu(K,Q) =  <b_K . (b.grad) u_Q>

    ``T_ub`` is the Lorentz-force transfer from magnetic shell ``Q`` to kinetic shell ``K``, ``T_bu`` the
    stretching of ``b_Q`` by ``u`` into magnetic shell ``K``.  Products are the exact ``d_j(a_j X_i)``
    (``div`` of ``u`` and ``b`` vanishes), formed on the solver grid (2/3-truncated transform, as the
    solver) and contracted in spectral space.  Hydrodynamic runs have ``T_uu`` only.

    Always: ``k`` and, per channel ``X``, ``T_X`` ``(nbins,)`` (``T_X(K) = sum_Q T_X(K,Q)``, unfiltered
    field in the Q slot) and the flux ``Pi_X = -cumsum(T_X)``: keys ``T_uu``, ``Pi_uu`` and, with a
    magnetic field, ``T_bb, T_ub, T_bu, Pi_bb, Pi_ub, Pi_bu``.  Cost: 9 inverse and 21 forward
    transforms (3 and 6 without ``b``), with at most about 6 physical fields alive.

    ``config["transfer"]["matrix"]: true`` adds ``shell_edges`` ``(ns+1,)`` and the matrices
    ``T_uu_matrix`` (and ``T_bb_matrix``, ``T_ub_matrix``, ``T_bu_matrix``), ``(ns, ns)`` with rows ``K``
    and columns ``Q``, on shells ``edges[n] <= |k|/dk < edges[n+1]`` (default edges: ``0, 1, 2, 4, ...``
    up to above the largest retained ``|k|``; ``config["transfer"]["shells"]`` gives others, modes
    outside them are dropped).  Cost per non-empty ``Q`` shell: 36 forward transforms (9 without ``b``)
    and 24 inverse (12) in the default memory-lean mode (the products' multipliers are transformed
    again for each use; about 6 physical fields alive), or 6 inverse (3) with
    ``"cache_fields": true`` (``u`` and ``b`` stay in physical space, about 10 fields).
    Other transfer options: ``batch`` (products per forward transform, default 2, never above the grid's
    FFT batch).
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .compressible import CompressibleSlabStepper
from .grid import SlabGrid3D
from .incompressible import IncompressibleSlabStepper

_F64 = torch.float64
_CHUNK_MODES = 1 << 23
_TRANSFER_BATCH = 2
_AXIS_TOLERANCE = 1e-8


# --- configuration -------------------------------------------------------------

def _section(config: Optional[dict], name: str) -> dict:
    """A sub-block of ``config``; a bare boolean becomes ``{"enabled": value}``."""
    value = (config or {}).get(name, {})
    if isinstance(value, (bool, np.bool_)):
        return {"enabled": bool(value)}
    return dict(value) if value else {}


def _chunk_modes(config: Optional[dict]) -> int:
    return max(1, int((config or {}).get("chunk_modes", _CHUNK_MODES)))


def _attrs(stepper, **extra) -> dict:
    attrs = {"shells": "unit width in |k| L_max/(2 pi), centred on integers (as stepper.spectra())"}
    t = getattr(stepper, "t", None)
    if t is not None:
        attrs["t"] = float(t)
    attrs.update(extra)
    return attrs


# --- geometry ------------------------------------------------------------------

def _z_chunks(grid: SlabGrid3D, modes: int):
    """Slices of this rank's kz planes holding at most ``modes`` modes (at least one plane)."""
    d = grid.decomp
    step = max(1, int(modes) // max(1, d.nky * d.Kx))
    for z0 in range(0, d.Kz, step):
        yield slice(z0, min(z0 + step, d.Kz))


def _k64(grid: SlabGrid3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wavenumbers in float64, shaped to broadcast against ``(Kz, nky, Kx)``."""
    def to(a, shape):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), device=grid.device).reshape(shape)

    d = grid.decomp
    return (to(2.0 * np.pi * grid.nx_modes / grid.Lx, (1, 1, d.Kx)),
            to(2.0 * np.pi * grid.ny_modes / grid.Ly, (1, d.nky, 1)),
            to(2.0 * np.pi * grid.nz_modes / grid.Lz, (d.Kz, 1, 1)))


def _plane(k: torch.Tensor, sl: slice) -> torch.Tensor:
    """The kz planes of a broadcastable wavenumber array (x and y ones are shared by all planes)."""
    return k if k.shape[0] == 1 else k[sl]


def _unit_geometry(grid: SlabGrid3D) -> Tuple[float, int]:
    """Shell width and count, as ``stepper.spectra()``."""
    dk = 2.0 * np.pi / max(grid.Lx, grid.Ly, grid.Lz)
    return dk, int(math.floor(math.sqrt(grid.k2_max) / dk + 0.5)) + 1


def _unit_index(grid: SlabGrid3D, sl: slice, dk: float, nbins: int) -> torch.Tensor:
    """Shell of each mode of the planes ``sl``, with exactly the arithmetic of ``spectra()``."""
    kz = grid.kz3[sl]
    k2 = grid.kx3 * grid.kx3 + grid.ky3 * grid.ky3 + kz * kz
    return torch.floor(torch.sqrt(k2) / dk + 0.5).to(torch.int64).clamp_(max=nbins - 1)


def _bin(idx: torch.Tensor, weights: torch.Tensor, nbins: int) -> torch.Tensor:
    # scatter_add into a fixed-size array: unlike CUDA bincount it never reads the
    # index range back to the host, so binning adds no device synchronisation.
    out = torch.zeros(nbins, dtype=_F64, device=weights.device)
    return out.scatter_add_(0, idx.reshape(-1).to(torch.int64), weights.reshape(-1).to(_F64))


def _reduce(comm, parts: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Sum device arrays over ranks with a single small allreduce (one host copy)."""
    names = list(parts)
    flat = torch.cat([parts[n].reshape(-1).to(_F64) for n in names]).cpu().numpy()
    total = comm.allreduce(flat, "sum")
    out, offset = {}, 0
    for n in names:
        size = parts[n].numel()
        out[n] = total[offset:offset + size].reshape(tuple(parts[n].shape))
        offset += size
    return out


def _ratio(num: float, den: float) -> float:
    if den > 0.0:
        return float(num / den)
    return float("inf") if num > 0.0 else float("nan")


def _energy_density(grid: SlabGrid3D, X: torch.Tensor) -> torch.Tensor:
    return 0.5 * grid.weight3 * (X.real ** 2 + X.imag ** 2).sum(dim=0)


def _shell_energy(grid: SlabGrid3D, S: torch.Tensor, chunk: int) -> torch.Tensor:
    """Unit-shell sums of ``w |S|^2 / 2`` over the components of ``S`` (device, float64)."""
    dk, nbins = _unit_geometry(grid)
    acc = torch.zeros(nbins, dtype=_F64, device=grid.device)
    for sl in _z_chunks(grid, chunk):
        acc += _bin(_unit_index(grid, sl, dk, nbins), _energy_density(grid, S[:, sl]), nbins)
    return acc


# --- anisotropy ----------------------------------------------------------------

def _parse_axis(spec) -> Tuple[str, Optional[np.ndarray]]:
    if spec is None:
        spec = "auto"
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "auto":
            return "auto", None
        if len(s) == 1 and s in "xyz":
            e = np.zeros(3)
            e["xyz".index(s)] = 1.0
            return s, e
        raise ValueError(f"anisotropy axis must be 'auto', 'x', 'y', 'z' or a 3-vector, got {spec!r}")
    v = np.asarray(spec, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0.0:
        raise ValueError(f"anisotropy axis vector must be three finite numbers, not all zero, got {spec!r}")
    return "vector", v / np.linalg.norm(v)


def _mean_field_axis(grid: SlabGrid3D, comm, Sb: Optional[torch.Tensor], chunk: int):
    """(unit vector, label) of the mean magnetic field, or z when it is absent or negligible."""
    z = np.array([0.0, 0.0, 1.0])
    if Sb is None:
        return z, "auto: z (no magnetic field)"
    d = grid.decomp
    mean = (Sb[:, 0, 0, 0].real.to(_F64) if d.ky0 == 0
            else torch.zeros(3, dtype=_F64, device=grid.device))
    energy = torch.zeros(1, dtype=_F64, device=grid.device)  # <|b|^2>
    for sl in _z_chunks(grid, chunk):
        energy += (2.0 * _energy_density(grid, Sb[:, sl]).to(_F64)).sum()
    red = _reduce(comm, {"mean": mean, "energy": energy})
    mean, brms = red["mean"], math.sqrt(max(float(red["energy"][0]), 0.0))
    norm = float(np.linalg.norm(mean))
    if norm > _AXIS_TOLERANCE * brms and norm > 0.0:
        return mean / norm, "auto: mean field"
    return z, "auto: z (no mean field)"


class _Anisotropy:
    """Binning of ``(k_par, k_perp)`` spectra and the component and gradient anisotropies."""

    def __init__(self, grid: SlabGrid3D, comm, fields: Dict[str, torch.Tensor], spec, chunk: int):
        """``fields``: ``{"kinetic": S}`` and optionally ``{"magnetic": S}``, ``S`` ``(3, Kz, nky, Kx)``."""
        label, e = _parse_axis(spec)
        if e is None:
            e, label = _mean_field_axis(grid, comm, fields.get("magnetic"), chunk)
        self.grid, self.fields, self.chunk = grid, fields, chunk
        self.e, self.label = e, label
        self.dk = 2.0 * np.pi / max(grid.Lx, grid.Ly, grid.Lz)
        d = grid.decomp
        corner = np.array([2.0 * np.pi * d.cx / grid.Lx, 2.0 * np.pi * d.cy / grid.Ly, 2.0 * np.pi * d.cz / grid.Lz])
        par = perp = 0.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):  # both norms are convex: extremes at corners
            k = corner * np.array(signs)
            kp = float(k @ e)
            par = max(par, abs(kp))
            perp = max(perp, float(np.linalg.norm(k - kp * e)))
        self.npar = int(math.floor(par / self.dk + 0.5)) + 1
        self.nperp = int(math.floor(perp / self.dk + 0.5)) + 1

    def accumulate(self) -> Dict[str, torch.Tensor]:
        g = self.grid
        dev = g.device
        w = g.weight3.to(_F64)
        kx, ky, kz = _k64(g)
        ex, ey, ez = (float(c) for c in self.e)
        parts = {}
        for name in self.fields:
            parts[f"{name}_2d"] = torch.zeros(self.npar * self.nperp, dtype=_F64, device=dev)
            parts[f"{name}_sums"] = torch.zeros(4, dtype=_F64, device=dev)
        for sl in _z_chunks(g, self.chunk):
            kzs = _plane(kz, sl)
            kpar = kx * ex + ky * ey + kzs * ez  # signed k . e
            perp2 = (kx - kpar * ex) ** 2 + (ky - kpar * ey) ** 2 + (kzs - kpar * ez) ** 2
            ipar = torch.floor(kpar.abs() / self.dk + 0.5).to(torch.int64).clamp_(max=self.npar - 1)
            iperp = torch.floor(perp2.sqrt() / self.dk + 0.5).to(torch.int64).clamp_(max=self.nperp - 1)
            idx = ipar * self.nperp + iperp
            kpar2 = kpar * kpar
            for name, S in self.fields.items():
                a, b = S[:, sl].real.to(_F64), S[:, sl].imag.to(_F64)
                tot = (a * a + b * b).sum(dim=0)
                along = (ex * a[0] + ey * a[1] + ez * a[2]) ** 2 + (ex * b[0] + ey * b[1] + ez * b[2]) ** 2
                parts[f"{name}_2d"] += _bin(idx, 0.5 * w * tot, self.npar * self.nperp)
                parts[f"{name}_sums"] += torch.stack([(w * along).sum(), (w * (tot - along)).sum(),
                                                      (w * kpar2 * tot).sum(), (w * perp2 * tot).sum()])
        return parts

    def finish(self, red: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {"k_par": np.arange(self.npar) * self.dk, "k_perp": np.arange(self.nperp) * self.dk}
        letter = {"kinetic": "u", "magnetic": "b"}
        for name in self.fields:
            e2 = red[f"{name}_2d"].reshape(self.npar, self.nperp)
            s = red[f"{name}_sums"]
            out[f"{name}_2d"] = e2
            out[f"{name}_par"] = e2.sum(axis=1)
            out[f"{name}_perp"] = e2.sum(axis=0)
            out[f"{letter[name]}_component_anisotropy"] = np.asarray(_ratio(s[0], s[1]))
            out[f"{letter[name]}_gradient_anisotropy"] = np.asarray(_ratio(s[2], s[3]))
        return out

    def attrs(self) -> dict:
        return {"axis": self.label, "axis_x": float(self.e[0]), "axis_y": float(self.e[1]),
                "axis_z": float(self.e[2]),
                "anisotropy_bins": "k_par = |k.e|, k_perp = |k - (k.e)e|; unit bins of 2 pi/L_max centred on integers"}


def _anisotropy_requested(config: Optional[dict]) -> Tuple[bool, object]:
    cfg = _section(config, "anisotropy")
    return bool(cfg.get("enabled", True)), cfg.get("axis", "auto")


# --- spectral diagnostics ------------------------------------------------------

def spectral_diagnostics(stepper, config: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Shell spectra, helicities, dissipation spectra and anisotropy (see the module docstring).

    Collective.  Incompressible steppers need no transform; compressible ones
    transform ``u`` and ``sqrt(rho) u`` (6 forward transforms).
    """
    config = config or {}
    if isinstance(stepper, IncompressibleSlabStepper):
        return _incompressible_spectral(stepper, config)
    if isinstance(stepper, CompressibleSlabStepper):
        return _compressible_spectral(stepper, config)
    raise TypeError(f"spectral_diagnostics: unsupported stepper {type(stepper).__name__}")


def _incompressible_spectral(st: IncompressibleSlabStepper, config: dict) -> Dict[str, np.ndarray]:
    g, comm = st.grid, st.comm
    chunk = _chunk_modes(config)
    current = bool(config.get("current_helicity", False))
    do_aniso, axis = _anisotropy_requested(config)
    aniso = None
    if do_aniso:
        fields = {"kinetic": st.uhat[0:3]}
        if st.mhd:
            fields["magnetic"] = st.uhat[3:6]
        aniso = _Anisotropy(g, comm, fields, axis, chunk)  # may reduce (auto axis)

    dk, nbins = _unit_geometry(g)
    names = ["kinetic", "kinetic_helicity", "dissipation_kinetic"]
    if st.mhd:
        names += ["magnetic", "cross_helicity", "magnetic_helicity", "dissipation_magnetic"]
        if current:
            names.append("current_helicity")
    parts = {n: torch.zeros(nbins, dtype=_F64, device=g.device) for n in names}
    w = g.weight3.to(_F64)
    kx, ky, kz = _k64(g)
    for sl in _z_chunks(g, chunk):
        idx = _unit_index(g, sl, dk, nbins)
        kzs = _plane(kz, sl)
        k2 = kx * kx + ky * ky + kzs * kzs

        def triple(a, b):
            """``k . (a x b)`` for real and imaginary parts ``a``, ``b``: ``Re(v . (i k x v)^*) / 2``."""
            return (kx * (a[1] * b[2] - a[2] * b[1]) + ky * (a[2] * b[0] - a[0] * b[2])
                    + kzs * (a[0] * b[1] - a[1] * b[0]))

        ua, ub = st.uhat[0:3, sl].real.to(_F64), st.uhat[0:3, sl].imag.to(_F64)
        eu = (ua * ua + ub * ub).sum(dim=0)
        parts["kinetic"] += _bin(idx, 0.5 * w * eu, nbins)
        parts["kinetic_helicity"] += _bin(idx, 2.0 * w * triple(ua, ub), nbins)
        parts["dissipation_kinetic"] += _bin(idx, w * st.damping(k2) * eu, nbins)
        if not st.mhd:
            continue
        ba, bb = st.uhat[3:6, sl].real.to(_F64), st.uhat[3:6, sl].imag.to(_F64)
        eb = (ba * ba + bb * bb).sum(dim=0)
        parts["magnetic"] += _bin(idx, 0.5 * w * eb, nbins)
        parts["cross_helicity"] += _bin(idx, w * (ua * ba + ub * bb).sum(dim=0), nbins)
        jb = 2.0 * w * triple(ba, bb)  # w Re(j . b^*), j = i k x b
        parts["magnetic_helicity"] += _bin(idx, torch.where(k2 > 0, jb / torch.where(k2 > 0, k2, torch.ones_like(k2)),
                                                            torch.zeros_like(jb)), nbins)
        parts["dissipation_magnetic"] += _bin(idx, w * st.damping(k2, True) * eb, nbins)
        if current:
            parts["current_helicity"] += _bin(idx, jb, nbins)
    if aniso is not None:
        parts.update({f"aniso_{n}": v for n, v in aniso.accumulate().items()})
    red = _reduce(comm, parts)
    out = {"k": np.arange(nbins) * dk}
    out.update({n: red[n] for n in names})
    extra = {}
    if aniso is not None:
        out.update(aniso.finish({n[len("aniso_"):]: v for n, v in red.items() if n.startswith("aniso_")}))
        extra = aniso.attrs()
    out["attrs"] = _attrs(st, system="incompressible_mhd" if st.mhd else "incompressible_hydro", **extra)
    return out


def _compressible_spectral(st: CompressibleSlabStepper, config: dict) -> Dict[str, np.ndarray]:
    g, comm = st.grid, st.comm
    chunk = _chunk_modes(config)
    do_aniso, axis = _anisotropy_requested(config)
    dk, nbins = _unit_geometry(g)
    U = st.physical()
    rho = U[0]
    sr = torch.sqrt(rho)
    Su = g.zeros_spectral(3)
    weighted = torch.zeros(nbins, dtype=_F64, device=g.device)
    Sw = g.zeros_spectral(1)
    for c in range(3):  # one component at a time: peak is the state plus ~2 fields
        g.forward((U[1 + c] / rho)[None], normalized=True, out=Su[c:c + 1])
        g.forward((U[1 + c] / sr)[None], normalized=True, out=Sw)
        weighted += _shell_energy(g, Sw, chunk)
    del Sw, sr
    parts = {"kinetic": _shell_energy(g, Su, chunk), "kinetic_density_weighted": weighted,
             "density": _shell_energy(g, st.uhat[0:1], chunk)}
    fields = {"kinetic": Su}
    if st.mhd:
        parts["magnetic"] = _shell_energy(g, st.uhat[5:8], chunk)
        fields["magnetic"] = st.uhat[5:8]
    aniso = _Anisotropy(g, comm, fields, axis, chunk) if do_aniso else None
    if aniso is not None:
        parts.update({f"aniso_{n}": v for n, v in aniso.accumulate().items()})
    red = _reduce(comm, parts)
    out = {"k": np.arange(nbins) * dk}
    out.update({n: v for n, v in red.items() if not n.startswith("aniso_")})
    extra = {}
    if aniso is not None:
        out.update(aniso.finish({n[len("aniso_"):]: v for n, v in red.items() if n.startswith("aniso_")}))
        extra = aniso.attrs()
    out["attrs"] = _attrs(st, system=st.system, **extra)
    return out


# --- transfer diagnostics ------------------------------------------------------

def transfer_diagnostics(stepper, config: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Shell transfer spectra and fluxes, and optionally shell-to-shell matrices (module docstring).

    Collective: every rank makes the same transforms in the same order.
    Raises ``NotImplementedError`` for compressible steppers.
    """
    if isinstance(stepper, CompressibleSlabStepper):
        raise NotImplementedError("transfer diagnostics are implemented for the incompressible solvers only")
    if not isinstance(stepper, IncompressibleSlabStepper):
        raise TypeError(f"transfer_diagnostics: unsupported stepper {type(stepper).__name__}")
    config = config or {}
    cfg = _section(config, "transfer")
    chunk = _chunk_modes(config)
    grid_batch = stepper.grid.fft.batch or _TRANSFER_BATCH
    batch = max(1, min(int(cfg.get("batch", _TRANSFER_BATCH)), grid_batch))
    out = _transfer_shells(stepper, chunk, batch)
    extra = {"transfer_sign": "T_X(K) > 0: shell K gains; Pi_X(k) = -sum_{K<=k} T_X(K)"}
    if cfg.get("matrix", False):
        out.update(_transfer_matrix(stepper, cfg.get("shells"), chunk, batch, bool(cfg.get("cache_fields", False))))
        extra["matrix"] = "T_X_matrix[K, Q]: transfer from shell Q (columns) to shell K (rows), shells in shell_edges"
    out["attrs"] = _attrs(stepper, system="incompressible_mhd" if stepper.mhd else "incompressible_hydro", **extra)
    return out


class _Contraction:
    """``acc += bincount(shell, sign w k_j Im(conj(X) P))``: the shell-binned term ``X_i d_j P`` of a transfer.

    ``<X d_j P> = sum_k w Re(X_k^* i k_j P_k) = -sum_k w k_j Im(X_k^* P_k)``, so the transfer of a
    term ``-<X_K d_j P>`` is ``+sum w k_j Im(X^* P)`` and ``sign`` is that of the transfer.
    """

    def __init__(self, grid: SlabGrid3D, idx: torch.Tensor, nbins: int, chunk: int):
        self.grid, self.idx, self.nbins, self.chunk = grid, idx, nbins, chunk
        self.w = grid.weight3
        self.k = (grid.kx3, grid.ky3, grid.kz3)

    def add(self, acc: torch.Tensor, X: torch.Tensor, P: torch.Tensor, j: int, sign: float) -> None:
        kj = self.k[j]
        for sl in _z_chunks(self.grid, self.chunk):
            x, p = X[sl], P[sl]
            im = x.real * p.imag - x.imag * p.real  # Im(conj(x) p)
            weights = (sign * self.w * _plane(kj, sl)) * im
            acc += _bin(self.idx[sl].long(), weights, self.nbins)


def _index_field(grid: SlabGrid3D, chunk: int, build: Callable[[slice], torch.Tensor]) -> torch.Tensor:
    """Per-mode shell index over the whole local spectrum, built chunk by chunk, stored as int32."""
    idx = torch.empty(grid.decomp.spectral_shape, dtype=torch.int32, device=grid.device)
    for sl in _z_chunks(grid, chunk):
        idx[sl] = build(sl).to(torch.int32)
    return idx


def _run_products(grid: SlabGrid3D, pbuf: torch.Tensor, jobs: list, handler: Callable) -> None:
    """Forward-transform the products ``A * B`` of ``jobs = [(A, B, tag)]`` in groups of ``len(pbuf)``.

    The transform count and order depend only on ``jobs``, never on local data.
    """
    size = pbuf.shape[0]
    for g0 in range(0, len(jobs), size):
        group = jobs[g0:g0 + size]
        for n, (A, B, _) in enumerate(group):
            torch.mul(A, B, out=pbuf[n])
        Ph = grid.forward(pbuf[:len(group)], normalized=True)
        for n, (_, _, tag) in enumerate(group):
            handler(tag, Ph[n])
        del Ph


def _transfer_shells(st: IncompressibleSlabStepper, chunk: int, batch: int) -> Dict[str, np.ndarray]:
    """``T_X(K)`` on the unit shells with the unfiltered field in the Q slot, and the fluxes."""
    g = st.grid
    dk, nbins = _unit_geometry(g)
    idx = _index_field(g, chunk, lambda sl: _unit_index(g, sl, dk, nbins))
    con = _Contraction(g, idx, nbins, chunk)
    channels = ["uu", "bb", "ub", "bu"] if st.mhd else ["uu"]
    acc = {ch: torch.zeros(nbins, dtype=_F64, device=g.device) for ch in channels}
    uh = st.uhat
    U = [uh[i] for i in range(3)]
    B = [uh[3 + i] for i in range(3)] if st.mhd else None
    pbuf = torch.empty((batch,) + g.decomp.physical_shape, dtype=g.rdtype, device=g.device)

    def handler(tag, P):
        for ch, X, j, sign in tag:
            con.add(acc[ch], X, P, j, sign)

    # T_uu(k) = sum_ij w k_j Im(u_i^* [u_j u_i]): the product is symmetric, so 6 transforms.
    u = g.inverse(uh[0:3])
    jobs = []
    for a in range(3):
        for b in range(a, 3):
            tag = [("uu", U[a], b, 1.0)] + ([("uu", U[b], a, 1.0)] if a != b else [])
            jobs.append((u[a], u[b], tuple(tag)))
    _run_products(g, pbuf, jobs, handler)
    del jobs, u  # the job list holds views of u
    if st.mhd:
        # T_ub(k) = -sum_ij w k_j Im(u_i^* [b_j b_i]); symmetric product, 6 transforms.
        b = g.inverse(uh[3:6])
        jobs = []
        for a in range(3):
            for c in range(a, 3):
                tag = [("ub", U[a], c, -1.0)] + ([("ub", U[c], a, -1.0)] if a != c else [])
                jobs.append((b[a], b[c], tuple(tag)))
        _run_products(g, pbuf, jobs, handler)
        del jobs
        # M_ac = u_a b_c: T_bb(k) += w k_a Im(b_c^* M_ac), T_bu(k) -= w k_c Im(b_a^* M_ac); 9 transforms.
        for a in range(3):
            ua = g.inverse(uh[a:a + 1])[0]
            jobs = [(ua, b[c], (("bb", B[c], a, 1.0), ("bu", B[a], c, -1.0))) for c in range(3)]
            _run_products(g, pbuf, jobs, handler)
            del jobs, ua
        del b
    del pbuf
    red = _reduce(st.comm, {ch: v for ch, v in acc.items()})
    out = {"k": np.arange(nbins) * dk}
    for ch in channels:
        out[f"T_{ch}"] = red[ch]
        out[f"Pi_{ch}"] = -np.cumsum(red[ch])
    return out


def _default_edges(grid: SlabGrid3D) -> np.ndarray:
    kmax = math.sqrt(grid.k2_max) / (2.0 * np.pi / max(grid.Lx, grid.Ly, grid.Lz))
    top = 1
    while top <= kmax:
        top *= 2
    return np.array([0.0] + [2.0 ** m for m in range(int(round(math.log2(top))) + 1)])


def _edges(grid: SlabGrid3D, shells) -> np.ndarray:
    if shells is None:
        return _default_edges(grid)
    edges = np.asarray(shells, dtype=np.float64).reshape(-1)
    if edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("transfer shell edges must be at least two strictly increasing numbers")
    return edges


def _transfer_matrix(st: IncompressibleSlabStepper, shells, chunk: int, batch: int,
                     cache: bool) -> Dict[str, np.ndarray]:
    """Shell-to-shell matrices ``T_X(K, Q)`` on the shells ``edges[n] <= |k|/dk < edges[n+1]``."""
    g, comm = st.grid, st.comm
    edges = _edges(g, shells)
    ns = edges.size - 1
    dk = 2.0 * np.pi / max(g.Lx, g.Ly, g.Lz)
    edge_t = torch.as_tensor(edges, device=g.device)
    kx, ky, kz = _k64(g)

    def build(sl):
        kzs = _plane(kz, sl)
        n = torch.bucketize(torch.sqrt(kx * kx + ky * ky + kzs * kzs) / dk, edge_t, right=True) - 1
        return torch.where((n < 0) | (n >= ns), torch.full_like(n, ns), n)  # ns: outside every shell

    idx = _index_field(g, chunk, build)
    con = _Contraction(g, idx, ns + 1, chunk)
    count = torch.zeros(ns + 1, dtype=_F64, device=g.device)
    for sl in _z_chunks(g, chunk):
        flat = idx[sl].reshape(-1).to(torch.int64)
        count.scatter_add_(0, flat, torch.ones(flat.numel(), dtype=_F64, device=g.device))
    present = _reduce(comm, {"count": count})["count"][:ns] > 0  # global: every rank skips the same shells
    channels = ["uu", "bb", "ub", "bu"] if st.mhd else ["uu"]
    matrix = {ch: torch.zeros((ns, ns), dtype=_F64, device=g.device) for ch in channels}
    pbuf = torch.empty((batch,) + g.decomp.physical_shape, dtype=g.rdtype, device=g.device)
    uh = st.uhat

    def pair(j: int) -> torch.Tensor:
        """Physical ``(u_j, b_j)`` (``u_j`` alone without a magnetic field)."""
        return g.inverse(uh[[j, 3 + j]] if st.mhd else uh[j:j + 1])

    held = [pair(j) for j in range(3)] if cache else None
    col: Dict[str, torch.Tensor] = {}

    def handler(tag, P):
        for ch, X, j, sign in tag:
            con.add(col[ch], X, P, j, sign)

    for q in range(ns):
        if not present[q]:
            continue
        col = {ch: torch.zeros(ns + 1, dtype=_F64, device=g.device) for ch in channels}
        for i in range(3):
            src = uh[[i, 3 + i]] if st.mhd else uh[i:i + 1]
            keep = torch.zeros_like(src)
            for sl in _z_chunks(g, chunk):
                keep[:, sl] = torch.where(idx[sl] == q, src[:, sl], torch.zeros_like(src[:, sl]))
            Q = g.inverse(keep)  # (u_Q,i[, b_Q,i])
            del keep
            Xu, Xb = uh[i], (uh[3 + i] if st.mhd else None)
            for j in range(3):
                M = held[j] if cache else pair(j)
                jobs = [(M[0], Q[0], (("uu", Xu, j, 1.0),))]
                if st.mhd:
                    jobs += [(M[0], Q[1], (("bb", Xb, j, 1.0),)),
                             (M[1], Q[1], (("ub", Xu, j, -1.0),)),
                             (M[1], Q[0], (("bu", Xb, j, -1.0),))]
                _run_products(g, pbuf, jobs, handler)
                del jobs, M
            del Q
        for ch in channels:
            matrix[ch][:, q] = col[ch][:ns]
    del held, pbuf
    red = _reduce(comm, {f"T_{ch}_matrix": matrix[ch] for ch in channels})
    red["shell_edges"] = edges
    return red
