"""Snapshots, restart and small diagnostics files of the distributed solvers.

A snapshot is a directory::

    snap_000012/
        meta.json            global sizes, time, physics, forcing state, config
        spec.00000.npy ...   one file per rank: its retained spectrum (C, Kz, nky, Kx)
        resid.00000.npy ...  only after an MHD accepted-state clamp: one file per rank,
                             its z slab of the out-of-band residual (5, nz, Ny, Nx)

The state is stored as the retained Fourier modes divided by N.  That is the
solver's own representation, it is exact for the band-limited fields the
solvers evolve, and it is 8/27 of the half spectrum, about 0.3 of a real field
per component.  ``meta.json`` is written last, by rank 0, so a directory that
has it is complete.

``read_spectrum`` rebuilds a rank's slab from any snapshot of the same system:
the rank count may differ and so may the resolution (modes are matched by
integer wavenumber; new modes start at zero and modes that no longer fit are
dropped), which is how a run is restarted on more GPUs or refined.

The residual is the content above the dealias cutoff that the MHD
accepted-state limiter leaves in a clamped state; the next step needs it (see
``compressible.py``).  It lives in physical space on the writer's z slabs, so
``read_residual`` returns it only for a restart with the same grid and rank
count, which then reproduces the uninterrupted run; otherwise it is dropped
with a ``RuntimeWarning`` and the first step after the restart differs from the
uninterrupted run by the clamp's out-of-band part.

``load_snapshot`` assembles a whole snapshot in one process for analysis.

Small analysis files (slices, shell spectra, diagnostics) are written by rank 0
as ``.h5`` (``io.format`` ``hdf5``) or ``.npz`` (``npy``) through ``write_arrays``;
``resolve_format`` turns ``auto`` into the first of the two that can be used.
The physical-space field output lives in ``output.py``.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .decomp import compact_modes, cutoff
from .grid import SlabGrid3D

FORMAT = "phrike-parallel-1"

_FIELD_NAMES = {
    "euler": ["rho", "mx", "my", "mz", "E"],
    "mhd": ["rho", "mx", "my", "mz", "E", "Bx", "By", "Bz"],
    "ins": ["ux", "uy", "uz"],
    "imhd": ["ux", "uy", "uz", "bx", "by", "bz"],
}


def get_h5py():
    """The ``h5py`` module, or ``None`` when it cannot be imported."""
    try:
        import h5py
    except ImportError:
        return None
    return h5py


def resolve_format(fmt: Optional[str] = "auto") -> str:
    """``io.format`` as ``"hdf5"`` or ``"npy"``; ``auto`` is HDF5 when h5py imports."""
    fmt = str(fmt or "auto").lower()
    if fmt not in ("auto", "hdf5", "npy"):
        raise ValueError(f"io.format must be 'auto', 'hdf5' or 'npy', got {fmt!r}")
    have = get_h5py() is not None
    if fmt == "auto":
        return "hdf5" if have else "npy"
    if fmt == "hdf5" and not have:
        raise ImportError("io.format 'hdf5' needs h5py (or use 'auto' or 'npy')")
    return fmt


def field_names(system: str) -> List[str]:
    return list(_FIELD_NAMES[system])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def rank_file(path: str, rank: int) -> str:
    return os.path.join(path, f"spec.{rank:05d}.npy")


def residual_file(path: str, rank: int) -> str:
    return os.path.join(path, f"resid.{rank:05d}.npy")


def write_snapshot(path: str, grid: SlabGrid3D, system: str, spectrum: torch.Tensor,
                   t: float, step: int, extra: Optional[Dict[str, Any]] = None,
                   residual: Optional[torch.Tensor] = None) -> str:
    """Write this rank's part of a snapshot (collective: call on every rank).

    ``residual`` is the stepper's out-of-band residual (``None`` while the
    state is band-limited); every rank passes its own slab or all pass ``None``.
    """
    comm = grid.comm
    if comm.is_root:
        os.makedirs(path, exist_ok=True)
        stale = os.path.join(path, "meta.json")
        if os.path.exists(stale):
            os.remove(stale)
    comm.barrier()
    target = rank_file(path, comm.rank)
    tmp = target + ".tmp"
    with open(tmp, "wb") as handle:
        np.save(handle, spectrum.detach().cpu().numpy())
    os.replace(tmp, target)
    extra_file = residual_file(path, comm.rank)
    if residual is not None:
        with open(extra_file + ".tmp", "wb") as handle:
            np.save(handle, residual.detach().cpu().numpy())
        os.replace(extra_file + ".tmp", extra_file)
    elif os.path.exists(extra_file):
        os.remove(extra_file)  # left by an earlier snapshot written to this directory
    comm.barrier()
    if comm.is_root:
        d = grid.decomp
        meta = {
            "format": FORMAT, "system": system, "fields": field_names(system),
            "ncomp": int(spectrum.shape[0]), "t": float(t), "step": int(step),
            "Nx": d.Nx, "Ny": d.Ny, "Nz": d.Nz, "Lx": grid.Lx, "Ly": grid.Ly, "Lz": grid.Lz,
            "Kx": d.Kx, "Ky": d.Ky, "Kz": d.Kz, "nprocs": d.nprocs,
            "ky_counts": [int(c) for c in d.ky_counts], "precision": grid.precision,
            "normalization": "retained rfftn modes divided by Nx*Ny*Nz; kx >= 0 half",
        }
        if residual is not None:
            meta["residual"] = {"ncomp": int(residual.shape[0]), "nprocs": d.nprocs,
                                "z_counts": [int(c) for c in d.z_counts]}
        meta.update(_json_ready(extra or {}))
        tmp = os.path.join(path, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(path, "meta.json"))
    return path


def read_meta(path: str) -> Dict[str, Any]:
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} snapshot")
    return meta


def _index_map(new_modes: np.ndarray, old_n: int):
    """Positions in the new axis and in an old compact axis of the modes both keep."""
    c_old = cutoff(old_n)
    keep = np.abs(new_modes) <= c_old
    new_idx = np.nonzero(keep)[0]
    values = new_modes[keep]
    old_idx = np.where(values >= 0, values, values + 2 * c_old + 1)
    return new_idx, old_idx


def read_spectrum(path: str, grid: SlabGrid3D, system: str, meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """This rank's retained spectrum from a snapshot of any rank count or resolution."""
    meta = meta or read_meta(path)
    if meta["system"] != system:
        raise ValueError(f"snapshot holds a {meta['system']!r} state, the run needs {system!r}")
    for key, value in (("Lx", grid.Lx), ("Ly", grid.Ly), ("Lz", grid.Lz)):
        if abs(float(meta[key]) - value) > 1e-12 * max(1.0, abs(value)):
            raise ValueError(f"snapshot {key}={meta[key]} does not match the grid ({value})")
    d = grid.decomp
    ncomp = int(meta["ncomp"])
    out = np.zeros((ncomp,) + d.spectral_shape, dtype=np.complex128)
    new_z, old_z = _index_map(d.modes_z(), int(meta["Nz"]))
    nkx = min(d.Kx, int(meta["Kx"]))
    old_rows = compact_modes(int(meta["Ny"]))  # integer ky of every stored row
    local = {int(v): j for j, v in enumerate(d.modes_y_local())}
    start = 0
    for rank, count in enumerate(meta["ky_counts"]):
        rows = old_rows[start:start + count]
        take = [(j, local[int(v)]) for j, v in enumerate(rows) if int(v) in local]
        start += count
        if not take:
            continue
        data = np.load(rank_file(path, rank), mmap_mode="r")
        old_y = np.asarray([j for j, _ in take])
        new_y = np.asarray([j for _, j in take])
        block = np.asarray(data[:, old_z][:, :, old_y][:, :, :, :nkx])
        out[np.ix_(range(ncomp), new_z, new_y, range(nkx))] = block
    return torch.as_tensor(out, dtype=grid.cdtype, device=grid.device)


def read_residual(path: str, grid: SlabGrid3D, system: str,
                  meta: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
    """This rank's out-of-band residual, or ``None`` if the snapshot has none or it cannot be reused.

    The residual is stored in physical space per writer rank, so it is read
    back only when the grid sizes and the rank count match the snapshot's;
    otherwise it is dropped with a ``RuntimeWarning`` (rank 0 only).
    """
    meta = meta or read_meta(path)
    info = meta.get("residual")
    if info is None:
        return None
    d = grid.decomp
    same = ((int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])) == (d.Nx, d.Ny, d.Nz)
            and int(info["nprocs"]) == d.nprocs and list(info["z_counts"]) == [int(c) for c in d.z_counts])
    if not same:
        if grid.comm.is_root:
            warnings.warn(
                f"{path} holds the out-of-band part of a limiter-clamped state, written on "
                f"{info['nprocs']} rank(s) for a {meta['Nx']}x{meta['Ny']}x{meta['Nz']} grid; "
                f"this run uses {d.nprocs} rank(s) and {d.Nx}x{d.Ny}x{d.Nz}, so it is dropped "
                "and the first step after the restart differs from the uninterrupted run.",
                RuntimeWarning, stacklevel=2)
        return None
    data = np.load(residual_file(path, grid.comm.rank))
    return torch.as_tensor(data, dtype=grid.rdtype, device=grid.device)


def load_snapshot(path: str, physical: bool = False) -> Dict[str, Any]:
    """Assemble a snapshot in this process: ``spectrum`` (C, Kz, Ky, Kx), optionally ``fields``.

    ``fields`` is the real-space state ``(C, Nz, Ny, Nx)``; it needs the
    memory of the full grid, so use it for moderate sizes or cut it up.  When the
    snapshot has an out-of-band residual it is returned as ``residual``
    ``(5, Nz, Ny, Nx)`` and added to the first five ``fields``.
    """
    meta = read_meta(path)
    parts = [np.load(rank_file(path, r)) for r in range(int(meta["nprocs"]))]
    spectrum = np.concatenate(parts, axis=2)
    out: Dict[str, Any] = {"meta": meta, "t": meta["t"], "spectrum": spectrum,
                           "modes_z": compact_modes(meta["Nz"]), "modes_y": compact_modes(meta["Ny"]),
                           "modes_x": np.arange(meta["Kx"])}
    if physical:
        Nx, Ny, Nz = meta["Nx"], meta["Ny"], meta["Nz"]
        iz = np.where(out["modes_z"] >= 0, out["modes_z"], out["modes_z"] + Nz)
        iy = np.where(out["modes_y"] >= 0, out["modes_y"], out["modes_y"] + Ny)
        padded = np.zeros((spectrum.shape[0], Nz, Ny, Nx // 2 + 1), dtype=np.complex128)
        padded[np.ix_(range(spectrum.shape[0]), iz, iy, range(meta["Kx"]))] = spectrum
        out["fields"] = np.fft.irfftn(padded, s=(Nz, Ny, Nx), axes=(-3, -2, -1)) * (Nx * Ny * Nz)
    if "residual" in meta:
        out["residual"] = np.concatenate(
            [np.load(residual_file(path, r)) for r in range(int(meta["nprocs"]))], axis=1)
        if physical:
            out["fields"][:out["residual"].shape[0]] += out["residual"]
    return out


# --- small files written by rank 0 ----------------------------------------------


class History:
    """A CSV time series appended by rank 0 (one row per monitoring call)."""

    def __init__(self, path: str, enabled: bool, append: bool = False):
        self.path = path
        self.enabled = bool(enabled)
        self.columns: Optional[List[str]] = None
        if self.enabled and append and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                header = handle.readline().strip()
            self.columns = header.split(",") if header else None
        elif self.enabled and os.path.exists(path):
            os.remove(path)

    def write(self, row: Dict[str, float]) -> None:
        if not self.enabled:
            return
        if self.columns is None:
            self.columns = list(row)
            with open(self.path, "w", encoding="utf-8") as handle:
                handle.write(",".join(self.columns) + "\n")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(",".join(repr(float(row.get(c, float("nan")))) for c in self.columns) + "\n")


def h5_attr(value: Any) -> Any:
    """An HDF5-attribute-safe form of ``value`` (None, dicts and lists of strings become text)."""
    if value is None:
        return "none"
    if isinstance(value, dict) or (isinstance(value, (list, tuple)) and any(
            isinstance(v, (str, dict, list, tuple, type(None))) for v in value)):
        return json.dumps(_json_ready(value))
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def write_arrays(stem: str, arrays: Dict[str, Any], attrs: Optional[Dict[str, Any]] = None,
                 fmt: str = "npy") -> str:
    """Write named arrays and scalar attributes to ``stem.h5`` or ``stem.npz`` (one process).

    In ``.npz`` the attributes are stored as 0-d entries next to the arrays;
    in ``.h5`` they are file attributes.  The file appears complete (written
    under a temporary name and renamed).  Returns the path.
    """
    attrs = dict(attrs or {})
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    if resolve_format(fmt) == "hdf5":
        h5py = get_h5py()
        path, tmp = stem + ".h5", stem + ".tmp.h5"
        with h5py.File(tmp, "w") as handle:
            for name, value in arrays.items():
                handle.create_dataset(name, data=np.asarray(value))
            for name, value in attrs.items():
                handle.attrs[name] = h5_attr(value)
    else:
        path, tmp = stem + ".npz", stem + ".tmp.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()},
                 **{k: np.asarray(v) for k, v in attrs.items() if k not in arrays})
    os.replace(tmp, path)
    return path


def write_slice(path: str, grid: SlabGrid3D, fields: torch.Tensor, names: List[str],
                t: float, z_index: Optional[int] = None, fmt: str = "npy") -> None:
    """Save the xy plane at ``z_index`` (default mid-box) of resident fields; the owning rank writes it.

    ``path`` is the file stem or a name ending in ``.npz``/``.h5``.
    """
    d = grid.decomp
    z_index = d.Nz // 2 if z_index is None else int(z_index)
    if not d.z0 <= z_index < d.z0 + d.nz:
        return
    plane = fields[:, z_index - d.z0].detach().cpu().numpy()
    stem = os.path.splitext(path)[0] if path.endswith((".npz", ".h5")) else path
    arrays = {"x": grid._x, "y": grid._y, **{name: plane[i] for i, name in enumerate(names)}}
    write_arrays(stem, arrays, {"t": float(t), "z_index": z_index, "z": z_index * grid.dz}, fmt)
