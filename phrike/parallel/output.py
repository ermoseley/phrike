"""Analysis output of the distributed solvers: physical-space fields, slices, spectra, small files.

Restart snapshots (``io.py``) are the solver's spectral state and stay at full
precision.  This module writes what is read by analysis and visualisation:

``fields/fields_NNNNNN.h5`` (+ ``.xmf``) or ``fields/fields_NNNNNN/``
    one ``(Nz, Ny, Nx)`` dataset per scalar component (``u_x``, ``b_z``,
    ``rho``, ``omega_y``, ...) in ``double``, ``single`` or ``half`` precision,
    on the simulation grid or on a coarser one;
``slices/``, ``spectra/``, ``diagnostics/``
    small files written by rank 0 (``write_small``) in the same format.

Every method that touches the state is collective, handles one variable or
component at a time, and costs O(local data); nothing is gathered on one rank
except the small files.  The cast to the output precision happens on the device,
before the copy to the host.

Formats.  With ``io.format: hdf5`` (the default when h5py imports) a field file
is one HDF5 file.  With an MPI build of h5py all ranks write their z slabs into
it (collective file and dataset creation).  With serial h5py every rank writes
``fields_NNNNNN.parts/part.RRRRR.h5`` and rank 0 writes ``fields_NNNNNN.h5``
holding a virtual dataset per component that stitches the parts, so a reader
sees one ``(Nz, Ny, Nx)`` dataset either way.  ``npy`` writes
``fields_NNNNNN/<name>.RRRRR.npy`` per rank and a ``meta.json`` (last, so a
directory that has it is complete).  ``load_fields`` reads both.

Coarsening.  The output on a grid of ``N_out`` points per axis is the exact
spectral low-pass of the field, sampled on that grid: the modes
``|n_i| <= (N_out_i - 1) // 2`` that the solver also retains are kept (coarse
Nyquist modes are zero) and inverse-transformed on the coarse grid.  The ky rows of the retained spectrum are redistributed
from their fine owners to coarse owners with one all-to-all, and the coarse
transform runs on ``min(ranks, Nz_out, Ny_out)`` ranks (a sub-communicator;
the others take part in the exchange and then have empty slabs).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .comm import Comm
from .decomp import SlabDecomposition
from .fft import SlabFFT
from .grid import SlabGrid3D
from .io import _json_ready, field_names, get_h5py, h5_attr, resolve_format, write_arrays

FIELDS_FORMAT = "phrike-fields-1"
XYZ = "xyz"
HALF_MAX = 65504.0

# precision name -> (torch dtype, numpy dtype)
_PRECISIONS = {
    "double": (torch.float64, np.float64), "float64": (torch.float64, np.float64),
    "single": (torch.float32, np.float32), "float32": (torch.float32, np.float32),
    "half": (torch.float16, np.float16), "float16": (torch.float16, np.float16),
}
_SCALARS = ("rho", "p", "E")
_VECTORS = ("u", "b", "B", "m", "omega", "j")
_SYSTEM_VARIABLES = {
    "ins": ["u", "omega"],
    "imhd": ["u", "b", "omega", "j"],
    "euler": ["rho", "u", "p", "m", "E", "omega"],
    "mhd": ["rho", "u", "p", "B", "m", "E", "omega", "j"],
}
_DEFAULT_VARIABLES = {"ins": ["u"], "imhd": ["u", "b"], "euler": ["rho", "u", "p"], "mhd": ["rho", "u", "p", "B"]}
_NOTES = ("Physical-space values of the retained (2/3-rule) modes on a uniform periodic grid; "
          "nodes at i*L/N, the periodic image is not repeated.  Code units; magnetic fields carry no "
          "4*pi or mu0.  omega = curl u and j = curl b (or B) are spectral curls.")


# --- configuration --------------------------------------------------------------


def parse_precision(name: Any) -> str:
    key = str(name or "single").lower()
    if key not in _PRECISIONS:
        raise ValueError(f"io.fields.precision must be 'double', 'single' or 'half', got {name!r}")
    return {"float64": "double", "float32": "single", "float16": "half"}.get(key, key)


def parse_coarsen(spec: Any, shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
    """Output grid ``(Nx_out, Ny_out, Nz_out)`` for ``io.fields.coarsen`` (``None``: full resolution).

    ``spec`` is ``None``, an int (the same size on every axis), a sequence
    ``[Nx_out, Ny_out, Nz_out]`` or ``{factor: f}`` (every size divided by ``f``).
    Each size must be an integer between 4 and the simulation's size.
    """
    if spec is None:
        return None
    Nx, Ny, Nz = (int(n) for n in shape)
    if isinstance(spec, dict):
        if set(spec) != {"factor"}:
            raise ValueError(f"io.fields.coarsen as a mapping takes only 'factor', got {sorted(spec)}")
        factor = spec["factor"]
        if isinstance(factor, bool) or int(factor) != factor or int(factor) < 1:
            raise ValueError(f"io.fields.coarsen.factor must be a positive integer, got {factor!r}")
        factor = int(factor)
        if any(n % factor for n in (Nx, Ny, Nz)):
            raise ValueError(f"io.fields.coarsen.factor {factor} does not divide the grid {Nx}x{Ny}x{Nz}")
        out = (Nx // factor, Ny // factor, Nz // factor)
    elif isinstance(spec, (list, tuple)):
        if len(spec) != 3:
            raise ValueError("io.fields.coarsen as a list is [Nx_out, Ny_out, Nz_out]")
        out = tuple(spec)
    else:
        out = (spec, spec, spec)
    for value in out:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"io.fields.coarsen sizes must be integers, got {value!r}")
    out = tuple(int(v) for v in out)
    for axis, n_out, n in zip("xyz", out, (Nx, Ny, Nz)):
        if n_out < 4 or n_out > n:
            raise ValueError(f"io.fields.coarsen N{axis}_out={n_out} must lie between 4 and the "
                             f"simulation's N{axis}={n}")
    return out


def parse_variables(spec: Any, system: str) -> List[str]:
    """Output variable names (vectors stay whole: ``u`` means ``u_x, u_y, u_z``)."""
    allowed = _SYSTEM_VARIABLES[system]
    if spec is None or spec == "default":
        return list(_DEFAULT_VARIABLES[system])
    names = [spec] if isinstance(spec, str) else list(spec)
    for name in names:
        if name not in allowed:
            raise ValueError(f"io.fields.variables: {name!r} is not available for the {system!r} system "
                             f"(choose from {allowed})")
    return list(dict.fromkeys(names))


def dataset_names(variable: str) -> List[str]:
    return [f"{variable}_{c}" for c in XYZ] if variable in _VECTORS else [variable]


# --- spectral coarsening ----------------------------------------------------------


class _Layout:
    """Global shape of an output grid and the z slab of each rank (empty for idle ranks)."""

    def __init__(self, shape: Tuple[int, int, int], z_counts: Sequence[int], rank: int):
        self.shape = tuple(int(n) for n in shape)  # (Nz, Ny, Nx)
        self.z_counts = [int(c) for c in z_counts]
        self.z_starts = [int(v) for v in np.concatenate(([0], np.cumsum(self.z_counts)[:-1]))]
        self.z0, self.nz = self.z_starts[rank], self.z_counts[rank]


class SpectralCoarsener:
    """Inverse transform of retained modes on a coarser grid, without gathering anything.

    ``to_physical(S)`` takes a rank's retained spectrum of one field,
    ``(1, Kz, nky, Kx)`` divided by N as the solvers store it, and returns this
    rank's z slab ``(nz_out, Ny_out, Nx_out)`` of the field sampled on the
    coarse grid after the exact low-pass described in the module docstring.
    Collective over the grid's communicator.
    """

    def __init__(self, grid: SlabGrid3D, n_out: Tuple[int, int, int]):
        self.grid = grid
        comm, d = grid.comm, grid.decomp
        self.n_out = tuple(int(n) for n in n_out)
        Nxo, Nyo, Nzo = self.n_out
        # Kept modes: those of the coarse grid that the solver also evolves.
        self.mx, self.my, self.mz = (min((n - 1) // 2, c) for n, c in ((Nxo, d.cx), (Nyo, d.cy), (Nzo, d.cz)))
        ranks = min(comm.size, Nzo, Nyo)
        self.comm = comm
        self.sub = comm.sub(ranks)
        self.active = self.sub is not None
        # Idle ranks only need the decomposition's global counts.
        self.cd = SlabDecomposition(Nxo, Nyo, Nzo, ranks, comm.rank if self.active else 0, full=True)
        counts = list(self.cd.z_counts) + [0] * (comm.size - ranks)
        self.layout = _Layout((Nzo, Nyo, Nxo), counts, comm.rank)
        self.fft = None
        if self.active:
            self.fft = SlabFFT(self.cd, self.sub, grid.device, grid.rdtype, batch=1, timers=grid.timers,
                               overlap=False)
        self._plan()

    def _plan(self) -> None:
        """Which rows of whom go to whom: every rank computes the same table, no messages needed."""
        d, cd, me = self.grid.decomp, self.cd, self.comm.rank
        P = self.comm.size
        Nzo, Nyo = self.n_out[2], self.n_out[1]
        modes_y = d.modes_y()
        rows = np.nonzero(np.abs(modes_y) <= self.my)[0]  # kept rows, global compact order
        coarse_row = modes_y[rows] % Nyo  # FFT index on the coarse grid
        owner = np.searchsorted(cd.ky_starts, coarse_row, side="right") - 1
        source = np.searchsorted(d.ky_starts, rows, side="right") - 1
        table = np.zeros((P, P), dtype=np.int64)
        np.add.at(table, (source, owner), 1)
        self.scounts, self.rcounts = table[me].tolist(), table[:, me].tolist()
        mine = np.nonzero(source == me)[0]
        send_order = mine[np.argsort(owner[mine], kind="stable")]  # by destination, rows ascending
        recv_order = np.nonzero(owner == me)[0]  # by source rank, rows ascending
        device = self.grid.device
        self._send_rows = torch.as_tensor(rows[send_order] - d.ky0, dtype=torch.int64, device=device)
        self._dest_rows = torch.as_tensor(coarse_row[recv_order] - cd.ky0 if self.active else [],
                                          dtype=torch.int64, device=device)
        modes_z = d.modes_z()
        zsel = np.nonzero(np.abs(modes_z) <= self.mz)[0]
        self._zsel = torch.as_tensor(zsel, dtype=torch.int64, device=device)
        self._zdest = torch.as_tensor(modes_z[zsel] % Nzo, dtype=torch.int64, device=device)
        self.nxs = self.mx + 1
        self.nsend, self.nrecv = int(sum(self.scounts)), int(sum(self.rcounts))
        self.unit = 2 * len(zsel) * self.nxs  # reals per row: one ky row of the kept (kz, kx) block

    def to_physical(self, S: torch.Tensor) -> torch.Tensor:
        g = self.grid
        nzs = int(self._zsel.numel())
        # Send buffers get one spare element: an empty tensor has no address to hand to MPI.
        send = torch.empty(max(self.nsend, 1) * nzs * self.nxs, dtype=g.cdtype, device=g.device)
        if self.nsend:
            block = S[0].index_select(1, self._send_rows).index_select(0, self._zsel)[:, :, :self.nxs]
            send[:self.nsend * nzs * self.nxs].view(self.nsend, nzs, self.nxs).copy_(block.permute(1, 0, 2))
            del block
        recv = torch.empty(max(self.nrecv, 1) * nzs * self.nxs, dtype=g.cdtype, device=g.device)
        self.comm.exchange(torch.view_as_real(send).reshape(-1), self.scounts,
                           torch.view_as_real(recv).reshape(-1), self.rcounts, self.unit)
        del send
        if not self.active:
            return torch.empty((0,) + self.layout.shape[1:], dtype=g.rdtype, device=g.device)
        cd = self.cd
        Sc = torch.zeros((1, cd.Nz, cd.nky, cd.Kx), dtype=g.cdtype, device=g.device)
        if self.nrecv:
            block = recv[:self.nrecv * nzs * self.nxs].view(self.nrecv, nzs, self.nxs)
            Sc[0, self._zdest[:, None], self._dest_rows[None, :], :self.nxs] = block.permute(1, 0, 2)
        del recv
        return self.fft.inverse(Sc)[0]

    def release(self) -> None:
        if self.fft is not None:
            self.fft.release_buffers()


# --- field files ------------------------------------------------------------------


class _NpyFields:
    """``fields_NNNNNN/<name>.RRRRR.npy`` per rank and a ``meta.json``."""

    def __init__(self, stem: str, comm: Comm, layout: _Layout, dtype):
        self.dir, self.comm, self.layout = stem, comm, layout
        if comm.is_root:
            shutil.rmtree(stem, ignore_errors=True)
            os.makedirs(stem)
        comm.barrier()
        self.names: List[str] = []

    def write(self, name: str, array: np.ndarray) -> None:
        self.names.append(name)
        if self.layout.nz:
            target = os.path.join(self.dir, f"{name}.{self.comm.rank:05d}.npy")
            with open(target + ".tmp", "wb") as handle:
                np.save(handle, array)
            os.replace(target + ".tmp", target)

    def close(self, attrs: Dict[str, Any]) -> str:
        self.comm.barrier()
        if self.comm.is_root:
            lay = self.layout
            meta = dict(attrs, format=FIELDS_FORMAT, datasets=self.names, shape=list(lay.shape),
                        z_starts=lay.z_starts, z_counts=lay.z_counts, nprocs=self.comm.size)
            tmp = os.path.join(self.dir, "meta.json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(_json_ready(meta), handle, indent=1, sort_keys=True)
            os.replace(tmp, os.path.join(self.dir, "meta.json"))
        self.comm.barrier()  # complete on return, on every rank
        return self.dir


class _H5Direct:
    """One HDF5 file: written by one rank, or by all of them through MPI-IO."""

    def __init__(self, stem: str, comm: Comm, layout: _Layout, dtype, parallel: bool):
        h5py = get_h5py()
        self.path, self.tmp = stem + ".h5", stem + ".tmp.h5"
        self.comm, self.layout, self.dtype = comm, layout, dtype
        if parallel:
            self.file = h5py.File(self.tmp, "w", driver="mpio", comm=comm.mpi)
        else:
            self.file = h5py.File(self.tmp, "w")

    def write(self, name: str, array: np.ndarray) -> None:
        lay = self.layout
        dset = self.file.create_dataset(name, shape=lay.shape, dtype=self.dtype)  # collective
        if lay.nz:  # independent writes: a rank with no slab has nothing to do
            dset[lay.z0:lay.z0 + lay.nz] = array

    def close(self, attrs: Dict[str, Any]) -> str:
        for key, value in attrs.items():
            self.file.attrs[key] = h5_attr(value)
        self.file.close()
        self.comm.barrier()
        if self.comm.is_root:
            os.replace(self.tmp, self.path)
        self.comm.barrier()  # the file is complete on return, on every rank
        return self.path


class _H5Parts:
    """A part file per rank and a main file of virtual datasets over them (serial h5py)."""

    def __init__(self, stem: str, comm: Comm, layout: _Layout, dtype):
        self.stem, self.comm, self.layout, self.dtype = stem, comm, layout, dtype
        self.parts = stem + ".parts"
        if comm.is_root:
            shutil.rmtree(self.parts, ignore_errors=True)
            os.makedirs(self.parts)
        comm.barrier()
        self.file = None
        self.names: List[str] = []

    def _part(self, rank: int) -> str:
        return os.path.join(self.parts, f"part.{rank:05d}.h5")

    def write(self, name: str, array: np.ndarray) -> None:
        self.names.append(name)
        if not self.layout.nz:
            return
        if self.file is None:
            self.file = get_h5py().File(self._part(self.comm.rank) + ".tmp", "w")
        self.file.create_dataset(name, data=array)

    def close(self, attrs: Dict[str, Any]) -> str:
        if self.file is not None:
            self.file.close()
            os.replace(self._part(self.comm.rank) + ".tmp", self._part(self.comm.rank))
        self.comm.barrier()
        path = self.stem + ".h5"
        if self.comm.is_root:
            h5py = get_h5py()
            lay = self.layout
            Nz, Ny, Nx = lay.shape
            rel = os.path.basename(self.parts)  # sources are found relative to the main file
            with h5py.File(self.stem + ".tmp.h5", "w") as handle:
                for name in self.names:
                    virtual = h5py.VirtualLayout(shape=lay.shape, dtype=self.dtype)
                    for rank, (z0, nz) in enumerate(zip(lay.z_starts, lay.z_counts)):
                        if nz:
                            source = h5py.VirtualSource(os.path.join(rel, f"part.{rank:05d}.h5"), name,
                                                        shape=(nz, Ny, Nx))
                            virtual[z0:z0 + nz] = source
                    handle.create_virtual_dataset(name, virtual, fillvalue=0)
                for key, value in attrs.items():
                    handle.attrs[key] = h5_attr(value)
            os.replace(self.stem + ".tmp.h5", path)
        self.comm.barrier()  # the main file is complete on return, on every rank
        return path


# --- XDMF ---------------------------------------------------------------------------


_XMF_HEADER = '<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'


def _xmf_text(root: ET.Element) -> str:
    if hasattr(ET, "indent"):
        ET.indent(root)
    return _XMF_HEADER + ET.tostring(root, encoding="unicode") + "\n"


def xdmf_grid(h5name: str, label: str, shape: Tuple[int, int, int], spacing: Tuple[float, float, float],
              t: float, names: Sequence[str], width: int) -> ET.Element:
    """Uniform ``3DCoRectMesh`` grid over the datasets of ``h5name``.

    Every dataset is a scalar attribute; each complete ``_x, _y, _z`` triple of a
    vector variable also becomes a vector attribute (``JOIN``, components x, y, z).
    ``spacing`` is ``(dx, dy, dz)``; the origin is zero.  ORIGIN_DXDYDZ is written
    in the order of ``Dimensions`` (Z Y X), which is how ParaView's Xdmf readers
    map it onto the x, y, z axes.
    """
    dims = " ".join(str(n) for n in shape)  # XDMF lists the slowest axis first: Z Y X
    grid = ET.Element("Grid", Name=label, GridType="Uniform")
    ET.SubElement(grid, "Time", Value=repr(float(t)))
    ET.SubElement(grid, "Topology", TopologyType="3DCoRectMesh", Dimensions=dims)
    geometry = ET.SubElement(grid, "Geometry", GeometryType="ORIGIN_DXDYDZ")
    ET.SubElement(geometry, "DataItem", Format="XML", Dimensions="3").text = "0 0 0"
    ET.SubElement(geometry, "DataItem", Format="XML", Dimensions="3").text = " ".join(
        repr(float(v)) for v in tuple(spacing)[::-1])  # dz dy dx

    def item(parent: ET.Element, name: str) -> None:
        ET.SubElement(parent, "DataItem", Format="HDF", NumberType="Float", Precision=str(width),
                      Dimensions=dims).text = f"{h5name}:/{name}"

    for name in names:
        attribute = ET.SubElement(grid, "Attribute", Name=name, AttributeType="Scalar", Center="Node")
        item(attribute, name)
    for variable in _VECTORS:
        parts = dataset_names(variable)
        if all(p in names for p in parts):
            attribute = ET.SubElement(grid, "Attribute", Name=variable, AttributeType="Vector", Center="Node")
            join = ET.SubElement(attribute, "DataItem", ItemType="Function", Function="JOIN($0, $1, $2)",
                                 Dimensions=dims + " 3")
            for part in parts:
                item(join, part)
    return grid


def write_xdmf(path: str, grid: ET.Element) -> None:
    root = ET.Element("Xdmf", Version="3.0")
    ET.SubElement(root, "Domain").append(grid)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(_xmf_text(root))
    os.replace(path + ".tmp", path)


_INDEXED = re.compile(r"fields_(\d{6})\.xmf$")


_MANIFEST = "series.json"


def _series_indices(fields_dir: str, upto: Optional[int] = None) -> List[int]:
    """Indices of the outputs in the series: the manifest's, else every ``.xmf`` with its ``.h5``."""
    try:
        with open(os.path.join(fields_dir, _MANIFEST), "r", encoding="utf-8") as handle:
            indices = [int(i) for i in json.load(handle)]
    except (OSError, ValueError):
        indices = []
        for path in glob.glob(os.path.join(fields_dir, "fields_??????.xmf")):
            match = _INDEXED.search(os.path.basename(path))
            if match:
                indices.append(int(match.group(1)))
    return sorted({i for i in indices if upto is None or i <= upto})


def _write_manifest(fields_dir: str, indices: List[int]) -> None:
    tmp = os.path.join(fields_dir, _MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(sorted(set(int(i) for i in indices)), handle)
    os.replace(tmp, os.path.join(fields_dir, _MANIFEST))


def write_series(fields_dir: str, upto: Optional[int] = None,
                 indices: Optional[List[int]] = None) -> Optional[str]:
    """(Re)write ``fields.xmf``, a temporal collection of the per-output ``.xmf`` files.

    The outputs are those listed in ``series.json`` (kept by the writer: an
    index is added when it is written and a restart truncates the list at the
    restart point), so frames of an abandoned continuation never come back,
    whatever the field cadence of the restarted run.  Only frames whose
    ``.xmf`` and ``.h5`` exist are included.
    """
    if indices is None:
        indices = _series_indices(fields_dir, upto)
    entries = []
    for i in indices:
        path = os.path.join(fields_dir, f"fields_{int(i):06d}.xmf")
        if (upto is None or i <= upto) and os.path.exists(path) and os.path.exists(path[:-4] + ".h5"):
            entries.append((int(i), path))
    if not entries:
        return None
    root = ET.Element("Xdmf", Version="3.0")
    series = ET.SubElement(ET.SubElement(root, "Domain"), "Grid", Name="fields", GridType="Collection",
                           CollectionType="Temporal")
    for _, path in sorted(entries):
        try:
            grid = ET.parse(path).getroot().find("Domain/Grid")
        except ET.ParseError:
            continue
        if grid is not None:
            series.append(grid)
    target = os.path.join(fields_dir, "fields.xmf")
    with open(target + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(_xmf_text(root))
    os.replace(target + ".tmp", target)
    return target


# --- the writer -----------------------------------------------------------------------


def _code_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                              text=True, timeout=5)
        return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class OutputWriter:
    """Writes the analysis output of one run; ``config`` is the whole run configuration.

    Construction is collective (it may split the communicator).  See the module
    docstring for what is written and ``docs/parallel.rst`` for the options.
    """

    def __init__(self, grid: SlabGrid3D, comm: Comm, system: str, config: Dict[str, Any],
                 outdir: Optional[str] = None):
        self.grid, self.comm, self.system = grid, comm, system
        self.compressible = system in ("euler", "mhd")
        io = (config or {}).get("io") or {}
        self.outdir = os.path.abspath(outdir or io.get("outdir") or "outputs")
        self.format = resolve_format(io.get("format", "auto"))
        block = io.get("fields") or {}
        if isinstance(block, bool):
            block = {"enabled": block}
        # A fields block that is given is on unless it says otherwise (``fields: false`` is off).
        self.fields_enabled = bool(block.get("enabled", bool(block)))
        self.fields_every = max(1, int(block.get("every", 1)))
        self.coarsener: Optional[SpectralCoarsener] = None
        self.commit = "unknown"
        if self.fields_enabled:
            self.precision = parse_precision(block.get("precision", "single"))
            self.variables = parse_variables(block.get("variables", "default"), system)
            self.n_out = parse_coarsen(block.get("coarsen"), (grid.Nx, grid.Ny, grid.Nz))
            if self.n_out is not None:
                self.coarsener = SpectralCoarsener(grid, self.n_out)
            self.commit = comm.bcast(_code_commit() if comm.is_root else None)
        self._gamma: Optional[float] = None
        self._warned_half = False
        self._parallel_h5 = False
        if self.format == "hdf5" and comm.mpi is not None and comm.size > 1:
            self._parallel_h5 = bool(get_h5py().get_config().mpi)

    # --- schedule -----------------------------------------------------------

    def fields_due(self, index: int) -> bool:
        return self.fields_enabled and index % self.fields_every == 0

    @property
    def fields_dir(self) -> str:
        return os.path.join(self.outdir, "fields")

    # --- small files --------------------------------------------------------

    def write_small(self, stem: str, arrays: Dict[str, Any], attrs: Optional[Dict[str, Any]] = None
                    ) -> Optional[str]:
        """Rank 0 writes ``stem.h5`` or ``stem.npz`` (call on every rank; others return ``None``)."""
        if not self.comm.is_root:
            return None
        return write_arrays(stem, arrays, attrs, self.format)

    def write_spectra(self, stepper, index: int, t: float, step: int) -> None:
        """Shell-summed spectra (collective)."""
        spectra = stepper.spectra()
        self.write_small(os.path.join(self.outdir, "spectra", f"spectra_{index:06d}"), spectra,
                         {"t": float(t), "step": int(step)})

    def write_slice(self, stepper, index: int, t: float, step: int) -> None:
        """The z mid-plane of the state variables (collective).

        A compressible state is resident in physical space, so its plane is
        cut out.  An incompressible one is not: its plane is summed over kz
        from the spectrum (a vector-matrix product per component), the plane's
        ky rows are gathered on rank 0 and transformed there, so no
        three-dimensional field is built.
        """
        g, d = self.grid, self.grid.decomp
        z_index = d.Nz // 2
        names = field_names(self.system)
        if self.compressible:
            U = stepper.physical()
            local = (U[:, z_index - d.z0].detach().cpu().numpy() if d.z0 <= z_index < d.z0 + d.nz else None)
            pieces = self.comm.gather(local)
            planes = None
            if self.comm.is_root:
                planes = next(p for p in pieces if p is not None)
        else:
            phase = torch.as_tensor(np.exp(2j * np.pi * d.modes_z() * z_index / d.Nz), dtype=g.cdtype,
                                    device=g.device)
            rows = torch.stack([torch.einsum("zyx,z->yx", stepper.uhat[c], phase)
                                for c in range(stepper.uhat.shape[0])])
            pieces = self.comm.gather(rows.cpu().numpy())
            planes = None
            if self.comm.is_root:
                spectrum = np.zeros((len(names), d.Ny, d.Nx // 2 + 1), dtype=np.complex128)
                spectrum[:, d.modes_y() % d.Ny, :d.Kx] = np.concatenate(pieces, axis=1)
                planes = np.fft.irfft2(spectrum, s=(d.Ny, d.Nx), norm="forward").astype(
                    np.float32 if g.rdtype == torch.float32 else np.float64)
        if self.comm.is_root:
            arrays = {"x": g._x, "y": g._y, **{name: planes[i] for i, name in enumerate(names)}}
            self.write_small(os.path.join(self.outdir, "slices", f"slice_{index:06d}"), arrays,
                             {"t": float(t), "step": int(step), "z_index": z_index, "z": z_index * g.dz})

    # --- field files --------------------------------------------------------

    def _layout(self) -> _Layout:
        if self.coarsener is not None:
            return self.coarsener.layout
        d = self.grid.decomp
        return _Layout((d.Nz, d.Ny, d.Nx), d.z_counts, self.comm.rank)

    def _curl(self, S: torch.Tensor, c: int) -> torch.Tensor:
        """Component ``c`` of the curl of the spectral vector ``S`` (3, Kz, nky, Kx), shape (1, Kz, nky, Kx)."""
        g = self.grid
        k = (g.kx3, g.ky3, g.kz3)
        a, b = (c + 1) % 3, (c + 2) % 3
        return (1j * (k[a] * S[b] - k[b] * S[a]))[None]

    def _pressure(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[0]
        kinetic = U[1] * U[1]
        kinetic += U[2] * U[2]
        kinetic += U[3] * U[3]
        kinetic /= rho
        kinetic *= 0.5
        if self.system == "mhd":
            magnetic = U[5] * U[5]
            magnetic += U[6] * U[6]
            magnetic += U[7] * U[7]
            magnetic *= 0.5
            kinetic += magnetic
        return (U[4] - kinetic).mul_(self._gamma - 1.0)

    def _components(self, st, variable: str) -> Iterator[Tuple[str, Optional[Callable], Optional[Callable]]]:
        """``(dataset, spectrum, physical)`` per component: closures building a spectral or physical array.

        Either may be ``None``; full resolution prefers ``physical``, coarsening ``spectrum``.
        """
        g, uh = self.grid, st.uhat
        names = dataset_names(variable)
        if not self.compressible:
            off = {"u": 0, "b": 3, "omega": 0, "j": 3}[variable]
            for c, name in enumerate(names):
                if variable in ("u", "b"):
                    yield name, (lambda c=c: uh[off + c:off + c + 1]), None
                else:
                    yield name, (lambda c=c: self._curl(uh[off:off + 3], c)), None
            return
        U = st.physical()
        cache: Dict[str, torch.Tensor] = {}
        if variable in ("rho", "E", "m", "B"):
            base = {"rho": 0, "E": 4, "m": 1, "B": 5}[variable]
            for c, name in enumerate(names):
                yield name, (lambda i=base + c: uh[i:i + 1]), (lambda i=base + c: U[i])
        elif variable == "u":
            for c, name in enumerate(names):
                yield name, None, (lambda c=c: U[1 + c] / U[0])
        elif variable == "p":
            yield "p", None, (lambda: self._pressure(U))
        else:  # omega, j
            def vector() -> torch.Tensor:
                if variable == "j":
                    return uh[5:8]
                if "u" not in cache:
                    cache["u"] = torch.empty((3,) + g.decomp.spectral_shape, dtype=g.cdtype, device=g.device)
                    for c in range(3):
                        g.forward((U[1 + c] / U[0])[None], normalized=True, out=cache["u"][c:c + 1])
                return cache["u"]

            for c, name in enumerate(names):
                yield name, (lambda c=c: self._curl(vector(), c)), None

    def _realize(self, spectrum: Optional[Callable], physical: Optional[Callable]) -> torch.Tensor:
        """This rank's slab of one output component, in the solver's precision."""
        g = self.grid
        if self.coarsener is None:
            if physical is not None:
                return physical()
            return g.inverse(spectrum())[0]
        S = spectrum() if spectrum is not None else g.forward(physical()[None], normalized=True)
        return self.coarsener.to_physical(S)

    def _attrs(self, t: float, step: int, layout: _Layout) -> Dict[str, Any]:
        g = self.grid
        Nz, Ny, Nx = layout.shape
        attrs = {
            "t": float(t), "step": int(step), "format": FIELDS_FORMAT, "system": self.system,
            "Nx": Nx, "Ny": Ny, "Nz": Nz, "Lx": g.Lx, "Ly": g.Ly, "Lz": g.Lz,
            "dx": g.Lx / Nx, "dy": g.Ly / Ny, "dz": g.Lz / Nz,
            "sim_Nx": g.Nx, "sim_Ny": g.Ny, "sim_Nz": g.Nz,
            "precision": self.precision, "coarsen": "none" if self.n_out is None else list(self.n_out),
            "spectral_lowpass": self.n_out is not None,
            "variables": list(self.variables), "ranks": self.comm.size,
            "code_commit": self.commit, "notes": _NOTES,
        }
        if self.system in ("euler", "mhd"):
            attrs["gamma"] = float(self._gamma)
        return attrs

    def write_fields(self, stepper, index: int, t: float, step: int) -> str:
        """Write the physical-space fields of this output event (collective); returns the path."""
        self._gamma = getattr(stepper, "gamma", None)
        tdtype, ndtype = _PRECISIONS[self.precision]
        layout = self._layout()
        stem = os.path.join(self.fields_dir, f"fields_{index:06d}")
        if self.comm.is_root:
            os.makedirs(self.fields_dir, exist_ok=True)
            # A rewrite of this index (restart from an older snapshot) must not leave the old
            # main file or sidecar behind if it is interrupted: the series skips a missing index.
            for stale in (stem + ".h5", stem + ".xmf", os.path.join(stem, "meta.json")):
                if os.path.exists(stale):
                    os.remove(stale)
        self.comm.barrier()
        if self.format == "npy":
            handle = _NpyFields(stem, self.comm, layout, ndtype)
        elif self.comm.size == 1 or self._parallel_h5:
            handle = _H5Direct(stem, self.comm, layout, ndtype, parallel=self.comm.size > 1)
        else:
            handle = _H5Parts(stem, self.comm, layout, ndtype)
        names: List[str] = []
        peaks: List[float] = []
        for variable in self.variables:
            for name, spectrum, physical in self._components(stepper, variable):
                field = self._realize(spectrum, physical)
                if self.precision == "half":
                    peaks.append(max(float(field.max()), -float(field.min())) if field.numel() else 0.0)  # no copy
                array = field.to(tdtype)  # on the device, before the copy to the host
                del field
                handle.write(name, np.ascontiguousarray(array.cpu().numpy()))
                del array
                names.append(name)
        if self.coarsener is not None:
            self.coarsener.release()
        if peaks:
            worst = self.comm.allreduce(peaks, "max")
            bad = [n for n, p in zip(names, worst) if p > HALF_MAX]
            if bad and self.comm.is_root:
                warnings.warn(f"io.fields.precision 'half': {', '.join(bad)} exceed the float16 maximum "
                              f"{HALF_MAX:g} and are stored as inf; use 'single'", RuntimeWarning, stacklevel=2)
        attrs = self._attrs(t, step, layout)
        xdmf = self.format == "hdf5" and self.precision != "half"
        if self.format == "hdf5" and not xdmf:
            attrs["xdmf"] = "not written: HDF5/XDMF cannot describe float16 for ParaView"
            if self.comm.is_root and not self._warned_half:
                self._warned_half = True
                warnings.warn("io.fields.precision 'half': the HDF5 files hold float16, which XDMF/ParaView "
                              "cannot read; no .xmf files are written (use 'single' for ParaView)",
                              RuntimeWarning, stacklevel=2)
        path = handle.close(attrs)
        if xdmf and self.comm.is_root:
            grid = xdmf_grid(os.path.basename(stem) + ".h5", os.path.basename(stem), layout.shape,
                             (attrs["dx"], attrs["dy"], attrs["dz"]), t, names, np.dtype(ndtype).itemsize)
            write_xdmf(stem + ".xmf", grid)
            indices = [i for i in _series_indices(self.fields_dir) if i < index] + [index]
            _write_manifest(self.fields_dir, indices)
            write_series(self.fields_dir, indices=indices)
        return path

    def rebuild_series(self, upto: int) -> None:
        """Rebuild ``fields.xmf`` from the outputs up to index ``upto`` that exist (rank 0; after a restart)."""
        if self.comm.is_root and self.fields_enabled and self.format == "hdf5" and upto >= 0 \
                and os.path.isdir(self.fields_dir):
            indices = _series_indices(self.fields_dir, upto)  # drops an abandoned continuation's frames
            _write_manifest(self.fields_dir, indices)
            write_series(self.fields_dir, indices=indices)


# --- reading --------------------------------------------------------------------------


def _expand(names: Sequence[str], available: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        if name in available:
            out.append(name)
        elif all(p in available for p in dataset_names(name)) and name in _VECTORS:
            out.extend(dataset_names(name))
        else:
            raise KeyError(f"{name!r} is not in this output (it has {sorted(available)})")
    return out


def load_fields(path: str, variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Assemble one field output in this process.

    ``path`` is ``fields_NNNNNN.h5``, ``fields_NNNNNN`` (either format) or the
    ``npy`` directory.  Returns ``{"t", "step", "attrs", "x", "y", "z", "fields"}``
    where ``fields`` maps dataset names (``u_x``, ``rho``, ...) to ``(Nz, Ny, Nx)``
    arrays in the stored precision.  ``variables`` selects datasets or whole
    vectors (``["u", "rho"]``); it keeps the memory to what is asked for.
    """
    if os.path.isdir(path):
        kind, source = "npy", path
    elif path.endswith(".h5") and os.path.exists(path):
        kind, source = "hdf5", path
    elif os.path.exists(path + ".h5"):
        kind, source = "hdf5", path + ".h5"
    else:
        raise FileNotFoundError(f"no field output at {path} (looked for a directory and {path}.h5)")
    fields: Dict[str, np.ndarray] = {}
    if kind == "hdf5":
        h5py = get_h5py()
        if h5py is None:
            raise ImportError("reading .h5 field files needs h5py")
        with h5py.File(source, "r") as handle:
            attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in handle.attrs.items()}
            available = [n for n in handle if isinstance(handle[n], h5py.Dataset)]
            for name in (available if variables is None else _expand(variables, available)):
                fields[name] = handle[name][...]
    else:
        with open(os.path.join(source, "meta.json"), "r", encoding="utf-8") as handle:
            attrs = json.load(handle)
        available = list(attrs["datasets"])
        counts = attrs["z_counts"]
        for name in (available if variables is None else _expand(variables, available)):
            parts = [np.load(os.path.join(source, f"{name}.{r:05d}.npy")) for r, c in enumerate(counts) if c]
            fields[name] = np.concatenate(parts, axis=0)
    Nx, Ny, Nz = int(attrs["Nx"]), int(attrs["Ny"]), int(attrs["Nz"])
    return {"t": float(attrs["t"]), "step": int(attrs["step"]), "attrs": attrs, "fields": fields,
            "x": np.arange(Nx) * float(attrs["dx"]), "y": np.arange(Ny) * float(attrs["dy"]),
            "z": np.arange(Nz) * float(attrs["dz"])}
