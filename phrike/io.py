from __future__ import annotations

import json
import os
import subprocess
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Identity fields written into every new snapshot. Restarts that predate these
# fields warn and continue; a present field that disagrees is an error.
SNAPSHOT_IDENTITY_FIELDS = (
    "dimension",
    "basis",
    "bc",
    "gamma",
    "Lx",
    "Ly",
    "Lz",
)

_git_hash_cache: Dict[str, Any] = {"done": False, "value": None}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            loaded = yaml.safe_load(handle)
        else:
            loaded = json.load(handle)
    if loaded is None:
        raise ValueError(f"Configuration file is empty: {path}")
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration file must contain a mapping: {path}")
    return loaded


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _to_numpy(value: Any) -> Any:
    if "torch" in str(type(value)):
        try:
            return value.detach().cpu().numpy()
        except (AttributeError, RuntimeError, TypeError) as exc:
            warnings.warn(
                f"Could not convert a torch value to NumPy ({exc}); "
                "the original object was left unchanged",
                UserWarning,
                stacklevel=2,
            )
    return value


def _json_ready(value: Any) -> Any:
    """Return a JSON-serializable copy of ``value``."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def grid_dimension(grid: Any) -> int:
    if hasattr(grid, "Nz") and getattr(grid, "Nz", None) is not None:
        return 3
    if hasattr(grid, "Ny") and getattr(grid, "Ny", None) is not None:
        return 2
    return 1


def grid_basis(grid: Any) -> Any:
    """Canonical basis label stored in snapshot metadata."""
    dimension = grid_dimension(grid)
    if dimension == 1:
        name = getattr(grid, "_basis_name", None) or getattr(grid, "basis", "fourier")
        return str(name).lower()
    if dimension == 2:
        return {
            "x": str(getattr(grid, "basis_x", "fourier")).lower(),
            "y": str(getattr(grid, "basis_y", "fourier")).lower(),
        }
    return "fourier"


def grid_boundary_condition(grid: Any) -> Any:
    """Normalized boundary condition, or the configured value when unset."""
    boundary_map = getattr(grid, "_bc_boundary_map", None)
    if boundary_map:
        return _json_ready(boundary_map)
    variable_map = getattr(grid, "_bc_map", None)
    if variable_map:
        return _json_ready(variable_map)
    global_bc = getattr(grid, "_bc_global", None)
    if global_bc:
        return str(global_bc)
    return _json_ready(getattr(grid, "bc", None))


def grid_snapshot_identity(grid: Any, equations: Any) -> Dict[str, Any]:
    """Domain, basis, boundary, and gamma values used to validate a restart."""
    dimension = grid_dimension(grid)
    meta: Dict[str, Any] = {
        "dimension": dimension,
        "gamma": float(equations.gamma),
        "Lx": float(grid.Lx),
        "basis": grid_basis(grid),
        "bc": grid_boundary_condition(grid),
    }
    if dimension >= 2:
        meta["Ly"] = float(grid.Ly)
    if dimension == 3:
        meta["Lz"] = float(grid.Lz)
    if dimension == 1:
        meta["N"] = int(grid.N)
    else:
        meta["Nx"] = int(grid.Nx)
        meta["Ny"] = int(grid.Ny)
        if dimension == 3:
            meta["Nz"] = int(grid.Nz)
    bc_config = getattr(grid, "bc_config", None)
    if bc_config is not None:
        meta["bc_config"] = _json_ready(bc_config)
    return meta


def _phrike_version() -> str:
    try:
        from phrike import __version__ as version
    except (ImportError, AttributeError):
        return "unknown"
    return str(version)


def _git_revision() -> Optional[str]:
    """Return a short git hash, or None when git is unavailable.

    The lookup is cached and never raises. A missing repository, a missing
    git executable, or a slow git command leaves the hash out of the snapshot.
    """
    if _git_hash_cache["done"]:
        return _git_hash_cache["value"]
    _git_hash_cache["done"] = True
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    if revision and revision.isalnum():
        _git_hash_cache["value"] = revision
    return _git_hash_cache["value"]


def snapshot_metadata(
    grid: Any,
    equations: Any,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """JSON-ready snapshot metadata, including version and the resolved config."""
    meta = grid_snapshot_identity(grid, equations)
    meta["phrike_version"] = _phrike_version()
    meta["created"] = datetime.now(timezone.utc).isoformat()
    revision = _git_revision()
    if revision:
        meta["git_hash"] = revision
    if provenance is None:
        provenance = getattr(grid, "_phrike_provenance", None)
    if isinstance(provenance, dict):
        config = provenance.get("config")
        if config is not None:
            meta["config"] = _json_ready(config)
        problem = provenance.get("problem")
        if problem:
            meta["problem"] = str(problem)
    return meta


def _legacy_pickle_warning(checkpoint_path: str) -> None:
    warnings.warn(
        "Loading a legacy snapshot whose metadata is pickled "
        f"({checkpoint_path}). New snapshots store JSON metadata and do "
        "not require pickle. Only load restart files you trust.",
        UserWarning,
        stacklevel=4,
    )


def _is_pickle_load_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "pickle" in message or "object" in message


def _materialize_npz(data: Any) -> Dict[str, Any]:
    """Copy arrays out of an open ``.npz`` file."""
    return {key: np.array(data[key]) for key in data.files}


def _read_checkpoint_arrays(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint arrays, falling back to pickle only for legacy files.

    ``np.load(allow_pickle=False)`` opens a legacy archive successfully.
    NumPy raises later, when the pickled metadata array is read.
    """
    try:
        data = np.load(checkpoint_path, allow_pickle=False)
    except ValueError as exc:
        if not _is_pickle_load_error(exc):
            raise
        _legacy_pickle_warning(checkpoint_path)
        with np.load(checkpoint_path, allow_pickle=True) as legacy:
            return _materialize_npz(legacy)

    try:
        try:
            return _materialize_npz(data)
        except ValueError as exc:
            if not _is_pickle_load_error(exc):
                raise
            _legacy_pickle_warning(checkpoint_path)
            with np.load(checkpoint_path, allow_pickle=True) as legacy:
                return _materialize_npz(legacy)
    finally:
        data.close()


def _decode_meta(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, np.ndarray):
        if raw.shape == () or raw.dtype == object:
            raw = raw.item()
        else:
            raw = raw.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Snapshot metadata is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("Snapshot metadata JSON must be an object")
        return loaded
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unrecognized snapshot metadata type: {type(raw).__name__}")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint file and return the simulation state.

    New snapshots store metadata as a JSON string and are loaded with
    ``allow_pickle=False``. Older snapshots stored a pickled dictionary;
    those are loaded with ``allow_pickle=True`` and a warning.

    Args:
        checkpoint_path: Path to the checkpoint file (.npz)

    Returns:
        Dictionary containing:
        - t: simulation time
        - U: conservative variables
        - meta: snapshot metadata (grid and physics parameters)
        - primitive_vars: primitive variables (rho, u, p, etc.)
        - optional tracer arrays (tracer_x, tracer_y, tracer_z, tracer_mass)
        - optional boundary_reference: the far-field/fixed boundary state
          captured at the start of the original run
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    data = _read_checkpoint_arrays(checkpoint_path)

    result = {
        "t": float(data["t"]),
        "U": data["U"],
        "meta": _decode_meta(data["meta"]) if "meta" in data else {},
    }

    if "x" in data:
        result["x"] = data["x"]
    if "y" in data:
        result["y"] = data["y"]
    if "z" in data:
        result["z"] = data["z"]

    primitive_vars = {}
    for var in ["rho", "u", "ux", "uy", "uz", "p"]:
        if var in data:
            primitive_vars[var] = data[var]
    result["primitive_vars"] = primitive_vars

    if "tracer_x" in data:
        result["tracer_x"] = data["tracer_x"]
    if "tracer_y" in data:
        result["tracer_y"] = data["tracer_y"]
    if "tracer_z" in data:
        result["tracer_z"] = data["tracer_z"]
    if "tracer_mass" in data:
        result["tracer_mass"] = data["tracer_mass"]
    for axis in ("x", "y", "z"):
        key = f"tracer_{axis}_unwrapped"
        if key in data:
            result[key] = data[key]
    if "tracer_lattice_shape" in data:
        result["tracer_lattice_shape"] = data["tracer_lattice_shape"]
    if "boundary_reference" in data:
        result["boundary_reference"] = data["boundary_reference"]

    return result


def _tracer_save_dict(tracers: Any) -> Dict[str, Any]:
    """Build dict of tracer arrays for np.savez (tracer_x, tracer_y, tracer_z, tracer_mass).
    Converts torch tensors to numpy so save works with PyTorch/MPS backend.
    """
    out: Dict[str, Any] = {}
    if tracers is None:
        return out
    out["tracer_x"] = _to_numpy(tracers.x)
    if hasattr(tracers, "y"):
        out["tracer_y"] = _to_numpy(tracers.y)
    if hasattr(tracers, "z"):
        out["tracer_z"] = _to_numpy(tracers.z)
    if hasattr(tracers, "mass"):
        out["tracer_mass"] = _to_numpy(tracers.mass)
    for axis in ("x", "y", "z"):
        name = f"{axis}_unwrapped"
        if hasattr(tracers, name):
            out[f"tracer_{name}"] = _to_numpy(getattr(tracers, name))
    lattice_shape = getattr(tracers, "lattice_shape", None)
    if lattice_shape is not None:
        out["tracer_lattice_shape"] = np.asarray(lattice_shape, dtype=np.int64)
    return out


def save_solution_snapshot(
    outdir: str,
    t: float,
    U: np.ndarray,
    grid: Any,
    equations: Any,
    tag: Optional[str] = None,
    tracers: Optional[Any] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    ts = f"t{t:.6f}"
    if tag:
        ts = f"{ts}_{tag}"
    filename = os.path.join(outdir, f"snapshot_{ts}.npz")
    tracer_data = _tracer_save_dict(tracers)

    is_3d = hasattr(grid, "Nz") and hasattr(grid, "z")
    is_2d = (not is_3d) and hasattr(grid, "Ny") and hasattr(grid, "y")
    is_mhd = hasattr(U, "shape") and U.shape[0] == 8

    meta = snapshot_metadata(grid, equations, provenance=provenance)
    if is_mhd:
        meta["mhd"] = True
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    save_kw: Dict[str, Any] = dict(
        t=t,
        U=_to_numpy(U),
        meta=meta_json,
        created=meta["created"],
        **tracer_data,
    )

    if is_mhd:
        rho, ux, uy, uz, p, Bx, By, Bz = equations.primitive(U)
        try:
            if is_3d:
                div_b = grid.divergence(Bx, By, Bz)
            elif is_2d:
                div_b = grid.divergence(Bx, By)
            else:
                div_b = grid.divergence_x(Bx)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"div(B) was not stored in the snapshot: {exc}",
                UserWarning,
                stacklevel=2,
            )
            div_b = None
        save_kw.update(
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            uz=_to_numpy(uz),
            p=_to_numpy(p),
            Bx=_to_numpy(Bx),
            By=_to_numpy(By),
            Bz=_to_numpy(Bz),
        )
        if div_b is not None:
            save_kw["div_b"] = _to_numpy(div_b)
        if is_3d:
            save_kw.update(x=_to_numpy(grid.x), y=_to_numpy(grid.y), z=_to_numpy(grid.z))
        elif is_2d:
            save_kw.update(x=_to_numpy(grid.x), y=_to_numpy(grid.y))
        else:
            save_kw.update(x=_to_numpy(grid.x))
    elif is_3d:
        rho, ux, uy, uz, p = equations.primitive(U)
        save_kw.update(
            x=_to_numpy(grid.x),
            y=_to_numpy(grid.y),
            z=_to_numpy(grid.z),
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            uz=_to_numpy(uz),
            p=_to_numpy(p),
        )
    elif is_2d:
        rho, ux, uy, p = equations.primitive(U)
        save_kw.update(
            x=_to_numpy(grid.x),
            y=_to_numpy(grid.y),
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            p=_to_numpy(p),
        )
    else:
        rho, u, p, _ = equations.primitive(U)
        save_kw.update(
            x=_to_numpy(grid.x),
            rho=_to_numpy(rho),
            u=_to_numpy(u),
            p=_to_numpy(p),
        )

    boundary_reference = getattr(grid, "_boundary_reference", None)
    if boundary_reference is not None:
        save_kw["boundary_reference"] = _to_numpy(boundary_reference)

    np.savez(filename, **save_kw)
    return filename
