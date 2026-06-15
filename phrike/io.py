from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _to_numpy(a: Any) -> Any:
    if "torch" in str(type(a)):
        try:
            return a.detach().cpu().numpy()
        except Exception:
            pass
    return a


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint file and return the simulation state.

    Args:
        checkpoint_path: Path to the checkpoint file (.npz)

    Returns:
        Dictionary containing:
        - t: simulation time
        - U: conservative variables
        - grid_params: grid parameters (N, Lx, etc.)
        - physics_params: physics parameters (gamma, etc.)
        - primitive_vars: primitive variables (rho, u, p, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    data = np.load(checkpoint_path, allow_pickle=True)

    # Extract basic simulation state
    result = {
        "t": float(data["t"]),
        "U": data["U"],
        "meta": data["meta"].item() if "meta" in data else {},
    }

    # Extract grid coordinates
    if "x" in data:
        result["x"] = data["x"]
    if "y" in data:
        result["y"] = data["y"]
    if "z" in data:
        result["z"] = data["z"]

    # Extract primitive variables if available
    primitive_vars = {}
    for var in ["rho", "u", "ux", "uy", "uz", "p"]:
        if var in data:
            primitive_vars[var] = data[var]
    result["primitive_vars"] = primitive_vars

    # Extract tracer data if available
    if "tracer_x" in data:
        result["tracer_x"] = data["tracer_x"]
    if "tracer_y" in data:
        result["tracer_y"] = data["tracer_y"]
    if "tracer_z" in data:
        result["tracer_z"] = data["tracer_z"]
    if "tracer_mass" in data:
        result["tracer_mass"] = data["tracer_mass"]

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
    return out


def save_solution_snapshot(
    outdir: str,
    t: float,
    U: np.ndarray,
    grid: Any,
    equations: Any,
    tag: Optional[str] = None,
    tracers: Optional[Any] = None,
) -> str:
    ts = f"t{t:.6f}"
    if tag:
        ts = f"{ts}_{tag}"
    filename = os.path.join(outdir, f"snapshot_{ts}.npz")
    tracer_data = _tracer_save_dict(tracers)

    # Support 1D, 2D, and 3D grids/equations
    is_3d = hasattr(grid, "Nz") and hasattr(grid, "z")
    is_2d = (not is_3d) and hasattr(grid, "Ny") and hasattr(grid, "y")

    # MHD: 8-component state [rho, mom_x, mom_y, mom_z, E, Bx, By, Bz].
    # Save the magnetic field and a div(B) diagnostic alongside the hydro fields
    # so existing viewers (rho, u*, p) keep working.
    is_mhd = hasattr(U, "shape") and U.shape[0] == 8
    if is_mhd:
        rho, ux, uy, uz, p, Bx, By, Bz = equations.primitive(U)
        try:
            if is_3d:
                div_b = grid.divergence(Bx, By, Bz)
            elif is_2d:
                div_b = grid.divergence(Bx, By)
            else:
                div_b = grid.divergence_x(Bx)
        except Exception:
            div_b = None
        meta = {
            "mhd": True,
            "gamma": equations.gamma,
            "Lx": grid.Lx,
        }
        save_kw = dict(
            t=t,
            U=_to_numpy(U),
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            uz=_to_numpy(uz),
            p=_to_numpy(p),
            Bx=_to_numpy(Bx),
            By=_to_numpy(By),
            Bz=_to_numpy(Bz),
            created=str(datetime.utcnow()),
            **tracer_data,
        )
        if div_b is not None:
            save_kw["div_b"] = _to_numpy(div_b)
        if is_3d:
            meta.update({"Nx": grid.Nx, "Ny": grid.Ny, "Nz": grid.Nz,
                         "Ly": grid.Ly, "Lz": grid.Lz})
            save_kw.update(x=_to_numpy(grid.x), y=_to_numpy(grid.y), z=_to_numpy(grid.z))
        elif is_2d:
            meta.update({"Nx": grid.Nx, "Ny": grid.Ny, "Ly": grid.Ly})
            save_kw.update(x=_to_numpy(grid.x), y=_to_numpy(grid.y))
        else:
            meta.update({"N": grid.N})
            save_kw.update(x=_to_numpy(grid.x))
        np.savez(filename, meta=meta, **save_kw)
        return filename

    if is_3d:
        rho, ux, uy, uz, p = equations.primitive(U)
        meta: Dict[str, Any] = {
            "Nx": getattr(grid, "Nx", None),
            "Ny": getattr(grid, "Ny", None),
            "Nz": getattr(grid, "Nz", None),
            "Lx": grid.Lx,
            "Ly": grid.Ly,
            "Lz": grid.Lz,
            "gamma": equations.gamma,
        }
        np.savez(
            filename,
            t=t,
            U=_to_numpy(U),
            x=_to_numpy(grid.x),
            y=_to_numpy(grid.y),
            z=_to_numpy(grid.z),
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            uz=_to_numpy(uz),
            p=_to_numpy(p),
            meta=meta,
            created=str(datetime.utcnow()),
            **tracer_data,
        )
    elif is_2d:
        rho, ux, uy, p = equations.primitive(U)
        meta: Dict[str, Any] = {
            "Nx": getattr(grid, "Nx", None),
            "Ny": getattr(grid, "Ny", None),
            "Lx": grid.Lx,
            "Ly": grid.Ly,
            "gamma": equations.gamma,
        }
        np.savez(
            filename,
            t=t,
            U=_to_numpy(U),
            x=_to_numpy(grid.x),
            y=_to_numpy(grid.y),
            rho=_to_numpy(rho),
            ux=_to_numpy(ux),
            uy=_to_numpy(uy),
            p=_to_numpy(p),
            meta=meta,
            created=str(datetime.utcnow()),
            **tracer_data,
        )
    else:
        rho, u, p, _ = equations.primitive(U)
        meta = {"N": grid.N, "Lx": grid.Lx, "gamma": equations.gamma}
        np.savez(
            filename,
            t=t,
            U=_to_numpy(U),
            x=_to_numpy(grid.x),
            rho=_to_numpy(rho),
            u=_to_numpy(u),
            p=_to_numpy(p),
            meta=meta,
            created=str(datetime.utcnow()),
            **tracer_data,
        )
    return filename
