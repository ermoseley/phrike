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
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore
    if 'torch' in str(type(a)):
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
        "meta": data["meta"].item() if "meta" in data else {}
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
    
    return result


def save_solution_snapshot(
    outdir: str,
    t: float,
    U: np.ndarray,
    grid: Any,
    equations: Any,
    tag: Optional[str] = None,
) -> str:
    ts = f"t{t:.6f}"
    if tag:
        ts = f"{ts}_{tag}"
    filename = os.path.join(outdir, f"snapshot_{ts}.npz")

    # Support 1D, 2D, and 3D grids/equations
    is_3d = hasattr(grid, "Nz") and hasattr(grid, "z")
    is_2d = (not is_3d) and hasattr(grid, "Ny") and hasattr(grid, "y")
    if is_3d:
        rho, ux, uy, uz, p = equations.primitive(U)
        meta: Dict[str, Any] = {"Nx": getattr(grid, 'Nx', None), "Ny": getattr(grid, 'Ny', None), "Nz": getattr(grid, 'Nz', None), "Lx": grid.Lx, "Ly": grid.Ly, "Lz": grid.Lz, "gamma": equations.gamma}
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
        )
    elif is_2d:
        rho, ux, uy, p = equations.primitive(U)
        meta: Dict[str, Any] = {"Nx": getattr(grid, 'Nx', None), "Ny": getattr(grid, 'Ny', None), "Lx": grid.Lx, "Ly": grid.Ly, "gamma": equations.gamma}
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
        )
    return filename


