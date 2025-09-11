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

    # Support both 1D and 2D grids/equations
    is_2d = hasattr(grid, "Ny") and hasattr(grid, "y")
    if is_2d:
        rho, ux, uy, p = equations.primitive(U)
        meta: Dict[str, Any] = {"Nx": grid.Nx, "Ny": grid.Ny, "Lx": grid.Lx, "Ly": grid.Ly, "gamma": equations.gamma}
        np.savez(
            filename,
            t=t,
            U=U,
            x=grid.x,
            y=grid.y,
            rho=rho,
            ux=ux,
            uy=uy,
            p=p,
            meta=meta,
            created=str(datetime.utcnow()),
        )
    else:
        rho, u, p, _ = equations.primitive(U)
        meta = {"N": grid.N, "Lx": grid.Lx, "gamma": equations.gamma}
        np.savez(
            filename,
            t=t,
            U=U,
            x=grid.x,
            rho=rho,
            u=u,
            p=p,
            meta=meta,
            created=str(datetime.utcnow()),
        )
    return filename


