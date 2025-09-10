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
    rho, u, p, _ = equations.primitive(U)
    np.savez(
        filename,
        t=t,
        U=U,
        x=grid.x,
        rho=rho,
        u=u,
        p=p,
        meta={"N": grid.N, "Lx": grid.Lx, "gamma": equations.gamma},
        created=str(datetime.utcnow()),
    )
    return filename


