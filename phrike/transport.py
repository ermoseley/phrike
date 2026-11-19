"""Transport coefficients of compressible visco-resistive MHD and their fluxes.

PHRIKE's MHD state is ``[rho, rho u, E, B]`` with the magnetic pressure
``|B|^2 / 2`` (mu0 = 1), so the current is ``J = curl B``.  With the kinematic
shear viscosity ``nu`` (``mu = rho nu``), the kinematic bulk viscosity
``nu_bulk``, the magnetic diffusivity ``eta`` and the heat conductivity
``kappa`` (``T = p / rho``, the gas constant set to one), the conservative
fluxes gain

* momentum: ``-sigma`` with the Newtonian stress
  ``sigma = 2 rho nu D + rho nu_bulk (div u) I`` and the trace-free strain
  ``D = (grad u + grad u^T) / 2 - (div u) I / 3``;
* energy: ``-sigma . u + eta J x B - kappa grad T`` (viscous work, Ohmic
  Poynting flux, heat flux);

and the induction equation gains ``-curl(eta J) = eta (lap B - grad div B)``.
The local heating ``2 rho nu D:D + rho nu_bulk (div u)^2 + eta |J|^2`` is then
what the internal energy receives, and total energy and momentum stay in flux
form.  The factor 1/3 holds in 1D and 2D as well: those solvers evolve all three
velocity components.

The pointwise flux additions are here once, for NumPy and Torch arrays alike,
so that the reference solver and the Torch tiers of the fast paths evaluate
the same expressions; the Triton kernels repeat them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

Array = Any

# (name in the configuration, attribute)
_KEYS = (("viscosity", "nu"), ("bulk_viscosity", "nu_bulk"), ("resistivity", "eta"),
         ("thermal_conductivity", "kappa"))

# ``physics.mhd.viscous_step_bound``: how the viscous time-step bound treats density.
VISCOUS_STEP_BOUNDS = ("density_ratio", "uniform")


def viscous_step_bound(cfg: Optional[Mapping[str, Any]]) -> str:
    """The configured viscous time-step bound (``density_ratio`` unless set to ``uniform``)."""
    value = str((cfg or {}).get("viscous_step_bound", "density_ratio")).lower()
    if value not in VISCOUS_STEP_BOUNDS:
        raise ValueError(f"physics.mhd.viscous_step_bound must be one of {VISCOUS_STEP_BOUNDS}, got {value!r}")
    return value


@dataclass(frozen=True)
class Transport:
    """Constant transport coefficients (all zero: ideal MHD)."""

    nu: float = 0.0
    nu_bulk: float = 0.0
    eta: float = 0.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        for _, name in _KEYS:
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"MHD transport coefficient {name} must be finite and nonnegative, got {value}")
            object.__setattr__(self, name, value)

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "Transport":
        """From an MHD configuration block (``viscosity``, ``bulk_viscosity``,
        ``resistivity`` capped by ``resistivity_max``, ``thermal_conductivity``)."""
        cfg = cfg or {}
        raw = cls(**{name: float(cfg.get(key, 0.0) or 0.0) for key, name in _KEYS})  # validated
        cap = cfg.get("resistivity_max")
        if cap is None:
            return raw
        cap = float(cap)
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError("MHD resistivity_max must be finite and positive")
        return cls(nu=raw.nu, nu_bulk=raw.nu_bulk, eta=min(raw.eta, cap), kappa=raw.kappa)

    # --- what a step needs -------------------------------------------------

    @property
    def shear(self) -> bool:
        return self.nu > 0.0

    @property
    def bulk(self) -> bool:
        return self.nu_bulk > 0.0

    @property
    def viscous(self) -> bool:
        """Velocity gradients are needed (shear or bulk viscosity)."""
        return self.shear or self.bulk

    @property
    def resistive(self) -> bool:
        return self.eta > 0.0

    @property
    def conductive(self) -> bool:
        return self.kappa > 0.0

    @property
    def active(self) -> bool:
        return self.viscous or self.resistive or self.conductive

    @property
    def gradients(self) -> bool:
        """The fluxes need derivative fields in physical space (viscosity or Ohmic flux)."""
        return self.viscous or self.resistive

    def max_diffusivity(self, gamma: float, rho_min: Optional[float] = None,
                        rho_max: Optional[float] = None, commuting: bool = True) -> float:
        """Largest diffusivity of the frozen-coefficient operator.

        At uniform density transverse velocity modes decay at ``nu k^2`` and
        longitudinal ones at ``(4 nu / 3 + nu_bulk) k^2``, B at ``eta k^2`` and
        temperature at ``(gamma - 1) kappa / rho k^2`` (this needs ``rho_min``).
        The 4/3 needs derivatives whose adjoints commute across axes (Fourier
        axes, or one axis): ``commuting=False`` (a mixed Fourier/SBP grid, whose
        boundary closure breaks that) uses ``2 nu + nu_bulk``. That bounds the
        volume strain energy for any discrete gradient G with per-axis norms
        ``k_a^2``: ``sum 2 nu |D|^2 <= 2 nu ||G u||^2`` and, by Cauchy-Schwarz
        across the axes, ``||div u||^2 <= (sum_a k_a^2) ||u||^2``, as is
        ``||G u||^2`` (operator norms, not a pointwise inequality).  With ``rho_max`` as well, the viscous rate is
        multiplied by ``rho_max / rho_min``: the operator on u is
        ``rho^-1 div(rho nu ...)``, whose Rayleigh quotient that ratio bounds at
        any density (sufficient, and pessimistic for smooth density).
        """
        viscous = (4.0 / 3.0 if commuting else 2.0) * self.nu + self.nu_bulk
        if (self.viscous and rho_max is not None) or self.conductive:
            if rho_min is None or not (math.isfinite(rho_min) and rho_min > 0.0):
                raise ValueError(f"the diffusive time-step bound needs a positive minimum density, got {rho_min}")
        if self.viscous and rho_max is not None:
            if not (math.isfinite(rho_max) and rho_max >= rho_min):
                raise ValueError(f"the viscous time-step bound needs a finite maximum density, got {rho_max}")
            viscous *= float(rho_max) / float(rho_min)
        rates = [viscous, self.eta]
        if self.conductive:
            rates.append((float(gamma) - 1.0) * self.kappa / float(rho_min))
        return max(rates)

    def describe(self) -> str:
        parts = [f"{key}={getattr(self, name):g}" for key, name in _KEYS if getattr(self, name) > 0.0]
        return ", ".join(parts) if parts else "ideal"

    def as_dict(self) -> Dict[str, float]:
        return {key: float(getattr(self, name)) for key, name in _KEYS}


def flux_buffer_rows(tr: Transport) -> Tuple[int, int, int, int]:
    """Rows of a stage's 12-row flux buffer that hold the derivative fields.

    Returns ``(rowD, rowT, rowJ, n)``: the trace-free strain (5 rows, with shear
    viscosity), div u (1, bulk viscosity) and the current (3, resistivity) are
    packed from row 0 in that order, so one inverse transform fills all ``n <= 9``
    of them.  Before that, row 8 holds T (conduction) and rows 9-11 the velocity,
    the fields transformed first; row 8 is overwritten only after T is consumed.
    The fused flux pass then reads every cell's inputs before writing its fluxes.
    """
    rowT = 5 if tr.shear else 0
    rowJ = rowT + (1 if tr.bulk else 0)
    return 0, rowT, rowJ, rowJ + (3 if tr.resistive else 0)


def strain_from_gradient(G: Sequence[Sequence[Array]]) -> Tuple[Tuple[Array, ...], Array]:
    """Trace-free strain ``(Dxx, Dyy, Dxy, Dxz, Dyz)`` and ``div u`` from ``G[a][b] = d_a u_b``.

    ``Dzz = -Dxx - Dyy``.  Missing derivatives may be passed as plain zeros.
    """
    theta = G[0][0] + G[1][1] + G[2][2]
    third = theta / 3.0
    D = (G[0][0] - third, G[1][1] - third,
         0.5 * (G[0][1] + G[1][0]), 0.5 * (G[0][2] + G[2][0]), 0.5 * (G[1][2] + G[2][1]))
    return D, theta


def flux_additions(rho: Array, u: Sequence[Array], B: Sequence[Array], tr: Transport,
                   D: Optional[Sequence[Array]] = None, theta: Optional[Array] = None,
                   J: Optional[Sequence[Array]] = None):
    """Additions to the momentum flux (xx, yy, zz, xy, xz, yz) and energy flux (x, y, z).

    ``D`` is the trace-free strain (needed with shear viscosity), ``theta`` the
    divergence of u (bulk viscosity) and ``J`` the current (resistivity).  Terms
    that are absent are returned as ``None``; the heat flux is not included
    (the solvers apply it as ``kappa lap T`` in the energy equation).
    """
    dT = [None] * 6
    dH = [None, None, None]
    if tr.viscous:
        ux, uy, uz = u
        if tr.shear:
            mu2 = 2.0 * tr.nu * rho
            sxx, syy = mu2 * D[0], mu2 * D[1]
            szz = -(sxx + syy)
            sxy, sxz, syz = mu2 * D[2], mu2 * D[3], mu2 * D[4]
            if tr.bulk:
                lam = tr.nu_bulk * rho * theta
                sxx, syy, szz = sxx + lam, syy + lam, szz + lam
        else:  # bulk viscosity alone: an isotropic stress
            lam = tr.nu_bulk * rho * theta
            sxx = syy = szz = lam
            sxy = sxz = syz = 0.0 * lam
        dT = [-sxx, -syy, -szz, -sxy, -sxz, -syz]
        dH = [-(sxx * ux + sxy * uy + sxz * uz), -(sxy * ux + syy * uy + syz * uz),
              -(sxz * ux + syz * uy + szz * uz)]
    if tr.resistive:
        Jx, Jy, Jz = J
        Bx, By, Bz = B
        ohm = [tr.eta * (Jy * Bz - Jz * By), tr.eta * (Jz * Bx - Jx * Bz), tr.eta * (Jx * By - Jy * Bx)]
        dH = [o if h is None else h + o for h, o in zip(dH, ohm)]
    return dT, dH


def heating(rho: Array, tr: Transport, D: Optional[Sequence[Array]] = None,
            theta: Optional[Array] = None, J: Optional[Sequence[Array]] = None):
    """Viscous and Ohmic heating rates per volume (``None`` where the coefficient is zero)."""
    viscous = ohmic = None
    if tr.shear:
        Dxx, Dyy, Dxy, Dxz, Dyz = D
        Dzz = -(Dxx + Dyy)
        viscous = 2.0 * tr.nu * rho * (Dxx * Dxx + Dyy * Dyy + Dzz * Dzz
                                       + 2.0 * (Dxy * Dxy + Dxz * Dxz + Dyz * Dyz))
    if tr.bulk:
        term = tr.nu_bulk * rho * theta * theta
        viscous = term if viscous is None else viscous + term
    if tr.resistive:
        ohmic = tr.eta * (J[0] * J[0] + J[1] * J[1] + J[2] * J[2])
    return viscous, ohmic
