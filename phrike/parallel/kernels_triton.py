"""Triton kernels of the distributed solvers that ``spectral3d_triton`` does not have (CUDA only).

* ``state_stats``: one pass over a compressible state for the three scalars
  every step needs (values below the positivity floors, a non-finite flag and
  the maximum wave speed).
* ``imhd_products``: the quadratic products of incompressible MHD in physical
  space, with the advective rate for the CFL condition.
* ``imhd_stage``: the spectral right-hand side of incompressible MHD (Leray
  projection included) and one integrating-factor RK stage update.

Each has a Torch counterpart in the stepper that calls it, used when Triton is
unavailable and checked against the kernel by ``python -m phrike.parallel.selftest``.
Reductions write one partial result per kernel block, so no float atomics are
needed.  Python scalars would reach Triton as fp32, so parameters travel in a
small tensor of the state's dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ld(ptr, off, mask):
    """Load complex values (re, im) at element offsets ``off`` of a real view."""
    two = tl.arange(0, 2)
    v = tl.load(ptr + off[:, None] + two[None, :], mask=mask[:, None], other=0.0)
    return tl.split(v)


@triton.jit
def _st(ptr, off, mask, re, im):
    two = tl.arange(0, 2)
    tl.store(ptr + off[:, None] + two[None, :], tl.join(re, im), mask=mask[:, None])


# --- compressible state statistics ---------------------------------------------


@triton.jit
def _reduce_stats(cnt_ptr, bad_ptr, nan_ptr, spd_ptr, pid, mask, rho, p, comp, probe, rho_min, p_min):
    """Per-block partials: floor violations, non-finite state values, non-finite
    wave speeds, and the maximum of the finite wave speeds."""
    below = ((rho < rho_min) & mask).to(tl.int32) + ((p < p_min) & mask).to(tl.int32)
    # x - x is 0 for a finite x and NaN for NaN or Inf.
    nonfinite = (((probe - probe) != 0.0) & mask).to(tl.int32)
    unusable = (((comp - comp) != 0.0) & mask).to(tl.int32)
    usable = mask & ((comp - comp) == 0.0)
    tl.store(cnt_ptr + pid, tl.sum(below, axis=0))
    tl.store(bad_ptr + pid, tl.sum(nonfinite, axis=0))
    tl.store(nan_ptr + pid, tl.sum(unusable, axis=0))
    tl.store(spd_ptr + pid, tl.max(tl.where(usable, comp, 0.0), axis=0))


@triton.jit
def _euler3d_stats_kernel(U_ptr, cnt_ptr, bad_ptr, nan_ptr, spd_ptr, n, P_ptr, BLOCK: tl.constexpr):
    # int32 scalar arguments: multiples of them overflow past 2**31 elements.
    n = n.to(tl.int64)
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gamma = tl.load(P_ptr)
    rho_min = tl.load(P_ptr + 1)
    p_min = tl.load(P_ptr + 2)
    rho = tl.load(U_ptr + offs, mask=mask, other=1.0)
    mx = tl.load(U_ptr + n + offs, mask=mask, other=0.0)
    my = tl.load(U_ptr + 2 * n + offs, mask=mask, other=0.0)
    mz = tl.load(U_ptr + 3 * n + offs, mask=mask, other=0.0)
    E = tl.load(U_ptr + 4 * n + offs, mask=mask, other=1.0)
    ux = mx / rho
    uy = my / rho
    uz = mz / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy + uz * uz))
    comp = tl.maximum(tl.maximum(tl.abs(ux), tl.abs(uy)), tl.abs(uz)) + tl.sqrt(gamma * p / rho)
    probe = rho + mx + my + mz + E
    _reduce_stats(cnt_ptr, bad_ptr, nan_ptr, spd_ptr, pid, mask, rho, p, comp, probe, rho_min, p_min)


@triton.jit
def _mhd3d_stats_kernel(U_ptr, cnt_ptr, bad_ptr, nan_ptr, spd_ptr, n, P_ptr, BLOCK: tl.constexpr):
    # int32 scalar arguments: multiples of them overflow past 2**31 elements.
    n = n.to(tl.int64)
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gamma = tl.load(P_ptr)
    rho_min = tl.load(P_ptr + 1)
    p_min = tl.load(P_ptr + 2)
    rho = tl.load(U_ptr + offs, mask=mask, other=1.0)
    mx = tl.load(U_ptr + n + offs, mask=mask, other=0.0)
    my = tl.load(U_ptr + 2 * n + offs, mask=mask, other=0.0)
    mz = tl.load(U_ptr + 3 * n + offs, mask=mask, other=0.0)
    E = tl.load(U_ptr + 4 * n + offs, mask=mask, other=1.0)
    Bx = tl.load(U_ptr + 5 * n + offs, mask=mask, other=0.0)
    By = tl.load(U_ptr + 6 * n + offs, mask=mask, other=0.0)
    Bz = tl.load(U_ptr + 7 * n + offs, mask=mask, other=0.0)
    ux = mx / rho
    uy = my / rho
    uz = mz / rho
    b2 = Bx * Bx + By * By + Bz * Bz
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy + uz * uz) - 0.5 * b2)
    fast2 = (gamma * p + b2) / rho
    # sqrt(clamp(., min=0)), keeping a NaN argument (NaN > 0 is false, so it is added back).
    fast = tl.sqrt(tl.where(fast2 > 0.0, fast2, 0.0)) + (fast2 - fast2)
    comp = tl.maximum(tl.maximum(tl.abs(ux), tl.abs(uy)), tl.abs(uz)) + fast
    probe = rho + mx + my + mz + E + Bx + By + Bz
    _reduce_stats(cnt_ptr, bad_ptr, nan_ptr, spd_ptr, pid, mask, rho, p, comp, probe, rho_min, p_min)


_STATS_KERNELS = {"euler3d": _euler3d_stats_kernel, "mhd3d": _mhd3d_stats_kernel}


def state_stats(kind, U, gamma, rho_min, p_min, block=1024, num_warps=8):
    """(values below a floor, non-finite state values, non-finite wave speeds,
    max finite wave speed) of a state, as floats."""
    kernel = _STATS_KERNELS[kind]
    U = U.contiguous()
    n = U[0].numel()
    nblocks = triton.cdiv(n, block)
    cnt = torch.empty(nblocks, dtype=torch.int32, device=U.device)
    bad = torch.empty(nblocks, dtype=torch.int32, device=U.device)
    nan = torch.empty(nblocks, dtype=torch.int32, device=U.device)
    spd = torch.empty(nblocks, dtype=U.dtype, device=U.device)
    params = torch.tensor([gamma, rho_min, p_min], dtype=U.dtype, device=U.device)
    kernel[(nblocks,)](U, cnt, bad, nan, spd, n, params, BLOCK=block, num_warps=num_warps)
    out = torch.stack([cnt.sum().to(torch.float64), bad.sum().to(torch.float64),
                       nan.sum().to(torch.float64), spd.max().to(torch.float64)]).tolist()
    return float(out[0]), float(out[1]), float(out[2]), float(out[3])


# --- incompressible MHD: physical-space products -------------------------------
#
# Output order: [Txx - Tzz, Tyy - Tzz, Txy, Txz, Tyz, Axy, Axz, Ayz] with
# T_ij = u_i u_j - b_i b_j and A_ij = u_i b_j - u_j b_i.  The isotropic part of
# T is a pressure gradient and is removed by the Leray projection, so Tzz is
# subtracted from the diagonal and never transformed.  Without a magnetic
# field only the first five fields exist.


@triton.jit
def _imhd_products_kernel(V_ptr, G_ptr, rate_ptr, bad_ptr, n, P_ptr,
                          MHD: tl.constexpr, STATS: tl.constexpr, BLOCK: tl.constexpr):
    # int32 scalar arguments: multiples of them overflow past 2**31 elements.
    n = n.to(tl.int64)
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(P_ptr)
    idy = tl.load(P_ptr + 1)
    idz = tl.load(P_ptr + 2)
    ux = tl.load(V_ptr + offs, mask=mask, other=0.0)
    uy = tl.load(V_ptr + n + offs, mask=mask, other=0.0)
    uz = tl.load(V_ptr + 2 * n + offs, mask=mask, other=0.0)
    if MHD:
        bx = tl.load(V_ptr + 3 * n + offs, mask=mask, other=0.0)
        by = tl.load(V_ptr + 4 * n + offs, mask=mask, other=0.0)
        bz = tl.load(V_ptr + 5 * n + offs, mask=mask, other=0.0)
        tzz = uz * uz - bz * bz
        tl.store(G_ptr + offs, ux * ux - bx * bx - tzz, mask=mask)
        tl.store(G_ptr + n + offs, uy * uy - by * by - tzz, mask=mask)
        tl.store(G_ptr + 2 * n + offs, ux * uy - bx * by, mask=mask)
        tl.store(G_ptr + 3 * n + offs, ux * uz - bx * bz, mask=mask)
        tl.store(G_ptr + 4 * n + offs, uy * uz - by * bz, mask=mask)
        tl.store(G_ptr + 5 * n + offs, ux * by - uy * bx, mask=mask)
        tl.store(G_ptr + 6 * n + offs, ux * bz - uz * bx, mask=mask)
        tl.store(G_ptr + 7 * n + offs, uy * bz - uz * by, mask=mask)
        rate = ((tl.abs(ux) + tl.abs(bx)) * idx + (tl.abs(uy) + tl.abs(by)) * idy
                + (tl.abs(uz) + tl.abs(bz)) * idz)
    else:
        tzz = uz * uz
        tl.store(G_ptr + offs, ux * ux - tzz, mask=mask)
        tl.store(G_ptr + n + offs, uy * uy - tzz, mask=mask)
        tl.store(G_ptr + 2 * n + offs, ux * uy, mask=mask)
        tl.store(G_ptr + 3 * n + offs, ux * uz, mask=mask)
        tl.store(G_ptr + 4 * n + offs, uy * uz, mask=mask)
        rate = tl.abs(ux) * idx + tl.abs(uy) * idy + tl.abs(uz) * idz
    if STATS:
        nonfinite = (((rate - rate) != 0.0) & mask).to(tl.int32)
        usable = mask & ((rate - rate) == 0.0)
        tl.store(bad_ptr + pid, tl.sum(nonfinite, axis=0))
        tl.store(rate_ptr + pid, tl.max(tl.where(usable, rate, 0.0), axis=0))


def imhd_products(V, inv_spacing, mhd, stats, block=1024, num_warps=8):
    """Products ``G`` of ``V`` = [u (3), b (3)]; with ``stats`` also
    (max of sum_i (|u_i| + |b_i|) / dx_i, non-finite count) as device tensors."""
    V = V.contiguous()
    n = V[0].numel()
    nblocks = triton.cdiv(n, block)
    G = torch.empty((8 if mhd else 5,) + tuple(V.shape[1:]), dtype=V.dtype, device=V.device)
    rate = torch.zeros(nblocks if stats else 1, dtype=V.dtype, device=V.device)
    bad = torch.zeros(nblocks if stats else 1, dtype=torch.int32, device=V.device)
    params = torch.tensor(list(inv_spacing), dtype=V.dtype, device=V.device)
    _imhd_products_kernel[(nblocks,)](V, G, rate, bad, n, params,
                                      MHD=bool(mhd), STATS=bool(stats), BLOCK=block,
                                      num_warps=num_warps)
    if stats:
        return G, rate.max(), bad.sum()
    return G


# --- incompressible MHD: spectral RHS and integrating-factor RK stage ----------
#
# P = [1/N, nu, eta, kcut^2, wa, wn, tau_0, tau_n, tau_a].  With E_D(t) =
# exp(-D k^2 t), D = nu for u and eta for b, and N the projected nonlinear term:
#   not FINAL:  acc  = [acc +] wa * E(tau_a) N,   Sout = E(tau_0) U0 + wn * E(tau_n) N
#   FINAL:      Sout = E(tau_0) U0 + [acc +] wa * E(tau_a) N
# Modes with k^2 > kcut^2 get N = 0 (spherical truncation; pass a huge kcut^2
# to keep the full 2/3 box).


@triton.jit
def _imhd_update(c, kr, ki, e0, en, ea, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc, mask, wa, wn,
                 FIRST: tl.constexpr, FINAL: tl.constexpr):
    ur, ui = _ld(U0_ptr, off + c * sSc, mask)
    if FIRST:
        ar = kr * 0.0
        ai = ki * 0.0
    else:
        ar, ai = _ld(A_ptr, off + c * sAc, mask)
    if FINAL:
        _st(Sout_ptr, off + c * sSc, mask, e0 * ur + ar + wa * ea * kr, e0 * ui + ai + wa * ea * ki)
    else:
        _st(A_ptr, off + c * sAc, mask, ar + wa * ea * kr, ai + wa * ea * ki)
        _st(Sout_ptr, off + c * sSc, mask, e0 * ur + wn * en * kr, e0 * ui + wn * en * ki)


@triton.jit
def _imhd_stage_kernel(
    G_ptr, sGc,
    U0_ptr, Sout_ptr, sSc,
    A_ptr, sAc,
    kx_ptr, ky_ptr, kz_ptr, P_ptr,
    total, Kx, Ky,
    MHD: tl.constexpr, FIRST: tl.constexpr, FINAL: tl.constexpr, BLOCK: tl.constexpr,
):
    # int32 scalar arguments: multiples of them overflow past 2**31 elements.
    sGc = sGc.to(tl.int64)
    sSc = sSc.to(tl.int64)
    sAc = sAc.to(tl.int64)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // Kx
    ix = offs - row * Kx
    iz = row // Ky
    iy = row - iz * Ky
    kx = tl.load(kx_ptr + ix, mask=mask, other=0.0)
    ky = tl.load(ky_ptr + iy, mask=mask, other=0.0)
    kz = tl.load(kz_ptr + iz, mask=mask, other=0.0)
    inv_n = tl.load(P_ptr)
    nu = tl.load(P_ptr + 1)
    eta = tl.load(P_ptr + 2)
    kcut2 = tl.load(P_ptr + 3)
    wa = tl.load(P_ptr + 4)
    wn = tl.load(P_ptr + 5)
    tau0 = tl.load(P_ptr + 6)
    taun = tl.load(P_ptr + 7)
    taua = tl.load(P_ptr + 8)
    k2 = kx * kx + ky * ky + kz * kz
    scale = tl.where(k2 <= kcut2, inv_n, 0.0)  # flux spectra come from an unnormalized transform
    inv_k2 = tl.where(k2 == 0.0, 0.0, 1.0 / tl.where(k2 == 0.0, 1.0, k2))
    off = 2 * offs

    # Momentum: N_i = -i k_j T_ij with Tzz = 0, then the Leray projection.
    g0r, g0i = _ld(G_ptr, off, mask)
    g1r, g1i = _ld(G_ptr, off + sGc, mask)
    g2r, g2i = _ld(G_ptr, off + 2 * sGc, mask)
    g3r, g3i = _ld(G_ptr, off + 3 * sGc, mask)
    g4r, g4i = _ld(G_ptr, off + 4 * sGc, mask)
    nxr = scale * (kx * g0i + ky * g2i + kz * g3i)
    nxi = -scale * (kx * g0r + ky * g2r + kz * g3r)
    nyr = scale * (kx * g2i + ky * g1i + kz * g4i)
    nyi = -scale * (kx * g2r + ky * g1r + kz * g4r)
    nzr = scale * (kx * g3i + ky * g4i)
    nzi = -scale * (kx * g3r + ky * g4r)
    cr = (kx * nxr + ky * nyr + kz * nzr) * inv_k2
    ci = (kx * nxi + ky * nyi + kz * nzi) * inv_k2
    e0 = tl.exp(-nu * k2 * tau0)
    en = tl.exp(-nu * k2 * taun)
    ea = tl.exp(-nu * k2 * taua)
    _imhd_update(0, nxr - kx * cr, nxi - kx * ci, e0, en, ea, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc,
                 mask, wa, wn, FIRST, FINAL)
    _imhd_update(1, nyr - ky * cr, nyi - ky * ci, e0, en, ea, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc,
                 mask, wa, wn, FIRST, FINAL)
    _imhd_update(2, nzr - kz * cr, nzi - kz * ci, e0, en, ea, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc,
                 mask, wa, wn, FIRST, FINAL)

    if MHD:
        # Induction: db_i/dt = i k_j A_ij, A antisymmetric (solenoidal by construction).
        axyr, axyi = _ld(G_ptr, off + 5 * sGc, mask)
        axzr, axzi = _ld(G_ptr, off + 6 * sGc, mask)
        ayzr, ayzi = _ld(G_ptr, off + 7 * sGc, mask)
        bxr = -scale * (ky * axyi + kz * axzi)
        bxi = scale * (ky * axyr + kz * axzr)
        byr = -scale * (kz * ayzi - kx * axyi)
        byi = scale * (kz * ayzr - kx * axyr)
        bzr = scale * (kx * axzi + ky * ayzi)
        bzi = -scale * (kx * axzr + ky * ayzr)
        f0 = tl.exp(-eta * k2 * tau0)
        fn = tl.exp(-eta * k2 * taun)
        fa = tl.exp(-eta * k2 * taua)
        _imhd_update(3, bxr, bxi, f0, fn, fa, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc, mask, wa, wn,
                     FIRST, FINAL)
        _imhd_update(4, byr, byi, f0, fn, fa, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc, mask, wa, wn,
                     FIRST, FINAL)
        _imhd_update(5, bzr, bzi, f0, fn, fa, U0_ptr, Sout_ptr, A_ptr, off, sSc, sAc, mask, wa, wn,
                     FIRST, FINAL)


def imhd_stage(G, U0, Sout, acc, waves, params, mhd, first, final, block=256, num_warps=4):
    """Launch one stage on compact contiguous spectra ``(C, Kz, nky, Kx)``.

    ``G`` holds the unnormalized product spectra, ``params`` the nine scalars
    listed above as a tensor of the real dtype.  ``Sout`` may be ``U0`` in the
    final stage.
    """
    Gr, U0r, Sr, Ar = (torch.view_as_real(x) for x in (G, U0, Sout, acc))
    assert U0r.stride() == Sr.stride() == Ar.stride()
    assert G.is_contiguous() and U0.is_contiguous() and Sout.is_contiguous() and acc.is_contiguous()
    Kz, Ky, Kx = U0.shape[1:]
    total = Kz * Ky * Kx
    kx, ky, kz = waves
    _imhd_stage_kernel[(triton.cdiv(total, block),)](
        Gr, Gr.stride(0),
        U0r, Sr, Sr.stride(0),
        Ar, Ar.stride(0),
        kx, ky, kz, params,
        total, Kx, Ky,
        MHD=bool(mhd), FIRST=bool(first), FINAL=bool(final), BLOCK=block, num_warps=num_warps,
    )
