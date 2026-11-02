"""Triton kernel for the spectral RK stages of ``spectral3d`` (CUDA only).

One kernel per RK stage forms the spectral RHS of all five conserved variables
and applies the stage update.  It visits only the Fourier modes kept by the
2/3 dealias rule (8/27 of the half spectrum), so spectral traffic scales with
the retained modes; the stage-state and solution buffers are zero elsewhere.

Spectral buffers hold rfftn coefficients divided by Nx*Ny*Nz (``norm="forward"``
convention), so ``irfftn(..., norm="forward")`` needs no separate scaling pass;
flux spectra come from a plain ``rfftn`` and are rescaled inside the kernel.
Masks and the split-step dissipation decay are separable and are rebuilt per
mode from 1D factors.
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


@triton.jit
def _div(kx, ky, kz, m, axr, axi, ayr, ayi, azr, azi):
    """-m * i (kx*a_x + ky*a_y + kz*a_z)."""
    sr = kx * axr + ky * ayr + kz * azr
    si = kx * axi + ky * ayi + kz * azi
    return m * si, -m * sr


@triton.jit
def _update(c, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn,
            FIRST: tl.constexpr, FINAL: tl.constexpr):
    ur, ui = _ld(U0_ptr, offS + c * sSc, mask)
    if FIRST:
        ar = kr * 0.0
        ai = ki * 0.0
    else:
        ar, ai = _ld(A_ptr, offA + c * sAc, mask)
    if FINAL:
        # Pre-projection state (what the positivity clamp sees) and the
        # dissipated, projected state.
        hr = ur + ar + wa * kr
        hi = ui + ai + wa * ki
        _st(H_ptr, offS + c * sSc, mask, hr, hi)
        _st(Sout_ptr, offS + c * sSc, mask, mfin * hr, mfin * hi)
    else:
        _st(A_ptr, offA + c * sAc, mask, ar + wa * kr, ai + wa * ki)
        _st(Sout_ptr, offS + c * sSc, mask, ur + wn * kr, ui + wn * ki)


@triton.jit
def _mode_index(BLOCK: tl.constexpr, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz):
    """Map a flat index over the visited modes to (ix, iy, iz) in the rfftn layout."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // Kx
    ix = offs - row * Kx
    izr = row // Kyr
    iyr = row - izr * Kyr
    iz = tl.where(izr <= cz, izr, izr + (Nz - Kzr))
    iy = tl.where(iyr <= cy, iyr, iyr + (Ny - Kyr))
    return offs, mask, ix, iy, iz


@triton.jit
def _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask):
    kx = tl.load(kx_ptr + ix, mask=mask, other=0.0)
    ky = tl.load(ky_ptr + iy, mask=mask, other=0.0)
    kz = tl.load(kz_ptr + iz, mask=mask, other=0.0)
    m = (tl.load(mz_ptr + iz, mask=mask, other=0.0) * tl.load(my_ptr + iy, mask=mask, other=0.0)
         * tl.load(mx_ptr + ix, mask=mask, other=0.0))
    return kx, ky, kz, m


@triton.jit
def _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask):
    return (tl.load(fz_ptr + iz, mask=mask, other=0.0) * tl.load(fy_ptr + iy, mask=mask, other=0.0)
            * tl.load(fx_ptr + ix, mask=mask, other=0.0))


@triton.jit
def _euler3d_stage_kernel(
    F_ptr, sFc, sFz, sFy,
    Sin_ptr, Sout_ptr, H_ptr, U0_ptr, sSc, sSz, sSy,
    A_ptr, sAc,
    kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, fx_ptr, fy_ptr, fz_ptr,
    w_ptr, ia, inx,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz,
    FIRST: tl.constexpr, FINAL: tl.constexpr, ACC_FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    kx, ky, kz, m = _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask)
    inv_n = tl.load(w_ptr)
    wa = tl.load(w_ptr + ia)
    wn = tl.load(w_ptr + inx)
    mf = m * inv_n  # flux spectra come from an unnormalized rfftn
    if FINAL:
        mfin = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
    else:
        mfin = m

    offF = iz * sFz + iy * sFy + 2 * ix
    offS = iz * sSz + iy * sSy + 2 * ix
    if ACC_FULL:  # accumulator shares the spectral layout (full-mode steps)
        offA = offS
    else:  # compact accumulator over the visited modes
        offA = 2 * offs

    # Mass: divergence of the stage momentum (read before Sout may overwrite Sin).
    m1r, m1i = _ld(Sin_ptr, offS + sSc, mask)
    m2r, m2i = _ld(Sin_ptr, offS + 2 * sSc, mask)
    m3r, m3i = _ld(Sin_ptr, offS + 3 * sSc, mask)
    kr, ki = _div(kx, ky, kz, m, m1r, m1i, m2r, m2i, m3r, m3i)
    _update(0, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)

    # Flux order: [Txx, Tyy, Tzz, Txy, Txz, Tyz, Hx, Hy, Hz]
    f0r, f0i = _ld(F_ptr, offF, mask)
    f3r, f3i = _ld(F_ptr, offF + 3 * sFc, mask)
    f4r, f4i = _ld(F_ptr, offF + 4 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f0r, f0i, f3r, f3i, f4r, f4i)
    _update(1, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)

    f1r, f1i = _ld(F_ptr, offF + sFc, mask)
    f5r, f5i = _ld(F_ptr, offF + 5 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f3r, f3i, f1r, f1i, f5r, f5i)
    _update(2, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)

    f2r, f2i = _ld(F_ptr, offF + 2 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f4r, f4i, f5r, f5i, f2r, f2i)
    _update(3, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)

    f6r, f6i = _ld(F_ptr, offF + 6 * sFc, mask)
    f7r, f7i = _ld(F_ptr, offF + 7 * sFc, mask)
    f8r, f8i = _ld(F_ptr, offF + 8 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f6r, f6i, f7r, f7i, f8r, f8i)
    _update(4, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)


@triton.jit
def _euler3d_rhs_kernel(
    F_ptr, sFc, sFz, sFy,
    S_ptr, sSc, sSz, sSy,
    K_ptr, sKc,
    kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, w_ptr,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz,
    BLOCK: tl.constexpr,
):
    """Compact spectral RHS k (5, total) from flux spectra and stage-state momentum."""
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    kx, ky, kz, m = _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask)
    mf = m * tl.load(w_ptr)
    offF = iz * sFz + iy * sFy + 2 * ix
    offS = iz * sSz + iy * sSy + 2 * ix
    offK = 2 * offs

    m1r, m1i = _ld(S_ptr, offS + sSc, mask)
    m2r, m2i = _ld(S_ptr, offS + 2 * sSc, mask)
    m3r, m3i = _ld(S_ptr, offS + 3 * sSc, mask)
    kr, ki = _div(kx, ky, kz, m, m1r, m1i, m2r, m2i, m3r, m3i)
    _st(K_ptr, offK, mask, kr, ki)

    f0r, f0i = _ld(F_ptr, offF, mask)
    f3r, f3i = _ld(F_ptr, offF + 3 * sFc, mask)
    f4r, f4i = _ld(F_ptr, offF + 4 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f0r, f0i, f3r, f3i, f4r, f4i)
    _st(K_ptr, offK + sKc, mask, kr, ki)

    f1r, f1i = _ld(F_ptr, offF + sFc, mask)
    f5r, f5i = _ld(F_ptr, offF + 5 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f3r, f3i, f1r, f1i, f5r, f5i)
    _st(K_ptr, offK + 2 * sKc, mask, kr, ki)

    f2r, f2i = _ld(F_ptr, offF + 2 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f4r, f4i, f5r, f5i, f2r, f2i)
    _st(K_ptr, offK + 3 * sKc, mask, kr, ki)

    f6r, f6i = _ld(F_ptr, offF + 6 * sFc, mask)
    f7r, f7i = _ld(F_ptr, offF + 7 * sFc, mask)
    f8r, f8i = _ld(F_ptr, offF + 8 * sFc, mask)
    kr, ki = _div(kx, ky, kz, mf, f6r, f6i, f7r, f7i, f8r, f8i)
    _st(K_ptr, offK + 4 * sKc, mask, kr, ki)


@triton.jit
def _combine_kernel(
    K_ptr, sKj, sKc,
    B_ptr, O_ptr, sSc, sSz, sSy,
    fx_ptr, fy_ptr, fz_ptr,
    w_ptr, w0, nk,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz,
    HAS_BASE: tl.constexpr, MASK: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr,
):
    """O = [f *] ([B +] sum_j w[w0 + j] * K[j]) over the visited modes, NC components."""
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    if MASK:
        m = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
    offS = iz * sSz + iy * sSy + 2 * ix
    offK = 2 * offs
    for c in tl.static_range(NC):
        if HAS_BASE:
            re, im = _ld(B_ptr, offS + c * sSc, mask)
        else:
            re = tl.zeros([BLOCK], dtype=O_ptr.dtype.element_ty)
            im = tl.zeros([BLOCK], dtype=O_ptr.dtype.element_ty)
        for j in range(nk):
            wj = tl.load(w_ptr + w0 + j)
            kr, ki = _ld(K_ptr, offK + j * sKj + c * sKc, mask)
            re += wj * kr
            im += wj * ki
        if MASK:
            re = m * re
            im = m * im
        _st(O_ptr, offS + c * sSc, mask, re, im)


class RetainedModes:
    """Index geometry of the retained (dealiased) part of the rfftn half spectrum."""

    def __init__(self, shape, dealias: bool):
        Nz, Ny, Nx = shape
        Xh = Nx // 2 + 1
        if dealias:  # Grid3D keeps |k| <= (N - 1) // 3 on every axis
            self.cz, self.cy = (Nz - 1) // 3, (Ny - 1) // 3
            self.Kzr, self.Kyr = 2 * self.cz + 1, 2 * self.cy + 1
            self.Kx = (Nx - 1) // 3 + 1
        else:
            self.cz, self.cy = Nz, Ny
            self.Kzr, self.Kyr = Nz, Ny
            self.Kx = Xh
        self.Nz, self.Ny = Nz, Ny
        self.total = self.Kzr * self.Kyr * self.Kx


def stage(F, Sin, Sout, U0, acc, w, ia, inx, waves, modes: RetainedModes, first: bool, final: bool,
          high=None, final_factors=None, block: int = 256, num_warps: int = 4) -> None:
    """Launch one RK stage. Complex tensors F (9,...), S/U0 (5,...) share the rfftn layout.

    ``acc`` is either compact, shape (5, modes.total), or shares the S layout.
    The final stage writes the pre-projection state to ``high`` and
    ``final_factors`` (per-axis 1D arrays) times it to ``Sout``.
    """
    Fr = torch.view_as_real(F)
    Sinr = torch.view_as_real(Sin)
    Soutr = torch.view_as_real(Sout)
    U0r = torch.view_as_real(U0)
    Ar = torch.view_as_real(acc)
    Hr = torch.view_as_real(high) if high is not None else Soutr
    assert Sinr.stride() == Soutr.stride() == U0r.stride() == Hr.stride()
    acc_full = acc.dim() > 2
    if acc_full:
        assert Ar.stride() == Soutr.stride()
    kx, ky, kz, mx, my, mz = waves
    fx, fy, fz = final_factors if final_factors is not None else (mx, my, mz)
    grid = (triton.cdiv(modes.total, block),)
    _euler3d_stage_kernel[grid](
        Fr, Fr.stride(0), Fr.stride(1), Fr.stride(2),
        Sinr, Soutr, Hr, U0r, Soutr.stride(0), Soutr.stride(1), Soutr.stride(2),
        Ar, Ar.stride(0),
        kx, ky, kz, mx, my, mz, fx, fy, fz,
        w, ia, inx,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        FIRST=first, FINAL=final, ACC_FULL=acc_full, BLOCK=block, num_warps=num_warps,
    )


def rhs(F, S, K, w, waves, modes: RetainedModes, block: int = 256, num_warps: int = 4) -> None:
    """Compact spectral RHS into K (5, modes.total) from F (9,...) and stage state S (5,...)."""
    Fr, Sr, Kr = torch.view_as_real(F), torch.view_as_real(S), torch.view_as_real(K)
    kx, ky, kz, mx, my, mz = waves
    _euler3d_rhs_kernel[(triton.cdiv(modes.total, block),)](
        Fr, Fr.stride(0), Fr.stride(1), Fr.stride(2),
        Sr, Sr.stride(0), Sr.stride(1), Sr.stride(2),
        Kr, Kr.stride(0),
        kx, ky, kz, mx, my, mz, w,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        BLOCK=block, num_warps=num_warps,
    )


def combine(K, nk, base, out, w, w0, factors, modes: RetainedModes, masked: bool,
            block: int = 256, num_warps: int = 4) -> None:
    """out = [f *] ([base +] sum_{j<nk} w[w0+j] K[j]); K is (s, 5, modes.total) or None.

    ``factors`` are per-axis 1D arrays (x, y, z) whose product scales the
    result when ``masked``.  ``base`` and ``out`` may be the same buffer.
    """
    Or = torch.view_as_real(out)
    Kr = torch.view_as_real(K) if K is not None else Or
    sKj, sKc = (Kr.stride(0), Kr.stride(1)) if K is not None else (0, 0)
    Br = torch.view_as_real(base) if base is not None else Or
    assert Br.stride() == Or.stride()
    fx, fy, fz = factors if factors is not None else (w, w, w)
    _combine_kernel[(triton.cdiv(modes.total, block),)](
        Kr, sKj, sKc,
        Br, Or, Or.stride(0), Or.stride(1), Or.stride(2),
        fx, fy, fz,
        w, w0, nk,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        HAS_BASE=base is not None, MASK=masked, NC=out.shape[0], BLOCK=block, num_warps=num_warps,
    )


# --- ideal/resistive MHD (8 components) --------------------------------------
#
# Flux spectra order: [Txx, Tyy, Tzz, Txy, Txz, Tyz, Hx, Hy, Hz, Axy, Axz, Ayz]
# with T_ij = rho u_i u_j + (p + B^2/2) delta_ij - B_i B_j,
# H_i = (E + p + B^2/2) u_i - B_i (u.B) and A_ij = u_i B_j - u_j B_i.
# The induction fluxes are F_x[By] = Axy, F_x[Bz] = Axz, F_y[Bz] = Ayz and
# F_j[B_i] = -F_i[B_j], so the spectral RHS of B is exactly solenoidal.


@triton.jit
def _mhd_emit(c, kr, ki, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, K_ptr, offA, sSc, sAc, mask,
              wa, wn, MODE: tl.constexpr, FIRST: tl.constexpr):
    """MODE 0: stage update; 1: store k only (embedded RK)."""
    if MODE == 1:
        _st(K_ptr, offA + c * sAc, mask, kr, ki)
    else:
        ur, ui = _ld(U0_ptr, offS + c * sSc, mask)
        if FIRST:
            ar = kr * 0.0
            ai = ki * 0.0
        else:
            ar, ai = _ld(A_ptr, offA + c * sAc, mask)
        _st(A_ptr, offA + c * sAc, mask, ar + wa * kr, ai + wa * ki)
        _st(Sout_ptr, offS + c * sSc, mask, ur + wn * kr, ui + wn * ki)


@triton.jit
def _mhd_final(c, kr, ki, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST: tl.constexpr):
    ur, ui = _ld(U0_ptr, offS + c * sSc, mask)
    if FIRST:
        return ur + wa * kr, ui + wa * ki
    ar, ai = _ld(A_ptr, offA + c * sAc, mask)
    return ur + ar + wa * kr, ui + ai + wa * ki


@triton.jit
def _mhd3d_stage_kernel(
    F_ptr, sFc, sFz, sFy,
    Sin_ptr, Sout_ptr, H_ptr, U0_ptr, sSc, sSz, sSy,
    A_ptr, sAc,
    kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, fx_ptr, fy_ptr, fz_ptr,
    w_ptr, ia, inx,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz,
    MODE: tl.constexpr, FIRST: tl.constexpr, FINAL: tl.constexpr, BLOCK: tl.constexpr,
):
    """One MHD RK stage (MODE 0) or the compact spectral RHS (MODE 1) over the visited modes.

    w = [1/N, eta, nu, ...]; the final stage writes the pre-projection state to
    H and f * (Leray-projected state) to Sout.
    """
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    kx, ky, kz, m = _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask)
    inv_n = tl.load(w_ptr)
    eta = tl.load(w_ptr + 1)
    nu = tl.load(w_ptr + 2)
    wa = tl.load(w_ptr + ia)
    wn = tl.load(w_ptr + inx)
    mf = m * inv_n
    k2m = (kx * kx + ky * ky + kz * kz) * m
    offF = iz * sFz + iy * sFy + 2 * ix
    offS = iz * sSz + iy * sSy + 2 * ix
    offA = 2 * offs
    zero = kx * 0.0

    # Everything read from Sin happens before Sout (possibly the same buffer) is written.
    m1r, m1i = _ld(Sin_ptr, offS + sSc, mask)
    m2r, m2i = _ld(Sin_ptr, offS + 2 * sSc, mask)
    m3r, m3i = _ld(Sin_ptr, offS + 3 * sSc, mask)
    b1r, b1i = _ld(Sin_ptr, offS + 5 * sSc, mask)
    b2r, b2i = _ld(Sin_ptr, offS + 6 * sSc, mask)
    b3r, b3i = _ld(Sin_ptr, offS + 7 * sSc, mask)

    # density
    k0r, k0i = _div(kx, ky, kz, m, m1r, m1i, m2r, m2i, m3r, m3i)
    # momentum (+ viscosity)
    f0r, f0i = _ld(F_ptr, offF, mask)
    f3r, f3i = _ld(F_ptr, offF + 3 * sFc, mask)
    f4r, f4i = _ld(F_ptr, offF + 4 * sFc, mask)
    k1r, k1i = _div(kx, ky, kz, mf, f0r, f0i, f3r, f3i, f4r, f4i)
    k1r -= nu * k2m * m1r
    k1i -= nu * k2m * m1i
    f1r, f1i = _ld(F_ptr, offF + sFc, mask)
    f5r, f5i = _ld(F_ptr, offF + 5 * sFc, mask)
    k2r, k2i = _div(kx, ky, kz, mf, f3r, f3i, f1r, f1i, f5r, f5i)
    k2r -= nu * k2m * m2r
    k2i -= nu * k2m * m2i
    f2r, f2i = _ld(F_ptr, offF + 2 * sFc, mask)
    k3r, k3i = _div(kx, ky, kz, mf, f4r, f4i, f5r, f5i, f2r, f2i)
    k3r -= nu * k2m * m3r
    k3i -= nu * k2m * m3i
    # energy
    f6r, f6i = _ld(F_ptr, offF + 6 * sFc, mask)
    f7r, f7i = _ld(F_ptr, offF + 7 * sFc, mask)
    f8r, f8i = _ld(F_ptr, offF + 8 * sFc, mask)
    k4r, k4i = _div(kx, ky, kz, mf, f6r, f6i, f7r, f7i, f8r, f8i)
    # induction (+ resistivity)
    axyr, axyi = _ld(F_ptr, offF + 9 * sFc, mask)
    axzr, axzi = _ld(F_ptr, offF + 10 * sFc, mask)
    ayzr, ayzi = _ld(F_ptr, offF + 11 * sFc, mask)
    k5r, k5i = _div(kx, ky, kz, mf, zero, zero, -axyr, -axyi, -axzr, -axzi)
    k6r, k6i = _div(kx, ky, kz, mf, axyr, axyi, zero, zero, -ayzr, -ayzi)
    k7r, k7i = _div(kx, ky, kz, mf, axzr, axzi, ayzr, ayzi, zero, zero)
    k5r -= eta * k2m * b1r
    k5i -= eta * k2m * b1i
    k6r -= eta * k2m * b2r
    k6i -= eta * k2m * b2i
    k7r -= eta * k2m * b3r
    k7i -= eta * k2m * b3i

    if FINAL:
        f = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
        h0r, h0i = _mhd_final(0, k0r, k0i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h1r, h1i = _mhd_final(1, k1r, k1i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h2r, h2i = _mhd_final(2, k2r, k2i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h3r, h3i = _mhd_final(3, k3r, k3i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h4r, h4i = _mhd_final(4, k4r, k4i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h5r, h5i = _mhd_final(5, k5r, k5i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h6r, h6i = _mhd_final(6, k6r, k6i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        h7r, h7i = _mhd_final(7, k7r, k7i, U0_ptr, offS, A_ptr, offA, sSc, sAc, mask, wa, FIRST)
        _st(H_ptr, offS, mask, h0r, h0i)
        _st(H_ptr, offS + sSc, mask, h1r, h1i)
        _st(H_ptr, offS + 2 * sSc, mask, h2r, h2i)
        _st(H_ptr, offS + 3 * sSc, mask, h3r, h3i)
        _st(H_ptr, offS + 4 * sSc, mask, h4r, h4i)
        _st(H_ptr, offS + 5 * sSc, mask, h5r, h5i)
        _st(H_ptr, offS + 6 * sSc, mask, h6r, h6i)
        _st(H_ptr, offS + 7 * sSc, mask, h7r, h7i)
        # Leray projection of B (k = 0 keeps the mean field).
        k2 = kx * kx + ky * ky + kz * kz
        inv_k2 = tl.where(k2 == 0.0, 0.0, 1.0 / tl.where(k2 == 0.0, 1.0, k2))
        cr = (kx * h5r + ky * h6r + kz * h7r) * inv_k2
        ci = (kx * h5i + ky * h6i + kz * h7i) * inv_k2
        _st(Sout_ptr, offS, mask, f * h0r, f * h0i)
        _st(Sout_ptr, offS + sSc, mask, f * h1r, f * h1i)
        _st(Sout_ptr, offS + 2 * sSc, mask, f * h2r, f * h2i)
        _st(Sout_ptr, offS + 3 * sSc, mask, f * h3r, f * h3i)
        _st(Sout_ptr, offS + 4 * sSc, mask, f * h4r, f * h4i)
        _st(Sout_ptr, offS + 5 * sSc, mask, f * (h5r - kx * cr), f * (h5i - kx * ci))
        _st(Sout_ptr, offS + 6 * sSc, mask, f * (h6r - ky * cr), f * (h6i - ky * ci))
        _st(Sout_ptr, offS + 7 * sSc, mask, f * (h7r - kz * cr), f * (h7i - kz * ci))
    else:
        _mhd_emit(0, k0r, k0i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(1, k1r, k1i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(2, k2r, k2i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(3, k3r, k3i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(4, k4r, k4i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(5, k5r, k5i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(6, k6r, k6i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)
        _mhd_emit(7, k7r, k7i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, A_ptr, offA, sSc, sAc, mask, wa, wn, MODE, FIRST)


@triton.jit
def _leray_scale_kernel(
    X_ptr, sSc, sSz, sSy, kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, fx_ptr, fy_ptr, fz_ptr,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz, BLOCK: tl.constexpr,
):
    """In place: X = f * X for all 8 components, with B Leray-projected first."""
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    kx, ky, kz, m = _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask)
    f = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
    offS = iz * sSz + iy * sSy + 2 * ix
    for c in tl.static_range(5):
        xr, xi = _ld(X_ptr, offS + c * sSc, mask)
        _st(X_ptr, offS + c * sSc, mask, f * xr, f * xi)
    b1r, b1i = _ld(X_ptr, offS + 5 * sSc, mask)
    b2r, b2i = _ld(X_ptr, offS + 6 * sSc, mask)
    b3r, b3i = _ld(X_ptr, offS + 7 * sSc, mask)
    k2 = kx * kx + ky * ky + kz * kz
    inv_k2 = tl.where(k2 == 0.0, 0.0, 1.0 / tl.where(k2 == 0.0, 1.0, k2))
    cr = (kx * b1r + ky * b2r + kz * b3r) * inv_k2
    ci = (kx * b1i + ky * b2i + kz * b3i) * inv_k2
    _st(X_ptr, offS + 5 * sSc, mask, f * (b1r - kx * cr), f * (b1i - kx * ci))
    _st(X_ptr, offS + 6 * sSc, mask, f * (b2r - ky * cr), f * (b2i - ky * ci))
    _st(X_ptr, offS + 7 * sSc, mask, f * (b3r - kz * cr), f * (b3i - kz * ci))


def mhd_stage(F, Sin, Sout, U0, acc, w, ia, inx, waves, modes: RetainedModes, first: bool,
              final: bool, rhs_only: bool = False, high=None, final_factors=None,
              block: int = 256, num_warps: int = 4) -> None:
    """Launch one MHD stage; with ``rhs_only`` the compact k goes to ``acc``."""
    Fr = torch.view_as_real(F)
    Sinr, Soutr, U0r = (torch.view_as_real(x) for x in (Sin, Sout, U0))
    Ar = torch.view_as_real(acc)
    Hr = torch.view_as_real(high) if high is not None else Soutr
    assert Sinr.stride() == Soutr.stride() == U0r.stride() == Hr.stride()
    kx, ky, kz, mx, my, mz = waves
    fx, fy, fz = final_factors if final_factors is not None else (mx, my, mz)
    _mhd3d_stage_kernel[(triton.cdiv(modes.total, block),)](
        Fr, Fr.stride(0), Fr.stride(1), Fr.stride(2),
        Sinr, Soutr, Hr, U0r, Soutr.stride(0), Soutr.stride(1), Soutr.stride(2),
        Ar, Ar.stride(0),
        kx, ky, kz, mx, my, mz, fx, fy, fz,
        w, ia, inx,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        MODE=1 if rhs_only else 0, FIRST=first, FINAL=final, BLOCK=block, num_warps=num_warps,
    )


def leray_scale(X, waves, factors, modes: RetainedModes, block: int = 256, num_warps: int = 4) -> None:
    Xr = torch.view_as_real(X)
    kx, ky, kz, mx, my, mz = waves
    fx, fy, fz = factors
    _leray_scale_kernel[(triton.cdiv(modes.total, block),)](
        Xr, Xr.stride(0), Xr.stride(1), Xr.stride(2), kx, ky, kz, mx, my, mz, fx, fy, fz,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        BLOCK=block, num_warps=num_warps,
    )


# --- 2D Euler (4 components), spectra viewed as (C, 1, Ny, Xh) ----------------
#
# Flux spectra order: [Txx, Tyy, Txy, Hx, Hy].


@triton.jit
def _div2(kx, ky, m, axr, axi, ayr, ayi):
    sr = kx * axr + ky * ayr
    si = kx * axi + ky * ayi
    return m * si, -m * sr


@triton.jit
def _euler2d_stage_kernel(
    F_ptr, sFc, sFz, sFy,
    Sin_ptr, Sout_ptr, H_ptr, U0_ptr, sSc, sSz, sSy,
    A_ptr, sAc,
    kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, fx_ptr, fy_ptr, fz_ptr,
    w_ptr, ia, inx,
    total, Kx, Kyr, Kzr, cy, cz, Ny, Nz,
    MODE: tl.constexpr, FIRST: tl.constexpr, FINAL: tl.constexpr, ACC_FULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """2D Euler RK stage (MODE 0) or compact spectral RHS into A (MODE 1)."""
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    kx, ky, kz, m = _waves(kx_ptr, ky_ptr, kz_ptr, mx_ptr, my_ptr, mz_ptr, ix, iy, iz, mask)
    inv_n = tl.load(w_ptr)
    wa = tl.load(w_ptr + ia)
    wn = tl.load(w_ptr + inx)
    mf = m * inv_n
    if FINAL:
        mfin = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
    else:
        mfin = m
    offF = iz * sFz + iy * sFy + 2 * ix
    offS = iz * sSz + iy * sSy + 2 * ix
    if ACC_FULL:
        offA = offS
    else:
        offA = 2 * offs

    m1r, m1i = _ld(Sin_ptr, offS + sSc, mask)
    m2r, m2i = _ld(Sin_ptr, offS + 2 * sSc, mask)
    k0r, k0i = _div2(kx, ky, m, m1r, m1i, m2r, m2i)
    f0r, f0i = _ld(F_ptr, offF, mask)
    f1r, f1i = _ld(F_ptr, offF + sFc, mask)
    f2r, f2i = _ld(F_ptr, offF + 2 * sFc, mask)
    k1r, k1i = _div2(kx, ky, mf, f0r, f0i, f2r, f2i)
    k2r, k2i = _div2(kx, ky, mf, f2r, f2i, f1r, f1i)
    f3r, f3i = _ld(F_ptr, offF + 3 * sFc, mask)
    f4r, f4i = _ld(F_ptr, offF + 4 * sFc, mask)
    k3r, k3i = _div2(kx, ky, mf, f3r, f3i, f4r, f4i)
    if MODE == 1:
        _st(A_ptr, offA, mask, k0r, k0i)
        _st(A_ptr, offA + sAc, mask, k1r, k1i)
        _st(A_ptr, offA + 2 * sAc, mask, k2r, k2i)
        _st(A_ptr, offA + 3 * sAc, mask, k3r, k3i)
    else:
        _update(0, k0r, k0i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)
        _update(1, k1r, k1i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)
        _update(2, k2r, k2i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)
        _update(3, k3r, k3i, U0_ptr, offS, Sout_ptr, H_ptr, A_ptr, offA, sSc, sAc, mask, mfin, wa, wn, FIRST, FINAL)


def _launch_euler2d(F, Sin, Sout, U0, acc, w, ia, inx, waves, modes, mode, first, final, high,
                    final_factors, block=256, num_warps=4):
    Fr = torch.view_as_real(F)
    Sinr, Soutr, U0r = (torch.view_as_real(x) for x in (Sin, Sout, U0))
    Ar = torch.view_as_real(acc)
    Hr = torch.view_as_real(high) if high is not None else Soutr
    assert Sinr.stride() == Soutr.stride() == U0r.stride() == Hr.stride()
    acc_full = acc.dim() > 2
    kx, ky, kz, mx, my, mz = waves
    fx, fy, fz = final_factors if final_factors is not None else (mx, my, mz)
    _euler2d_stage_kernel[(triton.cdiv(modes.total, block),)](
        Fr, Fr.stride(0), Fr.stride(1), Fr.stride(2),
        Sinr, Soutr, Hr, U0r, Soutr.stride(0), Soutr.stride(1), Soutr.stride(2),
        Ar, Ar.stride(0),
        kx, ky, kz, mx, my, mz, fx, fy, fz,
        w, ia, inx,
        modes.total, modes.Kx, modes.Kyr, modes.Kzr, modes.cy, modes.cz, modes.Ny, modes.Nz,
        MODE=mode, FIRST=first, FINAL=final, ACC_FULL=acc_full, BLOCK=block, num_warps=num_warps,
    )


def euler2d_stage(F, Sin, Sout, U0, acc, w, ia, inx, waves, modes: RetainedModes, first: bool,
                  final: bool, high=None, final_factors=None) -> None:
    """2D counterpart of ``stage``; arrays in the (C, 1, Ny, Xh) layout."""
    _launch_euler2d(F, Sin, Sout, U0, acc, w, ia, inx, waves, modes, 0, first, final, high,
                    final_factors)


def euler2d_rhs(F, S, K, w, waves, modes: RetainedModes) -> None:
    """Compact 2D spectral RHS into K (4, modes.total)."""
    _launch_euler2d(F, S, S, S, K, w, 0, 0, waves, modes, 1, False, False, None, None)


# --- physical-space flux kernels ----------------------------------------------
#
# One pass: each element's conserved values are loaded once and every flux
# field is stored; COUNT also adds the number of positivity-floor violations
# (rho < rho_min or p < p_min, NaN never counts) to cnt_ptr.


@triton.jit
def _count_store(cnt_ptr, rho, p, rho_min, p_min, mask, COUNT: tl.constexpr):
    if COUNT:
        bad = ((rho < rho_min) & mask).to(tl.int32) + ((p < p_min) & mask).to(tl.int32)
        tl.atomic_add(cnt_ptr, tl.sum(bad, axis=0).to(tl.int64))


@triton.jit
def _euler2d_flux_kernel(U_ptr, F_ptr, cnt_ptr, n, P_ptr,
                         COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gamma = tl.load(P_ptr)
    rho_min = tl.load(P_ptr + 1)
    p_min = tl.load(P_ptr + 2)
    rho = tl.load(U_ptr + offs, mask=mask, other=1.0)
    mx = tl.load(U_ptr + n + offs, mask=mask, other=0.0)
    my = tl.load(U_ptr + 2 * n + offs, mask=mask, other=0.0)
    E = tl.load(U_ptr + 3 * n + offs, mask=mask, other=1.0)
    ux = mx / rho
    uy = my / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy))
    H = E + p
    tl.store(F_ptr + offs, mx * ux + p, mask=mask)
    tl.store(F_ptr + n + offs, my * uy + p, mask=mask)
    tl.store(F_ptr + 2 * n + offs, mx * uy, mask=mask)
    tl.store(F_ptr + 3 * n + offs, H * ux, mask=mask)
    tl.store(F_ptr + 4 * n + offs, H * uy, mask=mask)
    _count_store(cnt_ptr, rho, p, rho_min, p_min, mask, COUNT)


@triton.jit
def _euler3d_flux_kernel(U_ptr, F_ptr, cnt_ptr, n, P_ptr,
                         COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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
    H = E + p
    tl.store(F_ptr + offs, mx * ux + p, mask=mask)
    tl.store(F_ptr + n + offs, my * uy + p, mask=mask)
    tl.store(F_ptr + 2 * n + offs, mz * uz + p, mask=mask)
    tl.store(F_ptr + 3 * n + offs, mx * uy, mask=mask)
    tl.store(F_ptr + 4 * n + offs, mx * uz, mask=mask)
    tl.store(F_ptr + 5 * n + offs, my * uz, mask=mask)
    tl.store(F_ptr + 6 * n + offs, H * ux, mask=mask)
    tl.store(F_ptr + 7 * n + offs, H * uy, mask=mask)
    tl.store(F_ptr + 8 * n + offs, H * uz, mask=mask)
    _count_store(cnt_ptr, rho, p, rho_min, p_min, mask, COUNT)


@triton.jit
def _mhd3d_flux_kernel(U_ptr, F_ptr, cnt_ptr, n, P_ptr,
                       COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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
    b2h = 0.5 * (Bx * Bx + By * By + Bz * Bz)
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy + uz * uz) - b2h)
    pstar = p + b2h
    uB = ux * Bx + uy * By + uz * Bz
    Ep = E + pstar
    tl.store(F_ptr + offs, mx * ux + pstar - Bx * Bx, mask=mask)
    tl.store(F_ptr + n + offs, my * uy + pstar - By * By, mask=mask)
    tl.store(F_ptr + 2 * n + offs, mz * uz + pstar - Bz * Bz, mask=mask)
    tl.store(F_ptr + 3 * n + offs, mx * uy - Bx * By, mask=mask)
    tl.store(F_ptr + 4 * n + offs, mx * uz - Bx * Bz, mask=mask)
    tl.store(F_ptr + 5 * n + offs, my * uz - By * Bz, mask=mask)
    tl.store(F_ptr + 6 * n + offs, Ep * ux - Bx * uB, mask=mask)
    tl.store(F_ptr + 7 * n + offs, Ep * uy - By * uB, mask=mask)
    tl.store(F_ptr + 8 * n + offs, Ep * uz - Bz * uB, mask=mask)
    tl.store(F_ptr + 9 * n + offs, ux * By - uy * Bx, mask=mask)
    tl.store(F_ptr + 10 * n + offs, ux * Bz - uz * Bx, mask=mask)
    tl.store(F_ptr + 11 * n + offs, uy * Bz - uz * By, mask=mask)
    _count_store(cnt_ptr, rho, p, rho_min, p_min, mask, COUNT)


_FLUX_KERNELS = {"euler2d": (_euler2d_flux_kernel, 5), "euler3d": (_euler3d_flux_kernel, 9),
                 "mhd3d": (_mhd3d_flux_kernel, 12)}


def fluxes(kind, U, gamma, rho_min=0.0, p_min=0.0, count=False, block=1024, num_warps=8):
    """Flux fields of ``U`` (contiguous); with ``count``, also the floor-violation count."""
    kernel, nflux = _FLUX_KERNELS[kind]
    U = U.contiguous()
    n = U[0].numel()
    F = torch.empty((nflux,) + tuple(U.shape[1:]), dtype=U.dtype, device=U.device)
    cnt = torch.zeros((), dtype=torch.int64, device=U.device)
    # Python floats would reach Triton as fp32 scalars; load them in U's dtype.
    params = torch.tensor([gamma, rho_min, p_min], dtype=U.dtype, device=U.device)
    kernel[(triton.cdiv(n, block),)](U, F, cnt, n, params,
                                     COUNT=count, BLOCK=block, num_warps=num_warps)
    return (F, cnt) if count else F
