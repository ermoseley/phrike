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
    HAS_BASE: tl.constexpr, MASK: tl.constexpr, BLOCK: tl.constexpr,
):
    """O = [f *] ([B +] sum_j w[w0 + j] * K[j]) over the visited modes."""
    offs, mask, ix, iy, iz = _mode_index(BLOCK, total, Kx, Kyr, Kzr, cy, cz, Ny, Nz)
    if MASK:
        m = _factor(fx_ptr, fy_ptr, fz_ptr, ix, iy, iz, mask)
    offS = iz * sSz + iy * sSy + 2 * ix
    offK = 2 * offs
    for c in tl.static_range(5):
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
        HAS_BASE=base is not None, MASK=masked, BLOCK=block, num_warps=num_warps,
    )
