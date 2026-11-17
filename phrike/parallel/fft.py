"""Truncated real-to-complex 3D FFT on a slab decomposition (Torch tensors).

``forward`` maps a real field on this rank's z slab, ``(C, nz, Ny, Nx)``, to
the retained Fourier modes of this rank's ky slab, ``(C, Kz, nky, Kx)``;
``inverse`` goes back.  The transforms act on the retained modes only (a
decomposition built with ``full=True`` keeps every mode instead, which makes
this the plain distributed ``rfftn``/``irfftn``):

* forward: 2D r2c in (y, x) per plane, drop the modes outside the 2/3 box in
  x and y, transpose, 1D FFT in z, drop the z modes outside the box;
* inverse: zero-pad and inverse FFT in z, transpose, zero-pad in y and x,
  2D c2r in (y, x).

Truncating before the transpose (and padding after it) means each transform
communicates 4/9 of the grid's data instead of all of it, and exactly once.
Every rank sends its whole local block once per transform, whatever the rank
count, so the traffic per rank is constant under weak scaling.

``forward`` is the unnormalized DFT (pass ``normalized=True`` for the 1/N
convention the solvers store); ``inverse`` is the plain sum over modes, so
``inverse(forward(f, normalized=True))`` returns the dealiased ``f``.

Packing and unpacking are a fixed number of strided copies: the balanced slab
sizes form at most two uniform groups (``decomp.uniform_groups``), and each
block is laid out so that the receiver's buffer is already the array the next
1D or 2D transform needs.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .comm import Comm, Timers
from .decomp import SlabDecomposition, uniform_groups


class SlabFFT:
    """Distributed truncated rfftn/irfftn for one ``SlabDecomposition``."""

    def __init__(self, decomp: SlabDecomposition, comm: Comm, device="cpu",
                 rdtype=None, batch: Optional[int] = 4, timers: Optional[Timers] = None):
        """``batch``: fields transformed (and exchanged) per pass; ``0`` or
        ``None`` takes all of them at once.  Smaller batches cap the transient
        memory of a transform at the cost of more, smaller MPI calls."""
        if decomp.nprocs != comm.size or decomp.rank != comm.rank:
            raise ValueError("decomposition and communicator disagree on rank or size")
        self.d = decomp
        self.comm = comm
        self.device = torch.device(device)
        self.rdtype = rdtype or torch.float64
        self.cdtype = torch.complex64 if self.rdtype == torch.float32 else torch.complex128
        self.batch = int(batch) if batch else 0
        self.timers = timers if timers is not None else Timers(enabled=False)
        self.norm = 1.0 / float(decomp.n_points)
        self._ky_groups = uniform_groups(decomp.ky_counts)
        self._z_groups = uniform_groups(decomp.z_counts)
        self._bufs: Dict[str, torch.Tensor] = {}
        self.forward_fields = 0
        self.inverse_fields = 0

    # --- public -------------------------------------------------------------

    def forward(self, f: torch.Tensor, normalized: bool = False,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retained spectrum ``(C, Kz, nky, Kx)`` of real ``f`` ``(C, nz, Ny, Nx)``."""
        d = self.d
        if f.dim() != 4 or tuple(f.shape[1:]) != d.physical_shape:
            raise ValueError(f"forward expects (C, {d.nz}, {d.Ny}, {d.Nx}), got {tuple(f.shape)}")
        C = f.shape[0]
        if out is None:
            out = torch.empty((C,) + d.spectral_shape, dtype=self.cdtype, device=self.device)
        step = self.batch or C
        for c0 in range(0, C, step):
            self._forward(f[c0:c0 + step], out[c0:c0 + step])
        if normalized:
            out.mul_(self.norm)
        self.forward_fields += C
        return out

    def inverse(self, S: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Real field ``(C, nz, Ny, Nx)`` summed from the retained modes ``S``."""
        d = self.d
        if S.dim() != 4 or tuple(S.shape[1:]) != d.spectral_shape:
            raise ValueError(f"inverse expects (C, {d.Kz}, {d.nky}, {d.Kx}), got {tuple(S.shape)}")
        C = S.shape[0]
        if out is None:
            out = torch.empty((C,) + d.physical_shape, dtype=self.rdtype, device=self.device)
        step = self.batch or C
        for c0 in range(0, C, step):
            self._inverse(S[c0:c0 + step], out[c0:c0 + step])
        self.inverse_fields += C
        return out

    # --- internals ----------------------------------------------------------

    def _buffer(self, key: str, n: int) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.numel() < n:
            buf = torch.empty(n, dtype=self.cdtype, device=self.device)
            self._bufs[key] = buf
        return buf[:n]

    def release_buffers(self) -> None:
        """Drop the cached exchange buffers (they are reallocated on demand)."""
        self._bufs.clear()

    def _exchange(self, send, scounts, recv, rcounts) -> None:
        # One row is a full kx line: Kx complex values.
        self.comm.exchange(torch.view_as_real(send).reshape(-1), scounts,
                           torch.view_as_real(recv).reshape(-1), rcounts, unit=2 * self.d.Kx)

    def _forward(self, f: torch.Tensor, out: torch.Tensor) -> None:
        d = self.d
        C, nz = f.shape[0], d.nz
        with self.timers("fft_xy"):
            X = torch.fft.rfft2(f)  # (C, nz, Ny, Xh)
        with self.timers("fft_pack"):
            if d.Ky == d.Ny and d.Kx == d.Xh:
                T = X  # nothing to drop
            elif d.ty > 0:
                T = torch.cat((X[:, :, :d.hy, :d.Kx], X[:, :, d.Ny - d.ty:, :d.Kx]), dim=2)
            else:
                T = X[:, :, :d.hy, :d.Kx].contiguous()
            del X
            # Block for rank q, laid out (z, c, ky, kx): the receiver's blocks
            # then concatenate along z into (Nz, C, nky, Kx).
            send = self._buffer("send", C * nz * d.Ky * d.Kx)
            offset = 0
            for _, nblocks, size, row0 in self._ky_groups:
                n = nblocks * nz * C * size * d.Kx
                block = T[:, :, row0:row0 + nblocks * size, :].reshape(C, nz, nblocks, size, d.Kx)
                send[offset:offset + n].view(nblocks, nz, C, size, d.Kx).copy_(
                    block.permute(2, 1, 0, 3, 4))
                offset += n
            del T
            recv = self._buffer("recv", d.Nz * C * d.nky * d.Kx)
        self._exchange(send, [nz * C * c for c in d.ky_counts],
                       recv, [zc * C * d.nky for zc in d.z_counts])
        with self.timers("fft_z"):
            Y = torch.fft.fft(recv.view(d.Nz, C, d.nky, d.Kx), dim=0)
        with self.timers("fft_pack"):
            out[:, :d.hz].copy_(Y[:d.hz].permute(1, 0, 2, 3))
            if d.tz > 0:
                out[:, d.hz:].copy_(Y[d.Nz - d.tz:].permute(1, 0, 2, 3))

    def _inverse(self, S: torch.Tensor, out: torch.Tensor) -> None:
        d = self.d
        C, nky = S.shape[0], d.nky
        with self.timers("fft_pack"):
            padded = torch.zeros((C, nky, d.Kx, d.Nz), dtype=self.cdtype, device=self.device)
            Sz = S.permute(0, 2, 3, 1)  # (C, nky, Kx, Kz)
            padded[..., :d.hz] = Sz[..., :d.hz]
            if d.tz > 0:
                padded[..., d.Nz - d.tz:] = Sz[..., d.hz:]
        with self.timers("fft_z"):
            Q = torch.fft.ifft(padded, dim=-1, norm="forward")  # (C, nky, Kx, Nz), no 1/N
            del padded
        with self.timers("fft_pack"):
            # Block for rank q, laid out (ky, c, z, kx): the receiver's blocks
            # then concatenate along ky into (Ky, C, nz, Kx).
            send = self._buffer("send", C * nky * d.Kx * d.Nz)
            offset = 0
            for _, nblocks, size, z0 in self._z_groups:
                n = nblocks * nky * C * size * d.Kx
                block = Q[..., z0:z0 + nblocks * size].reshape(C, nky, d.Kx, nblocks, size)
                send[offset:offset + n].view(nblocks, nky, C, size, d.Kx).copy_(
                    block.permute(3, 1, 0, 4, 2))
                offset += n
            del Q
            recv = self._buffer("recv", d.Ky * C * d.nz * d.Kx)
        self._exchange(send, [nky * C * zc for zc in d.z_counts],
                       recv, [kc * C * d.nz for kc in d.ky_counts])
        with self.timers("fft_pack"):
            R = recv.view(d.Ky, C, d.nz, d.Kx).permute(1, 2, 0, 3)  # (C, nz, Ky, Kx)
            W = torch.zeros((C, d.nz, d.Ny, d.Xh), dtype=self.cdtype, device=self.device)
            W[:, :, :d.hy, :d.Kx] = R[:, :, :d.hy]
            if d.ty > 0:
                W[:, :, d.Ny - d.ty:, :d.Kx] = R[:, :, d.hy:]
        with self.timers("fft_xy"):
            out.copy_(torch.fft.irfft2(W, s=(d.Ny, d.Nx), norm="forward"))
