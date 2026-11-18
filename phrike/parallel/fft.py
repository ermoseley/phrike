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

Fields are transformed in batches, and with a transport that runs in the
background (``nccl``) the batches are pipelined: while one batch is on the
wire the device packs the next one and finishes the previous one, so the
exchange hides behind the transforms instead of adding to them.  Two batches
are in flight at most, each with its own pair of exchange buffers.
"""

from __future__ import annotations

import os
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import torch

from .comm import Comm, Timers
from .decomp import SlabDecomposition, uniform_groups


class SlabFFT:
    """Distributed truncated rfftn/irfftn for one ``SlabDecomposition``."""

    def __init__(self, decomp: SlabDecomposition, comm: Comm, device="cpu",
                 rdtype=None, batch: Optional[int] = 1, timers: Optional[Timers] = None,
                 overlap: Optional[bool] = None):
        """``batch``: fields transformed (and exchanged) per pass; ``0`` or
        ``None`` takes all of them at once.  Smaller batches cap the transient
        memory of a transform at the cost of more, smaller exchanges, and give
        the pipeline more to overlap.  ``overlap``: run the exchange of a batch
        behind the transforms of its neighbours (default: whenever the
        transport allows it; ``PHRIKE_FFT_OVERLAP=0`` turns it off)."""
        if decomp.nprocs != comm.size or decomp.rank != comm.rank:
            raise ValueError("decomposition and communicator disagree on rank or size")
        self.d = decomp
        self.comm = comm
        self.device = torch.device(device)
        self.rdtype = rdtype or torch.float64
        self.cdtype = torch.complex64 if self.rdtype == torch.float32 else torch.complex128
        self.batch = int(batch) if batch else 0
        if overlap is None:
            flag = os.environ.get("PHRIKE_FFT_OVERLAP", "").strip().lower()
            overlap = None if flag in ("", "auto") else flag not in ("0", "false", "no", "off")
        self.overlap = overlap  # None: decided by the transport on first use
        self.timers = timers if timers is not None else Timers(enabled=False)
        self.norm = 1.0 / float(decomp.n_points)
        self._ky_copies = self._ky_pack_plan(decomp)
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
        depth = self._depth()
        flying: List[Tuple] = []
        for i, c0 in enumerate(range(0, C, step)):
            flying.append(self._forward_begin(f[c0:c0 + step], i % depth) + (out[c0:c0 + step],))
            if len(flying) == depth:
                self._forward_end(*flying.pop(0))
        while flying:
            self._forward_end(*flying.pop(0))
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
        step = self.batch or C
        self.inverse_fields += C
        if out is None and C <= step:
            return self._inverse_end(*self._inverse_begin(S, 0), None)  # the transform's own output
        if out is None:
            out = torch.empty((C,) + d.physical_shape, dtype=self.rdtype, device=self.device)
        depth = self._depth()
        flying: List[Tuple] = []
        for i, c0 in enumerate(range(0, C, step)):
            flying.append(self._inverse_begin(S[c0:c0 + step], i % depth) + (out[c0:c0 + step],))
            if len(flying) == depth:
                self._inverse_end(*flying.pop(0))
        while flying:
            self._inverse_end(*flying.pop(0))
        return out

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _ky_pack_plan(d):
        """Strided copies from the two retained FFT bands directly into peer blocks."""
        bands = [(0, d.Ky, 0)] if d.Ky == d.Ny else [(0, d.hy, 0), (d.hy, d.Ky, d.Ny - d.Ky)]
        plan = []
        for lo, hi, shift in bands:
            spans = []
            for row, size in zip(d.ky_starts, d.ky_counts):
                start, end = max(row, lo), min(row + size, hi)
                if start < end:
                    spans.append((row, size, start - row, end - start))
            # Adjacent peers with the same layout share one copy. Only a peer
            # straddling the gap in the full FFT needs two partial writes.
            for (size, offset, width), peers in groupby(spans, key=lambda span: span[1:]):
                rows = [span[0] for span in peers]
                plan.append((rows[0], len(rows), size, offset, width, rows[0] + offset + shift))
        return plan

    def _buffer(self, key: str, n: int) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.numel() < n:
            buf = torch.empty(n, dtype=self.cdtype, device=self.device)
            self._bufs[key] = buf
        return buf[:n]

    def release_buffers(self) -> None:
        """Drop the cached exchange buffers (they are reallocated on demand)."""
        self._bufs.clear()

    def _depth(self) -> int:
        """Batches in flight at once: 2 when the exchange runs in the background, else 1."""
        if self.overlap is False or self.comm.size == 1:
            return 1
        return 2 if self.comm.can_overlap(self.device) else 1

    def _exchange(self, send, scounts, recv, rcounts):
        # One row is a full kx line: Kx complex values.
        return self.comm.exchange_begin(torch.view_as_real(send).reshape(-1), scounts,
                                        torch.view_as_real(recv).reshape(-1), rcounts,
                                        unit=2 * self.d.Kx, overlap=self.overlap is not False)

    def _forward_begin(self, f: torch.Tensor, slot: int):
        """2D transform, truncation and packing of one batch; starts its exchange."""
        d = self.d
        C, nz = f.shape[0], d.nz
        with self.timers("fft_xy"):
            X = torch.fft.rfft2(f)  # (C, nz, Ny, Xh)
        with self.timers("fft_pack"):
            # Block for rank q, laid out (z, c, ky, kx): the receiver's blocks
            # then concatenate along z into (Nz, C, nky, Kx).
            send = self._buffer(f"send{slot}", C * nz * d.Ky * d.Kx)
            rows = send.view(d.Ky, nz * C * d.Kx)
            for row, nblocks, size, offset, width, source in self._ky_copies:
                target = rows[row:row + nblocks * size].view(nblocks, nz, C, size, d.Kx)
                band = X[:, :, source:source + nblocks * width, :d.Kx].view(C, nz, nblocks, width, d.Kx)
                target[:, :, :, offset:offset + width].copy_(band.permute(2, 1, 0, 3, 4))
            del band, X
            # A single rank already packed its own complete receive block.
            recv = send if self.comm.size == 1 else self._buffer(f"recv{slot}", d.Nz * C * d.nky * d.Kx)
        handle = self._exchange(send, [nz * C * c for c in d.ky_counts],
                                recv, [zc * C * d.nky for zc in d.z_counts])
        return handle, recv, C

    def _forward_end(self, handle, recv: torch.Tensor, C: int, out: torch.Tensor) -> None:
        """After the exchange of a batch: 1D transform in z and truncation into ``out``."""
        d = self.d
        handle.wait()
        with self.timers("fft_z"):
            Y = torch.fft.fft(recv.view(d.Nz, C, d.nky, d.Kx), dim=0)
        with self.timers("fft_pack"):
            out[:, :d.hz].copy_(Y[:d.hz].permute(1, 0, 2, 3))
            if d.tz > 0:
                out[:, d.hz:].copy_(Y[d.Nz - d.tz:].permute(1, 0, 2, 3))

    def _inverse_begin(self, S: torch.Tensor, slot: int):
        """Padding and inverse transform in z of one batch, packing; starts its exchange."""
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
            send = self._buffer(f"send{slot}", C * nky * d.Kx * d.Nz)
            offset = 0
            for _, nblocks, size, z0 in self._z_groups:
                n = nblocks * nky * C * size * d.Kx
                block = Q[..., z0:z0 + nblocks * size].reshape(C, nky, d.Kx, nblocks, size)
                send[offset:offset + n].view(nblocks, nky, C, size, d.Kx).copy_(
                    block.permute(3, 1, 0, 4, 2))
                offset += n
            del Q
            recv = send if self.comm.size == 1 else self._buffer(f"recv{slot}", d.Ky * C * d.nz * d.Kx)
        handle = self._exchange(send, [nky * C * zc for zc in d.z_counts],
                                recv, [kc * C * d.nz for kc in d.ky_counts])
        return handle, recv, C

    def _inverse_end(self, handle, recv: torch.Tensor, C: int,
                     out: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """After the exchange of a batch: padding and 2D transform into ``out`` (or returned)."""
        d = self.d
        handle.wait()
        with self.timers("fft_pack"):
            R = recv.view(d.Ky, C, d.nz, d.Kx).permute(1, 2, 0, 3)  # (C, nz, Ky, Kx)
            W = torch.zeros((C, d.nz, d.Ny, d.Xh), dtype=self.cdtype, device=self.device)
            W[:, :, :d.hy, :d.Kx] = R[:, :, :d.hy]
            if d.ty > 0:
                W[:, :, d.Ny - d.ty:, :d.Kx] = R[:, :, d.hy:]
        with self.timers("fft_xy"):
            result = torch.fft.irfft2(W, s=(d.Ny, d.Nx), norm="forward")
            if out is None:
                return result
            out.copy_(result)
        return None
