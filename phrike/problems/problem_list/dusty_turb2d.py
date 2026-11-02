"""Decaying 2D gas turbulence with dust and an active SGS reservoir."""

import numpy as np

from .dusty_box import DustyBoxProblem


def turbulent_velocity_2d(X, Y, vrms=0.1, kmin=1.0, kmax=4.0, seed=42):
    """Seeded, band-limited solenoidal field with physical-wavevector projection."""
    ny, nx = X.shape
    if not 0 <= kmin < kmax or vrms < 0:
        raise ValueError("require 0 <= kmin < kmax and vrms >= 0")
    mx = np.fft.fftfreq(nx, d=1/nx)[None, :]
    my = np.fft.fftfreq(ny, d=1/ny)[:, None]
    kmag = np.sqrt(mx*mx + my*my)
    kx = 2*np.pi*mx/(nx*(X[0, 1]-X[0, 0]))
    ky = 2*np.pi*my/(ny*(Y[1, 0]-Y[0, 0]))
    k2 = kx*kx + ky*ky
    rng = np.random.default_rng(seed)
    U, V = np.fft.fft2(rng.normal(size=(2, ny, nx)), axes=(-2, -1))
    dot = (kx*U + ky*V)/np.where(k2 > 0, k2, 1.0)
    envelope = np.sqrt(np.maximum((kmag-kmin)*(kmax-kmag), 0))
    envelope *= (kmag > kmin) & (kmag < kmax)
    # Drop Nyquist modes so the real inverse remains exactly solenoidal.
    if nx % 2 == 0:
        envelope[:, nx//2] = 0
    if ny % 2 == 0:
        envelope[ny//2] = 0
    u = np.fft.ifft2((U-kx*dot)*envelope).real
    v = np.fft.ifft2((V-ky*dot)*envelope).real
    rms = np.sqrt(np.mean(u*u+v*v))
    if rms == 0 and vrms > 0:
        raise ValueError("turbulence band contains no nonzero modes")
    return (u*vrms/rms, v*vrms/rms) if rms > 0 else (u, v)


class DustyTurb2DProblem(DustyBoxProblem):
    def create_initial_conditions(self, grid):
        if self.ndim != 2:
            raise ValueError("dusty_turb2d requires a 2D grid")
        ic = self.config.get("initial_conditions", {})
        X, Y = grid.xy_mesh()
        u, v = turbulent_velocity_2d(X, Y, float(ic.get("vrms", 0.1)),
                                    float(ic.get("kmin", 1)), float(ic.get("kmax", 4)),
                                    int(ic.get("seed", 42)))
        rho = np.full_like(u, float(ic.get("rho0", 1.0)))
        p = np.full_like(u, float(ic.get("p0", 1.0)))
        return self.create_equations().conservative(rho, u, v, p)
