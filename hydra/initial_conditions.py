from __future__ import annotations

from typing import Dict

import numpy as np

from .equations import EulerEquations1D, EulerEquations2D, EulerEquations3D


def sod_shock_tube(
    x: np.ndarray,
    x0: float,
    left: Dict[str, float],
    right: Dict[str, float],
    gamma: float,
) -> np.ndarray:
    """Return conservative state U for the Sod shock tube initial condition.

    left/right dictionaries should include keys: rho, u, p
    """
    rho = np.where(x < x0, left["rho"], right["rho"])  # type: ignore[index]
    u = np.where(x < x0, left["u"], right["u"])  # type: ignore[index]
    p = np.where(x < x0, left["p"], right["p"])  # type: ignore[index]
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


def sinusoidal_density(
    x: np.ndarray,
    rho0: float = 1.0,
    u0: float = 0.0,
    p0: float = 1.0,
    amplitude: float = 1e-3,
    k: int = 1,
    Lx: float = 1.0,
    gamma: float = 1.4,
) -> np.ndarray:
    rho = rho0 * (1.0 + amplitude * np.sin(2.0 * np.pi * k * x / Lx))
    u = u0 * np.ones_like(x)
    p = p0 * np.ones_like(x)
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


def kelvin_helmholtz_2d(
    X: np.ndarray,
    Y: np.ndarray,
    rho_outer: float = 1.0,
    rho_inner: float = 1.0,
    u0: float = 1.0,
    shear_thickness: float = 0.02,
    pressure_outer: float = 1.0,
    pressure_inner: float = 1.0,
    perturb_eps: float = 0.01,
    perturb_sigma: float = 0.02,
    perturb_kx: int = 2,
    gamma: float = 1.4,
) -> np.ndarray:
    """2D Kelvin-Helmholtz initial condition matching RAMSES khi.py profile.

    Creates two shear layers at y=0.25 and y=0.75 with:
    - Velocity: outer regions at -0.5*u0, middle region at +0.5*u0
    - Density/pressure: can differ between outer and inner regions
    - Perturbation: Gaussian-modulated sinusoidal vy at both interfaces
    """
    # Check if we're using torch tensors
    try:
        import torch  # type: ignore
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore
    
    is_torch = _TORCH_AVAILABLE and isinstance(X, (torch.Tensor,))
    
    if is_torch:
        # Normalize coordinates to [0,1) like RAMSES script
        Lx = X.max() - X.min()
        Ly = Y.max() - Y.min()
        yn = Y / Ly  # normalized y coordinates
        
        # Velocity profile: two tanh transitions at y=0.25 and y=0.75
        U1 = -0.5 * u0  # outer regions
        U2 = +0.5 * u0  # middle region
        a = float(shear_thickness)
        tanh_low = torch.tanh((yn - 0.25) / a)
        tanh_high = torch.tanh((yn - 0.75) / a)
        ux = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)
        
        # Density profile: same tanh structure
        rho_outer_val = float(rho_outer)
        rho_inner_val = float(rho_inner)
        rho = rho_outer_val + 0.5 * (rho_inner_val - rho_outer_val) * (tanh_low - tanh_high)
        
        # Pressure profile: same tanh structure
        p0 = float(pressure_outer)
        p_inner = float(pressure_inner)
        p = p0 + 0.5 * (p_inner - p0) * (tanh_low - tanh_high)
        
        # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
        sig = float(perturb_sigma)
        gauss = (torch.exp(-((yn - 0.25) ** 2) / (2.0 * sig * sig)) + 
                 torch.exp(-((yn - 0.75) ** 2) / (2.0 * sig * sig)))
        sinus = torch.sin(2.0 * torch.pi * perturb_kx * X / Lx)
        uy = perturb_eps * sinus * gauss
    else:
        # Normalize coordinates to [0,1) like RAMSES script
        Lx = X.max() - X.min()
        Ly = Y.max() - Y.min()
        yn = Y / Ly  # normalized y coordinates
        
        # Velocity profile: two tanh transitions at y=0.25 and y=0.75
        U1 = -0.5 * u0  # outer regions
        U2 = +0.5 * u0  # middle region
        a = float(shear_thickness)
        tanh_low = np.tanh((yn - 0.25) / a)
        tanh_high = np.tanh((yn - 0.75) / a)
        ux = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)
        
        # Density profile: same tanh structure
        rho_outer_val = float(rho_outer)
        rho_inner_val = float(rho_inner)
        rho = rho_outer_val + 0.5 * (rho_inner_val - rho_outer_val) * (tanh_low - tanh_high)
        
        # Pressure profile: same tanh structure
        p0 = float(pressure_outer)
        p_inner = float(pressure_inner)
        p = p0 + 0.5 * (p_inner - p0) * (tanh_low - tanh_high)
        
        # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
        sig = float(perturb_sigma)
        gauss = (np.exp(-((yn - 0.25) ** 2) / (2.0 * sig * sig)) + 
                 np.exp(-((yn - 0.75) ** 2) / (2.0 * sig * sig)))
        sinus = np.sin(2.0 * np.pi * perturb_kx * X / Lx)
        uy = perturb_eps * sinus * gauss
    
    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, p)


def taylor_green_vortex_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    rho0: float = 1.0,
    p0: float = 100.0,
    U0: float = 1.0,
    k: int = 1,
    gamma: float = 1.4,
) -> np.ndarray:
    """3D Taylor-Green vortex initial condition (compressible variant).

    Velocity field:
        u =  U0 * sin(k x) cos(k y) cos(k z)
        v = -U0 * cos(k x) sin(k y) cos(k z)
        w =  0
    Constant density rho0 and pressure p0.
    """
    # Torch interop if needed
    try:
        import torch  # type: ignore
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore

    is_torch = _TORCH_AVAILABLE and any(isinstance(a, (torch.Tensor,)) for a in (X, Y, Z))

    if is_torch:
        assert torch is not None
        ux = U0 * torch.sin(k * X) * torch.cos(k * Y) * torch.cos(k * Z)
        uy = -U0 * torch.cos(k * X) * torch.sin(k * Y) * torch.cos(k * Z)
        uz = torch.zeros_like(X)
        rho = torch.full_like(X, float(rho0))
        p = torch.full_like(X, float(p0))
    else:
        ux = U0 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
        uy = -U0 * np.cos(k * X) * np.sin(k * Y) * np.cos(k * Z)
        uz = np.zeros_like(X)
        rho = rho0 * np.ones_like(X)
        p = p0 * np.ones_like(X)

    eqs = EulerEquations3D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p)


def turbulent_velocity_3d(
    X: np.ndarray,
    Y: np.ndarray, 
    Z: np.ndarray,
    rho0: float = 1.0,
    p0: float = 1.0,
    vrms: float = 0.1,
    kmin: float = 2.0,
    kmax: float = 16.0,
    alpha: float = 0.3333,  # Compressive fraction (0=solenoidal, 1=compressive)
    spectrum_type: str = "parabolic",  # "parabolic" or "power_law"
    power_law_slope: float = -5.0/3.0,  # Kolmogorov slope
    seed: int = 42,
    gamma: float = 1.4,
) -> np.ndarray:
    """3D turbulent velocity initial condition matching RAMSES turb.py.
    
    Generates a divergence/rotation-controlled random velocity field using 
    spectral filtering between k_min and k_max.
    
    Args:
        X, Y, Z: 3D coordinate arrays (shape: Nz, Ny, Nx)
        rho0: Uniform density
        p0: Uniform pressure  
        vrms: Target RMS velocity magnitude
        kmin: Minimum wavenumber for band-pass
        kmax: Maximum wavenumber for band-pass
        alpha: Compressive fraction [0..1] (1=compressive, 0=solenoidal)
        spectrum_type: "parabolic" (band-limited) or "power_law"
        power_law_slope: Power law slope for spectrum='power_law'
        seed: Random seed for reproducibility
        gamma: Adiabatic index
    """
    # Torch interop if needed
    try:
        import torch  # type: ignore
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore

    is_torch = _TORCH_AVAILABLE and any(isinstance(a, (torch.Tensor,)) for a in (X, Y, Z))
    
    # Get grid dimensions
    if is_torch:
        nz, ny, nx = X.shape
    else:
        nz, ny, nx = X.shape
    
    # Set up random number generator
    rng = np.random.default_rng(seed)
    
    # Generate k-grid for FFT
    def generate_k_grid(n1, n2, n3):
        kx = np.fft.fftfreq(n1, d=1.0 / n1).reshape(n1, 1, 1)
        ky = np.fft.fftfreq(n2, d=1.0 / n2).reshape(1, n2, 1) 
        kz = np.fft.fftfreq(n3, d=1.0 / n3).reshape(1, 1, n3)
        kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
        return kx, ky, kz, kmag
    
    # Generate turbulent velocity field
    def build_turbulent_velocity(n1, n2, n3, kmin, kmax, alpha, vrms, rng, spectrum_type, power_law_slope):
        # Real-space Gaussian noise
        u0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)
        v0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)
        w0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)

        # FFT to spectral domain
        U = np.fft.fftn(u0)
        V = np.fft.fftn(v0)
        W = np.fft.fftn(w0)

        # k-grid and spectral envelope
        kx, ky, kz, kmag = generate_k_grid(n1, n2, n3)
        with np.errstate(invalid="ignore"):
            band = (kmag >= kmin) & (kmag <= kmax)
            
            if spectrum_type == "parabolic":
                # Parabolic band-pass: w(k) ∝ (k - kmin)(kmax - k) within [kmin,kmax]
                envelope = (kmag - kmin) * (kmax - kmag)
                envelope = np.where(band, envelope, 0.0)
                envelope = np.clip(envelope, 0.0, None)
            elif spectrum_type == "power_law":
                # Power law: w(k) ∝ k^slope within [kmin,kmax]
                kmag_safe = np.where(kmag == 0.0, 1.0, kmag)
                envelope = np.where(band, kmag_safe ** power_law_slope, 0.0)
                # Apply smooth cutoff at boundaries
                k_transition = 0.1
                dk = kmax - kmin
                k1 = kmin + k_transition * dk
                k2 = kmax - k_transition * dk
                low_transition = np.where(
                    (kmag >= kmin) & (kmag < k1),
                    0.5 * (1.0 + np.cos(np.pi * (kmag - kmin) / (k1 - kmin))),
                    1.0
                )
                high_transition = np.where(
                    (kmag > k2) & (kmag <= kmax),
                    0.5 * (1.0 + np.cos(np.pi * (kmag - k2) / (kmax - k2))),
                    1.0
                )
                envelope *= low_transition * high_transition
            else:
                raise ValueError(f"Unknown spectrum_type: {spectrum_type}")
            
            filt = np.sqrt(envelope, dtype=np.float64)

        # Avoid division by zero at k=0
        kmag_safe = np.where(kmag == 0.0, 1.0, kmag)

        # Projection to compressive (parallel) and solenoidal (perpendicular)
        dot = (kx * U + ky * V + kz * W)
        U_par = kx * dot / (kmag_safe * kmag_safe)
        V_par = ky * dot / (kmag_safe * kmag_safe)
        W_par = kz * dot / (kmag_safe * kmag_safe)
        U_perp = U - U_par
        V_perp = V - V_par
        W_perp = W - W_par

        # Mix
        a = float(alpha)
        U_mix = a * U_par + (1.0 - a) * U_perp
        V_mix = a * V_par + (1.0 - a) * V_perp
        W_mix = a * W_par + (1.0 - a) * W_perp

        # Apply spectral envelope
        U_mix *= filt
        V_mix *= filt
        W_mix *= filt

        # Enforce zero at k=0
        U_mix[0, 0, 0] = 0.0
        V_mix[0, 0, 0] = 0.0
        W_mix[0, 0, 0] = 0.0

        # Back to real space
        u = np.fft.ifftn(U_mix).real.astype(np.float32)
        v = np.fft.ifftn(V_mix).real.astype(np.float32)
        w = np.fft.ifftn(W_mix).real.astype(np.float32)

        # Normalize to desired vrms
        speed2 = u * u + v * v + w * w
        rms = float(np.sqrt(np.mean(speed2)))
        if rms > 0:
            s = float(vrms) / rms
            u *= s
            v *= s
            w *= s

        return u, v, w
    
    # Generate velocity field
    u, v, w = build_turbulent_velocity(nx, ny, nz, kmin, kmax, alpha, vrms, rng, spectrum_type, power_law_slope)
    
    # Convert to torch if needed
    if is_torch:
        assert torch is not None
        u = torch.from_numpy(u).to(device=X.device, dtype=X.dtype)
        v = torch.from_numpy(v).to(device=X.device, dtype=X.dtype)
        w = torch.from_numpy(w).to(device=X.device, dtype=X.dtype)
        rho = torch.full_like(X, float(rho0))
        p = torch.full_like(X, float(p0))
    else:
        rho = rho0 * np.ones_like(X)
        p = p0 * np.ones_like(X)
    
    eqs = EulerEquations3D(gamma=gamma)
    return eqs.conservative(rho, u, v, w, p)
