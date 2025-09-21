"""Gradient-based artificial viscosity module for pseudo-spectral solvers.

This module provides artificial viscosity capabilities for stabilizing shocks and
steep gradients in pseudo-spectral CFD simulations. It uses gradient-based
smoothness sensors to apply viscosity only where needed, preserving smooth regions.

Key Features:
- Multi-dimensional support (1D, 2D, 3D)
- Gradient-based smoothness sensing
- Tunable viscosity parameters
- Spectral differentiation for consistency
- Periodic boundary condition support
- High-order time-stepping compatibility
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):  # type: ignore
        def inner(func):
            return func
        return inner
    
    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


@dataclass
class ArtificialViscosityConfig:
    """Configuration for artificial viscosity parameters.
    
    Attributes:
        enabled: Whether artificial viscosity is enabled
        mode: Viscosity mode - "sensor" for gradient-based or "constant" for uniform
        nu_constant: Constant viscosity coefficient (used when mode="constant")
        nu_max: Maximum viscosity coefficient (used when mode="sensor")
        s_ref: Reference smoothness threshold (used when mode="sensor")
        s_min: Minimum smoothness for applying viscosity (used when mode="sensor")
        p: Exponent for smoothness scaling (used when mode="sensor")
        epsilon: Small number to avoid division by zero (used when mode="sensor")
        variable_weights: Weights for different conserved variables
        sensor_variable: Which variable to use for smoothness sensing (used when mode="sensor")
        diagnostic_output: Whether to output diagnostic information
    """
    enabled: bool = True
    mode: str = "sensor"  # "sensor" or "constant"
    nu_constant: float = 0.0  # used when mode == "constant"
    nu_max: float = 1e-3
    s_ref: float = 1.0
    s_min: float = 0.1
    p: float = 2.0
    epsilon: float = 1e-12
    variable_weights: Dict[str, float] = None
    sensor_variable: str = "density"  # "density", "pressure", "velocity_magnitude"
    diagnostic_output: bool = False
    
    def __post_init__(self):
        if self.variable_weights is None:
            self.variable_weights = {
                "density": 1.0,
                "momentum_x": 1.0,
                "momentum_y": 1.0,
                "momentum_z": 1.0,
                "energy": 1.0
            }
        # Normalize mode value
        if isinstance(self.mode, str):
            self.mode = self.mode.lower().strip()
        else:
            self.mode = "sensor"
        
        # Validate mode
        if self.mode not in ["sensor", "constant"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'sensor' or 'constant'")
        
        # Validate nu_constant for constant mode
        if self.mode == "constant" and self.nu_constant < 0:
            raise ValueError("nu_constant must be non-negative")


class SmoothnessSensor(ABC):
    """Abstract base class for smoothness sensors."""
    
    @abstractmethod
    def compute_sensor(self, U: np.ndarray, grid, equations) -> np.ndarray:
        """Compute smoothness sensor for given solution state.
        
        Args:
            U: Conservative variables array
            grid: Grid object with differentiation methods
            equations: Equations object with primitive variable methods
            
        Returns:
            Smoothness sensor array (same shape as U[0])
        """
        pass


class GradientBasedSensor(SmoothnessSensor):
    """Gradient-based smoothness sensor using spectral derivatives.
    
    Computes smoothness as s = |∇q| / (|q| + ε) where q is the sensor variable.
    """
    
    def __init__(self, config: ArtificialViscosityConfig):
        self.config = config
    
    def compute_sensor(self, U: np.ndarray, grid, equations) -> np.ndarray:
        """Compute gradient-based smoothness sensor."""
        # Get the sensor variable
        if self.config.sensor_variable == "density":
            q = U[0]  # density
        elif self.config.sensor_variable == "pressure":
            _, _, p, _ = equations.primitive(U)
            q = p
        elif self.config.sensor_variable == "velocity_magnitude":
            rho, u, p, _ = equations.primitive(U)
            if len(u.shape) == 1:  # 1D
                q = np.abs(u)
            elif len(u.shape) == 2:  # 2D
                q = np.sqrt(u[0]**2 + u[1]**2)
            else:  # 3D
                q = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
        else:
            raise ValueError(f"Unknown sensor variable: {self.config.sensor_variable}")
        
        # Compute gradient magnitude
        grad_magnitude = self._compute_gradient_magnitude(q, grid)
        
        # Torch-aware math to avoid implicit numpy conversions on device tensors
        is_torch = False
        try:
            import torch  # type: ignore
            is_torch = isinstance(q, torch.Tensor)
        except Exception:
            torch = None  # type: ignore
            is_torch = False

        if is_torch:
            eps = torch.tensor(float(self.config.epsilon), dtype=q.dtype, device=q.device)
            sensor = grad_magnitude / (torch.abs(q) + eps)
            if self.config.s_ref > 0:
                sref = torch.tensor(float(self.config.s_ref), dtype=q.dtype, device=q.device)
                sensor = torch.clamp(sensor / sref, 0.0, 1.0)
            return sensor
        else:
            sensor = grad_magnitude / (np.abs(q) + self.config.epsilon)
            if self.config.s_ref > 0:
                sensor = np.clip(sensor / self.config.s_ref, 0.0, 1.0)
            return sensor
    
    def _compute_gradient_magnitude(self, q: np.ndarray, grid) -> np.ndarray:
        """Compute |∇q| using spectral differentiation."""
        # Determine if we are working with torch tensors
        is_torch = False
        try:
            import torch  # type: ignore
            is_torch = isinstance(q, torch.Tensor)
        except Exception:
            torch = None  # type: ignore
            is_torch = False

        if hasattr(grid, 'dx1') and not hasattr(grid, 'dy1') and not hasattr(grid, 'dz1'):
            # 1D
            dqdx = grid.dx1(q)
            return (torch.abs(dqdx) if is_torch else np.abs(dqdx))
        elif hasattr(grid, 'dx1') and hasattr(grid, 'dy1') and not hasattr(grid, 'dz1'):
            # 2D
            dqdx = grid.dx1(q)
            dqdy = grid.dy1(q)
            if is_torch:
                return torch.sqrt(dqdx * dqdx + dqdy * dqdy)
            return np.sqrt(dqdx**2 + dqdy**2)
        elif hasattr(grid, 'dx1') and hasattr(grid, 'dy1') and hasattr(grid, 'dz1'):
            # 3D
            dqdx = grid.dx1(q)
            dqdy = grid.dy1(q)
            dqdz = grid.dz1(q)
            if is_torch:
                return torch.sqrt(dqdx * dqdx + dqdy * dqdy + dqdz * dqdz)
            return np.sqrt(dqdx**2 + dqdy**2 + dqdz**2)
        else:
            raise ValueError("Grid does not support required differentiation methods")


class ArtificialViscosityCoefficient:
    """Computes artificial viscosity coefficient based on smoothness sensor."""
    
    def __init__(self, config: ArtificialViscosityConfig):
        self.config = config
    
    def compute_viscosity(self, sensor: np.ndarray) -> np.ndarray:
        """Compute viscosity coefficient from smoothness sensor.
        
        Args:
            sensor: Smoothness sensor array
            
        Returns:
            Viscosity coefficient array
        """
        # Torch-aware path
        is_torch = False
        try:
            import torch  # type: ignore
            is_torch = isinstance(sensor, torch.Tensor)
        except Exception:
            torch = None  # type: ignore
            is_torch = False

        if is_torch:
            s_min = torch.tensor(float(self.config.s_min), dtype=sensor.dtype, device=sensor.device)
            active_sensor = torch.where(sensor > s_min, sensor, torch.zeros((), dtype=sensor.dtype, device=sensor.device))
            if self.config.s_ref > 0:
                s_ref = torch.tensor(float(self.config.s_ref), dtype=sensor.dtype, device=sensor.device)
                scaled_sensor = active_sensor / s_ref
            else:
                scaled_sensor = active_sensor
            nu_max = torch.tensor(float(self.config.nu_max), dtype=sensor.dtype, device=sensor.device)
            viscosity = nu_max * (scaled_sensor ** float(self.config.p))
            return viscosity
        else:
            active_sensor = np.where(sensor > self.config.s_min, sensor, 0.0)
            if self.config.s_ref > 0:
                scaled_sensor = active_sensor / self.config.s_ref
            else:
                scaled_sensor = active_sensor
            viscosity = self.config.nu_max * (scaled_sensor ** self.config.p)
            return viscosity


class SpectralArtificialViscosity:
    """Main artificial viscosity class for pseudo-spectral solvers.
    
    This class provides the complete artificial viscosity implementation
    for 1D, 2D, and 3D pseudo-spectral CFD simulations.
    """
    
    def __init__(self, config: ArtificialViscosityConfig):
        """Initialize artificial viscosity module.
        
        Args:
            config: Artificial viscosity configuration
        """
        self.config = config
        self.sensor = GradientBasedSensor(config)
        self.viscosity_coeff = ArtificialViscosityCoefficient(config)
        
        # Diagnostic storage
        self.last_sensor = None
        self.last_viscosity = None
        self.last_viscosity_terms = None
    
    def compute_viscosity_terms(self, U: np.ndarray, grid, equations) -> List[np.ndarray]:
        """Compute artificial viscosity terms for all conserved variables.
        
        Args:
            U: Conservative variables array
            grid: Grid object with differentiation methods
            equations: Equations object
            
        Returns:
            List of viscosity terms for each conserved variable
        """
        if not self.config.enabled:
            return [np.zeros_like(U[i]) for i in range(len(U))]
        
        # Compute viscosity field
        if self.config.mode == "constant":
            # Use a uniform constant viscosity everywhere
            nu_val = float(self.config.nu_constant)
            # If nu_constant is 0, fall back to nu_max for backward compatibility
            if nu_val == 0.0:
                nu_val = float(self.config.nu_max)
            # Preserve tensor type/device if U is a torch tensor
            try:
                import torch  # type: ignore
                if isinstance(U, torch.Tensor) or (isinstance(U, (list, tuple)) and len(U) > 0 and hasattr(U[0], 'device')):
                    base = U if isinstance(U, torch.Tensor) else U[0]
                    viscosity = torch.full_like(base, nu_val)
                else:
                    viscosity = np.full_like(U[0], nu_val)
            except Exception:
                viscosity = np.full_like(U[0], nu_val)
            sensor = None
        else:
            # Sensor-based viscosity
            sensor = self.sensor.compute_sensor(U, grid, equations)
            viscosity = self.viscosity_coeff.compute_viscosity(sensor)
        
        # Compute viscosity terms for each conserved variable
        viscosity_terms = []
        variable_names = ["density", "momentum_x", "momentum_y", "momentum_z", "energy"]
        
        for i, var_name in enumerate(variable_names[:len(U)]):
            if i < len(U):
                # Get weight for this variable
                weight = self.config.variable_weights.get(var_name, 1.0)
                
                # Compute ∇·(ν ∇U[i])
                viscosity_term = self._compute_viscosity_term(U[i], viscosity * weight, grid)
                viscosity_terms.append(viscosity_term)
            else:
                viscosity_terms.append(np.zeros_like(U[0]))
        
        # Store diagnostics
        if self.config.diagnostic_output:
            # Store CPU copies for diagnostics to avoid device constraints
            try:
                import torch  # type: ignore
                if sensor is None:
                    self.last_sensor = None
                else:
                    self.last_sensor = sensor.detach().cpu().clone().numpy() if isinstance(sensor, torch.Tensor) else sensor.copy()
                self.last_viscosity = viscosity.detach().cpu().clone().numpy() if isinstance(viscosity, torch.Tensor) else viscosity.copy()
                self.last_viscosity_terms = [
                    term.detach().cpu().clone().numpy() if isinstance(term, torch.Tensor) else term.copy()
                    for term in viscosity_terms
                ]
            except Exception:
                self.last_sensor = None if sensor is None else sensor.copy()
                self.last_viscosity = viscosity.copy()
                self.last_viscosity_terms = [term.copy() for term in viscosity_terms]
        
        return viscosity_terms
    
    def _compute_viscosity_term(self, U_component: np.ndarray, viscosity: np.ndarray, grid) -> np.ndarray:
        """Compute ∇·(ν ∇U) for a single component using spectral differentiation.
        
        Args:
            U_component: Single component of conservative variables
            viscosity: Viscosity coefficient array
            grid: Grid object with differentiation methods
            
        Returns:
            Viscosity term ∇·(ν ∇U)
        """
        if hasattr(grid, 'dx1'):  # 1D
            return self._compute_viscosity_term_1d(U_component, viscosity, grid)
        elif hasattr(grid, 'dx1') and hasattr(grid, 'dy1'):  # 2D
            return self._compute_viscosity_term_2d(U_component, viscosity, grid)
        elif hasattr(grid, 'dx1') and hasattr(grid, 'dy1') and hasattr(grid, 'dz1'):  # 3D
            return self._compute_viscosity_term_3d(U_component, viscosity, grid)
        else:
            raise ValueError("Grid does not support required differentiation methods")
    
    def _compute_viscosity_term_1d(self, U: np.ndarray, viscosity: np.ndarray, grid) -> np.ndarray:
        """Compute ∇·(ν ∇U) in 1D."""
        # Compute ∇U
        dUdx = grid.dx1(U)
        
        # Compute ν ∇U
        nu_grad_U = viscosity * dUdx
        
        # Compute ∇·(ν ∇U)
        return grid.dx1(nu_grad_U)
    
    def _compute_viscosity_term_2d(self, U: np.ndarray, viscosity: np.ndarray, grid) -> np.ndarray:
        """Compute ∇·(ν ∇U) in 2D."""
        # Compute ∇U
        dUdx = grid.dx1(U)
        dUdy = grid.dy1(U)
        
        # Compute ν ∇U components
        nu_grad_Ux = viscosity * dUdx
        nu_grad_Uy = viscosity * dUdy
        
        # Compute ∇·(ν ∇U) = ∂/∂x(ν ∂U/∂x) + ∂/∂y(ν ∂U/∂y)
        return grid.dx1(nu_grad_Ux) + grid.dy1(nu_grad_Uy)
    
    def _compute_viscosity_term_3d(self, U: np.ndarray, viscosity: np.ndarray, grid) -> np.ndarray:
        """Compute ∇·(ν ∇U) in 3D."""
        # Compute ∇U
        dUdx = grid.dx1(U)
        dUdy = grid.dy1(U)
        dUdz = grid.dz1(U)
        
        # Compute ν ∇U components
        nu_grad_Ux = viscosity * dUdx
        nu_grad_Uy = viscosity * dUdy
        nu_grad_Uz = viscosity * dUdz
        
        # Compute ∇·(ν ∇U) = ∂/∂x(ν ∂U/∂x) + ∂/∂y(ν ∂U/∂y) + ∂/∂z(ν ∂U/∂z)
        return grid.dx1(nu_grad_Ux) + grid.dy1(nu_grad_Uy) + grid.dz1(nu_grad_Uz)
    
    def get_diagnostics(self) -> Dict[str, np.ndarray]:
        """Get diagnostic information about the last computation.
        
        Returns:
            Dictionary containing sensor, viscosity, and terms
        """
        if not self.config.diagnostic_output:
            return {}
        
        return {
            "sensor": self.last_sensor,
            "viscosity": self.last_viscosity,
            "viscosity_terms": self.last_viscosity_terms
        }


# Numba-accelerated kernels for performance
@njit(cache=True, fastmath=True)
def _compute_gradient_magnitude_1d_numba(q: np.ndarray, dqdx: np.ndarray) -> np.ndarray:
    """Numba-accelerated gradient magnitude computation for 1D."""
    return np.abs(dqdx)


@njit(cache=True, fastmath=True)
def _compute_gradient_magnitude_2d_numba(dqdx: np.ndarray, dqdy: np.ndarray) -> np.ndarray:
    """Numba-accelerated gradient magnitude computation for 2D."""
    return np.sqrt(dqdx**2 + dqdy**2)


@njit(cache=True, fastmath=True)
def _compute_gradient_magnitude_3d_numba(dqdx: np.ndarray, dqdy: np.ndarray, dqdz: np.ndarray) -> np.ndarray:
    """Numba-accelerated gradient magnitude computation for 3D."""
    return np.sqrt(dqdx**2 + dqdy**2 + dqdz**2)


@njit(cache=True, fastmath=True)
def _compute_smoothness_sensor_numba(grad_magnitude: np.ndarray, q: np.ndarray, epsilon: float) -> np.ndarray:
    """Numba-accelerated smoothness sensor computation."""
    return grad_magnitude / (np.abs(q) + epsilon)


@njit(cache=True, fastmath=True)
def _compute_viscosity_coefficient_numba(sensor: np.ndarray, s_min: float, nu_max: float, 
                                       s_ref: float, p: float) -> np.ndarray:
    """Numba-accelerated viscosity coefficient computation."""
    # Apply minimum threshold
    active_sensor = np.where(sensor > s_min, sensor, 0.0)
    
    # Compute viscosity: nu = nu_max * (s / s_ref)^p
    if s_ref > 0:
        scaled_sensor = active_sensor / s_ref
    else:
        scaled_sensor = active_sensor
    
    return nu_max * (scaled_sensor ** p)


def create_artificial_viscosity(config_dict: Dict) -> SpectralArtificialViscosity:
    """Factory function to create artificial viscosity module from configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured SpectralArtificialViscosity instance
    """
    config = ArtificialViscosityConfig(**config_dict)
    return SpectralArtificialViscosity(config)


def create_default_config() -> ArtificialViscosityConfig:
    """Create default artificial viscosity configuration.
    
    Returns:
        Default configuration
    """
    return ArtificialViscosityConfig()
