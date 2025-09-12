"""Unit tests for the equations module.

This test suite provides comprehensive testing of the Euler equations
implementation including primitive/conservative conversions, flux calculations,
and wave speed computations.
"""

import pytest
import numpy as np
from phrike.equations import EulerEquations1D, EulerEquations2D, EulerEquations3D


class TestEulerEquations1D:
    """Test suite for 1D Euler equations."""
    
    @pytest.fixture
    def equations(self):
        """Create 1D Euler equations instance."""
        return EulerEquations1D(gamma=1.4)
    
    def test_primitive_conversion(self, equations):
        """Test conversion from conservative to primitive variables."""
        # Test case: uniform state
        U = np.array([[1.0], [0.0], [2.5]])  # rho=1, u=0, E=2.5
        rho, u, p, a = equations.primitive(U)
        
        assert np.allclose(rho, 1.0)
        assert np.allclose(u, 0.0)
        assert np.allclose(p, 1.0)  # E = p/(gamma-1) + 0.5*rho*u^2
        assert np.allclose(a, np.sqrt(1.4))  # a = sqrt(gamma*p/rho)
    
    def test_conservative_conversion(self, equations):
        """Test conversion from primitive to conservative variables."""
        rho = np.array([1.0, 2.0])
        u = np.array([0.5, -0.3])
        p = np.array([1.0, 2.0])
        
        U = equations.conservative(rho, u, p)
        
        # Check shape
        assert U.shape == (3, 2)
        
        # Check mass conservation
        assert np.allclose(U[0], rho)
        
        # Check momentum conservation
        assert np.allclose(U[1], rho * u)
        
        # Check energy conservation
        expected_E = p / (equations.gamma - 1.0) + 0.5 * rho * u**2
        assert np.allclose(U[2], expected_E)
    
    def test_round_trip_conversion(self, equations):
        """Test that primitive -> conservative -> primitive preserves values."""
        rho = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, -0.3, 1.2])
        p = np.array([1.0, 2.0, 0.8])
        
        U = equations.conservative(rho, u, p)
        rho2, u2, p2, a2 = equations.primitive(U)
        
        assert np.allclose(rho, rho2)
        assert np.allclose(u, u2)
        assert np.allclose(p, p2)
    
    def test_flux_calculation(self, equations):
        """Test flux vector calculation."""
        U = np.array([[1.0], [0.5], [2.5]])  # rho=1, mom=0.5, E=2.5
        F = equations.flux(U)
        
        # F1 = rho*u = mom
        assert np.allclose(F[0], 0.5)
        
        # F2 = rho*u^2 + p
        rho, u, p, _ = equations.primitive(U)
        expected_F2 = rho * u**2 + p
        assert np.allclose(F[1], expected_F2)
        
        # F3 = (E + p)*u
        expected_F3 = (U[2] + p) * u
        assert np.allclose(F[2], expected_F3)
    
    def test_max_wave_speed(self, equations):
        """Test maximum wave speed calculation."""
        U = np.array([[1.0], [0.0], [2.5]])  # rho=1, u=0, E=2.5
        max_speed = equations.max_wave_speed(U)
        
        rho, u, p, a = equations.primitive(U)
        expected_speed = np.abs(u) + a
        assert np.allclose(max_speed, expected_speed)
    
    def test_conserved_quantities(self, equations):
        """Test conserved quantities calculation."""
        U = np.array([[1.0, 2.0], [0.5, -0.6], [2.5, 5.0]])
        
        quantities = equations.conserved_quantities(U)
        
        assert np.isclose(quantities["mass"], 3.0)  # sum of rho
        assert np.isclose(quantities["momentum"], -0.1)  # sum of rho*u
        assert np.isclose(quantities["energy"], 7.5)  # sum of E


class TestEulerEquations2D:
    """Test suite for 2D Euler equations."""
    
    @pytest.fixture
    def equations(self):
        """Create 2D Euler equations instance."""
        return EulerEquations2D(gamma=1.4)
    
    def test_primitive_conversion(self, equations):
        """Test conversion from conservative to primitive variables."""
        U = np.array([[[1.0]], [[0.0]], [[0.0]], [[2.5]]])  # rho=1, ux=0, uy=0, E=2.5
        rho, ux, uy, p = equations.primitive(U)
        
        assert np.allclose(rho, 1.0)
        assert np.allclose(ux, 0.0)
        assert np.allclose(uy, 0.0)
        assert np.allclose(p, 1.0)
    
    def test_conservative_conversion(self, equations):
        """Test conversion from primitive to conservative variables."""
        rho = np.array([[1.0, 2.0]])
        ux = np.array([[0.5, -0.3]])
        uy = np.array([[0.2, 0.4]])
        p = np.array([[1.0, 2.0]])
        
        U = equations.conservative(rho, ux, uy, p)
        
        # Check shape
        assert U.shape == (4, 1, 2)
        
        # Check mass conservation
        assert np.allclose(U[0], rho)
        
        # Check momentum conservation
        assert np.allclose(U[1], rho * ux)
        assert np.allclose(U[2], rho * uy)
        
        # Check energy conservation
        expected_E = p / (equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)
        assert np.allclose(U[3], expected_E)


class TestEulerEquations3D:
    """Test suite for 3D Euler equations."""
    
    @pytest.fixture
    def equations(self):
        """Create 3D Euler equations instance."""
        return EulerEquations3D(gamma=1.4)
    
    def test_primitive_conversion(self, equations):
        """Test conversion from conservative to primitive variables."""
        U = np.array([[[[1.0]]], [[[0.0]]], [[[0.0]]], [[[0.0]]], [[[2.5]]]])  # rho=1, ux=0, uy=0, uz=0, E=2.5
        rho, ux, uy, uz, p = equations.primitive(U)
        
        assert np.allclose(rho, 1.0)
        assert np.allclose(ux, 0.0)
        assert np.allclose(uy, 0.0)
        assert np.allclose(uz, 0.0)
        assert np.allclose(p, 1.0)
    
    def test_conservative_conversion(self, equations):
        """Test conversion from primitive to conservative variables."""
        rho = np.array([[[1.0, 2.0]]])
        ux = np.array([[[0.5, -0.3]]])
        uy = np.array([[[0.2, 0.4]]])
        uz = np.array([[[0.1, -0.2]]])
        p = np.array([[[1.0, 2.0]]])
        
        U = equations.conservative(rho, ux, uy, uz, p)
        
        # Check shape
        assert U.shape == (5, 1, 1, 2)
        
        # Check mass conservation
        assert np.allclose(U[0], rho)
        
        # Check momentum conservation
        assert np.allclose(U[1], rho * ux)
        assert np.allclose(U[2], rho * uy)
        assert np.allclose(U[3], rho * uz)
        
        # Check energy conservation
        expected_E = p / (equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2 + uz**2)
        assert np.allclose(U[4], expected_E)


class TestEquationsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_density_behavior(self):
        """Test behavior with zero density (should not crash)."""
        equations = EulerEquations1D()
        U = np.array([[0.0], [0.0], [0.0]])  # Zero density
        
        # Should not crash, but may produce inf/nan values
        rho, u, p, a = equations.primitive(U)
        assert np.allclose(rho, 0.0)
        # u, p, a may be inf/nan, which is expected
    
    def test_negative_density_behavior(self):
        """Test behavior with negative density (should not crash)."""
        equations = EulerEquations1D()
        U = np.array([[-1.0], [0.0], [2.5]])  # Negative density
        
        # Should not crash, but may produce unexpected values
        rho, u, p, a = equations.primitive(U)
        assert np.allclose(rho, -1.0)
    
    def test_low_energy_behavior(self):
        """Test behavior with very low energy."""
        equations = EulerEquations1D()
        U = np.array([[1.0], [0.0], [0.1]])  # Very low energy
        
        # Should not crash
        rho, u, p, a = equations.primitive(U)
        assert np.allclose(rho, 1.0)
        assert np.allclose(u, 0.0)
    
    def test_different_gamma_values(self):
        """Test equations with different gamma values."""
        for gamma in [1.2, 1.4, 1.67, 2.0]:
            equations = EulerEquations1D(gamma=gamma)
            U = np.array([[1.0], [0.0], [2.5]])
            rho, u, p, a = equations.primitive(U)
            
            # Check that sound speed scales correctly
            expected_a = np.sqrt(gamma * p / rho)
            assert np.allclose(a, expected_a)
