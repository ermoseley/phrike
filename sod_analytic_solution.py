#!/usr/bin/env python3
"""
Analytic solution for the Sod shock tube problem.
Based on the exact solution derived by Sod (1978).
"""

import numpy as np
import matplotlib.pyplot as plt

def sod_analytic_solution(x, t, gamma=1.4):
    """
    Compute the analytic solution for the Sod shock tube problem.
    
    Parameters:
    -----------
    x : array_like
        Spatial coordinates
    t : float
        Time
    gamma : float
        Ratio of specific heats (default: 1.4)
    
    Returns:
    --------
    rho, u, p : tuple of arrays
        Density, velocity, and pressure profiles
    """
    x = np.asarray(x)
    
    # Initial conditions
    rho_L = 1.0    # Left density
    u_L = 0.0      # Left velocity
    p_L = 1.0      # Left pressure
    rho_R = 0.125  # Right density
    u_R = 0.0      # Right velocity
    p_R = 0.1      # Right pressure
    
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Initialize arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    
    # For t = 0, return initial conditions
    if t <= 0:
        rho = np.where(x <= 0.5, rho_L, rho_R)
        u = np.where(x <= 0.5, u_L, u_R)
        p = np.where(x <= 0.5, p_L, p_R)
        return rho, u, p
    
    # Find the pressure in the intermediate region
    # This requires solving the Riemann problem
    p_star = solve_riemann_pressure(p_L, p_R, rho_L, rho_R, gamma, u_L, u_R)
    
    # Find the velocity in the intermediate region
    u_star = solve_riemann_velocity(p_L, p_R, rho_L, rho_R, gamma, p_star, u_L, u_R)
    
    # Find the density in the intermediate region
    rho_star_L, rho_star_R = solve_riemann_density(p_L, p_R, rho_L, rho_R, gamma, p_star)
    
    # Find the wave speeds
    c_star_L = c_L * (p_star / p_L)**((gamma - 1) / (2 * gamma))
    c_star_R = c_R * (p_star / p_R)**((gamma - 1) / (2 * gamma))
    
    # Wave positions
    x_contact = 0.5 + u_star * t
    x_shock = 0.5 + (u_star + c_star_R) * t
    x_rarefaction_left = 0.5 + (u_L - c_L) * t
    x_rarefaction_right = 0.5 + (u_star - c_star_L) * t
    
    # Assign values based on region
    for i, xi in enumerate(x):
        if xi <= x_rarefaction_left:
            # Left state (unperturbed)
            rho[i] = rho_L
            u[i] = u_L
            p[i] = p_L
        elif xi <= x_rarefaction_right:
            # Left rarefaction fan
            c = (xi - 0.5) / t + c_L
            rho[i] = rho_L * (c / c_L)**(2 / (gamma - 1))
            u[i] = 2 * (c - c_L) / (gamma - 1)
            p[i] = p_L * (c / c_L)**(2 * gamma / (gamma - 1))
        elif xi <= x_contact:
            # Left intermediate state
            rho[i] = rho_star_L
            u[i] = u_star
            p[i] = p_star
        elif xi <= x_shock:
            # Right intermediate state
            rho[i] = rho_star_R
            u[i] = u_star
            p[i] = p_star
        else:
            # Right state (unperturbed)
            rho[i] = rho_R
            u[i] = u_R
            p[i] = p_R
    
    return rho, u, p

def solve_riemann_pressure(p_L, p_R, rho_L, rho_R, gamma, u_L=0.0, u_R=0.0):
    """Solve for the pressure in the intermediate region."""
    # Initial guess
    p_star = 0.5 * (p_L + p_R)
    
    # Newton-Raphson iteration
    for _ in range(20):
        f, df = riemann_functions(p_L, p_R, rho_L, rho_R, gamma, p_star, u_L, u_R)
        if abs(f) < 1e-10:
            break
        p_star = p_star - f / df
    
    return p_star

def riemann_functions(p_L, p_R, rho_L, rho_R, gamma, p_star, u_L=0.0, u_R=0.0):
    """Compute the Riemann problem functions and their derivatives."""
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    if p_star > p_L:
        # Left shock
        A_L = 2 / ((gamma + 1) * rho_L)
        B_L = (gamma - 1) / (gamma + 1) * p_L
        f_L = (p_star - p_L) * np.sqrt(A_L / (p_star + B_L))
        df_L = np.sqrt(A_L / (p_star + B_L)) * (1 - (p_star - p_L) / (2 * (p_star + B_L)))
    else:
        # Left rarefaction
        f_L = 2 * c_L / (gamma - 1) * ((p_star / p_L)**((gamma - 1) / (2 * gamma)) - 1)
        df_L = c_L / (gamma * p_L) * (p_star / p_L)**(-(gamma + 1) / (2 * gamma))
    
    if p_star > p_R:
        # Right shock
        A_R = 2 / ((gamma + 1) * rho_R)
        B_R = (gamma - 1) / (gamma + 1) * p_R
        f_R = (p_star - p_R) * np.sqrt(A_R / (p_star + B_R))
        df_R = np.sqrt(A_R / (p_star + B_R)) * (1 - (p_star - p_R) / (2 * (p_star + B_R)))
    else:
        # Right rarefaction
        f_R = 2 * c_R / (gamma - 1) * ((p_star / p_R)**((gamma - 1) / (2 * gamma)) - 1)
        df_R = c_R / (gamma * p_R) * (p_star / p_R)**(-(gamma + 1) / (2 * gamma))
    
    f = f_L + f_R + u_R - u_L
    df = df_L + df_R
    
    return f, df

def solve_riemann_velocity(p_L, p_R, rho_L, rho_R, gamma, p_star, u_L=0.0, u_R=0.0):
    """Solve for the velocity in the intermediate region."""
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    if p_star > p_L:
        # Left shock
        A_L = 2 / ((gamma + 1) * rho_L)
        B_L = (gamma - 1) / (gamma + 1) * p_L
        u_star = u_L + (p_star - p_L) * np.sqrt(A_L / (p_star + B_L))
    else:
        # Left rarefaction
        u_star = u_L + 2 * c_L / (gamma - 1) * ((p_star / p_L)**((gamma - 1) / (2 * gamma)) - 1)
    
    return u_star

def solve_riemann_density(p_L, p_R, rho_L, rho_R, gamma, p_star):
    """Solve for the density in the intermediate region."""
    if p_star > p_L:
        # Left shock
        A_L = 2 / ((gamma + 1) * rho_L)
        B_L = (gamma - 1) / (gamma + 1) * p_L
        rho_star_L = rho_L * (p_star / p_L + B_L) / (1 + B_L * p_star / p_L)
    else:
        # Left rarefaction
        rho_star_L = rho_L * (p_star / p_L)**(1 / gamma)
    
    if p_star > p_R:
        # Right shock
        A_R = 2 / ((gamma + 1) * rho_R)
        B_R = (gamma - 1) / (gamma + 1) * p_R
        rho_star_R = rho_R * (p_star / p_R + B_R) / (1 + B_R * p_star / p_R)
    else:
        # Right rarefaction
        rho_star_R = rho_R * (p_star / p_R)**(1 / gamma)
    
    return rho_star_L, rho_star_R

def test_analytic_solution():
    """Test the analytic solution."""
    x = np.linspace(0, 1, 1000)
    t = 0.2
    
    rho, u, p = sod_analytic_solution(x, t)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(x, rho, 'b-', linewidth=2, label='Density')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Analytic Density Profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(x, u, 'g-', linewidth=2, label='Velocity')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Analytic Velocity Profile')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(x, p, 'r-', linewidth=2, label='Pressure')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Analytic Pressure Profile')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('sod_analytic_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Analytic solution at t={t}:")
    print(f"  Density range: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"  Velocity range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  Pressure range: [{p.min():.3f}, {p.max():.3f}]")

if __name__ == "__main__":
    test_analytic_solution()
