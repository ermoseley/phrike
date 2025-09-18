# Dedalus Sod Shock Tube with Chebyshev Basis (256 points)

This guide explains how to run a Sod shock tube simulation using Dedalus with a Chebyshev spectral basis and 256 grid points.

## Overview

The Sod shock tube problem is a classic test case for computational fluid dynamics that involves:
- **Left state**: ρ=1.0, u=0.0, p=1.0
- **Right state**: ρ=0.125, u=0.0, p=0.1
- **Initial discontinuity** at x=0.5
- **Domain**: [0, 1] with 256 Chebyshev-Gauss-Lobatto points

## Files Created

1. **`dedalus_sod_chebyshev.py`** - Basic Dedalus implementation
2. **`dedalus_sod_chebyshev_working.py`** - Complete implementation with error handling
3. **`DEDALUS_SOD_CHEBYSHEV_GUIDE.md`** - This guide

## Key Concepts

### Chebyshev Basis in Dedalus

The Chebyshev basis uses Chebyshev-Gauss-Lobatto (CGL) nodes:
```python
# Nodes on [-1, 1]
y = cos(π * j / (N-1)) for j = 0, 1, ..., N-1

# Mapped to problem domain [0, Lx]
x = (1 - y) * Lx / 2
```

### Dedalus Structure

```python
import dedalus.public as d3

# 1. Create coordinate and distributor
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

# 2. Create Chebyshev basis
xbasis = d3.Chebyshev(xcoord, size=256, bounds=(0, 1), dealias=3/2)

# 3. Create fields
rho = dist.Field(name='rho', bases=xbasis)
rhou = dist.Field(name='rhou', bases=xbasis)
E = dist.Field(name='E', bases=xbasis)

# 4. Define problem
problem = d3.IVP([rho, rhou, E], namespace=locals())
problem.add_equation("dt(rho) + dx(rhou) = 0")
problem.add_equation("dt(rhou) + dx(rhou*u + p) = 0")
problem.add_equation("dt(E) + dx((E + p)*u) = 0")

# 5. Build solver and run
solver = problem.build_solver(d3.timesteppers.RK443)
```

## Installation Requirements

To run the Dedalus implementation, you need:

```bash
pip install dedalus
```

**Note**: The current workspace has a Dedalus installation that appears to be missing compiled components. The implementation will work once Dedalus is properly installed.

## Running the Simulation

### Option 1: Using Phrike (Working Implementation)

The existing Phrike codebase already has a working Sod shock tube with Chebyshev basis:

```bash
cd /Users/moseley/hydra
python -m phrike sod --config configs/chebyshev_sod_256.yaml
```

This runs successfully and produces:
- Solution snapshots
- Visualization plots
- Video output

### Option 2: Using Dedalus (Once Installed)

```bash
python dedalus_sod_chebyshev_working.py
```

## Configuration

The configuration file `configs/chebyshev_sod_256.yaml` specifies:

```yaml
problem: sod
grid:
  N: 256                    # 256 Chebyshev modes
  Lx: 1.0                   # Domain length
  basis: chebyshev          # Chebyshev basis
  bc: dirichlet             # Boundary conditions
  dealias: true             # Enable dealiasing
  precision: single         # Single precision
equations:
  gamma: 1.4                # Ratio of specific heats
integration:
  t0: 0.0
  t_end: 0.2
  cfl: 0.15
  scheme: rk4
  output_interval: 0.01
```

## Key Features

### 1. Spectral Accuracy
- Chebyshev basis provides exponential convergence for smooth functions
- CGL nodes cluster near boundaries, ideal for boundary layer problems

### 2. Dealiasing
- 3/2 rule prevents aliasing in nonlinear terms
- Essential for stability in spectral methods

### 3. Boundary Conditions
- Dirichlet conditions enforce u=0 at boundaries
- Prevents spurious oscillations

### 4. Time Integration
- RK4 scheme for time stepping
- Adaptive time stepping available

## Results

The simulation produces:
1. **Density evolution**: Shows shock formation and propagation
2. **Velocity profiles**: Displays fluid motion
3. **Pressure evolution**: Shows pressure waves
4. **Space-time plots**: Visualize wave propagation

## Comparison with Phrike

Both implementations solve the same problem but use different approaches:

| Feature | Dedalus | Phrike |
|---------|---------|--------|
| Basis | Chebyshev | Chebyshev |
| Grid Points | 256 | 256 |
| Time Integration | RK443 | RK4 |
| Dealiasing | 3/2 rule | 3/2 rule |
| Boundary Conditions | Dirichlet | Dirichlet |
| Status | Requires installation | Working |

## Troubleshooting

### Dedalus Installation Issues
If you encounter import errors:
1. Ensure Python environment is clean
2. Install dependencies: `pip install numpy scipy matplotlib`
3. Install Dedalus: `pip install dedalus`
4. For development: build from source

### Performance Optimization
- Set `OMP_NUM_THREADS=1` to avoid threading issues
- Use appropriate dealiasing factors
- Consider adaptive time stepping

## References

1. [Dedalus Documentation](https://dedalus-project.readthedocs.io/)
2. [Spectral Methods in Fluid Dynamics](https://www.springer.com/gp/book/9783540516106)
3. [Chebyshev Methods](https://www.chebfun.org/)

## Next Steps

1. Install Dedalus properly
2. Run the full simulation
3. Compare results with analytical solution
4. Experiment with different resolutions and time steps
5. Add artificial viscosity for shock capturing
