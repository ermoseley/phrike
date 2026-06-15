Here’s a concrete stabilization plan tailored to phrike’s structure, without code yet.

Suspect: Implement tau terms: Use weak boundary conditions with auxiliary variables
Use more stable time stepping: Consider implementing SBDF2 or similar
Add proper dealiasing: Use dealias=3/2 consistently

Could also speed up with sparse matrix multiplication for differentiation matrices, as well as grid generation jacobi.quadrature (may not matter)

### 0) Config toggles
- Add to YAML:
  - integration.split_form: skew | conservative
  - integration.boundary_scheme: sat | strong
  - integration.sat_flux: rusanov | ec
  - integration.sat_tau: float (penalty scale)
  - integration.filter.interval: 1
  - artificial_viscosity.enabled: true (tiny ν) for bootstrapping

### 1) Grid/Basis data needed (already mostly present)
- From `Grid2D` when Legendre:
  - `Dx` via basis or `dx1`, `Dy` via `dy1`.
  - `wx`, `wy` 1D quadrature weights.
  - `e0_y, eN_y` boundary selectors along y (can be simple index access).
  - `e0_x, eN_x` along x for left/right.
  - `dx_min, dy_min` for CFL (done).

### 2) RHS architecture in `solver.py`
Add a selectable path in `_compute_rhs_2d`:

- Path A: Strong form (current)
  - Fluxes Fx,Fy = eqs.flux(U)
  - Divergences dFdx = dx1(Fx), dFdy = dy1(Fy)
  - rhs = -(dFdx + dFdy)
  - Only keep as baseline/fallback.

- Path B: Skew-symmetric split form (recommended interior)
  - Compute primitives (ρ, ux, uy, p).
  - Compute derivatives of primitives and conservative variables with `dx1`, `dy1`.
  - Build each divergence in skew form to reduce aliasing, e.g. for 1D analogy:
    - ∂x(ρu) ≈ 0.5[∂x(ρu) + u∂xρ + ρ∂x u]
  - Do this for x and y, all equations, assemble rhs_split = -(∂x F + ∂y F) in skew form.
  - Still O(N^2) free and matches current operators.

- Boundary terms via SAT (if boundary_scheme=sat)
  - Build numerical normal fluxes F* at boundaries using Rusanov:
    - Bottom (n = -ŷ): F*_y = 0.5(F_y(U_in) + F_y(U_bc)) - 0.5 α (U_in - U_bc)
    - Top (n = +ŷ), Left (n = -x̂), Right (n = +x̂) similarly.
    - α = max wave speed normal to the boundary (from `equations.max_wave_speed` on a boundary slice).
  - Convert numerical fluxes into SAT contributions using SBP/LGL weights:
    - For bottom row j=0: rhs[:,0,:] += (1/wy[0]) * (+F*_y) with correct sign/orientation.
    - For top row j=Ny-1: rhs[:,-1,:] -= (1/wy[-1]) * (+F*_y).
    - Analogous for x-sides using wx weights and columns.
  - Boundary states U_bc:
    - Bottom wall: normal velocity zero (v=0), tangential unchanged; recompute E from p and u_t.
    - Outflow (top/left/right): copy state from interior (zero-gradient) into U_bc for weak enforcement.

- If boundary_scheme=strong (interim fallback)
  - Enforce strong BCs each substage (as we did), but keep SAT path as the goal.

- Add gravity source after flux divergence and SAT:
  - Same as current.

- Positivity limiter (keep):
  - Clamp ρ, p and rebuild U at every RK substage.

- Filtering and viscosity:
  - Apply modal Legendre filter every step (interval=1).
  - Add tiny constant ν (artificial_viscosity.constant.nu_constant) to momentum and energy (scalar Laplacian via two `dx1/dy1` applications or via basis-provided ∇²) as a safety net.

### 3) Equations helpers in `equations.py`
- Add split-form builder routines for 2D:
  - Functions to compute skew-symmetric divergence pieces given (ρ, ux, uy, p) and their derivatives.
  - Small utilities to build wall/outflow U_bc and compute normal flux F_n(U) for given face normal.

### 4) Monitoring (done/extend)
- Keep weighted integrals using `wy, wx`.
- For diagnostics, also report:
  - min/max of ρ and p to detect onset of negativity.
  - boundary normal momentum norms to assess BC enforcement.

### 5) Integration and adaptive stepping
- Use min spacing-based CFL (done).
- Start with small cfl (0.01–0.05) until SAT+skew stabilize the scheme.
- Keep adaptive RK (rk78) with conservative safety_factor (0.5).

### 6) RTI-specific boundary states
- Bottom wall:
  - Enforce v=0, mirror or zero normal momentum; pressure from hydrostatic interior; recompute E.
- Top/left/right outflow:
  - U_bc = U_interior (copy from j=Ny-2, i=1 or i=Nx-2) giving zero-gradient; SAT flux handles the weak imposition.

### 7) Order of operations per RK stage
For each substage state U_s:
- If split_form: compute primitive variables and their derivatives.
- Compute interior rhs (skew form if enabled; else strong).
- Build SAT contributions at all four boundaries with Rusanov flux (using U_s and U_bc).
- Add gravity sources.
- Apply positivity limiter to U_s (or clamp rhs if desired).
- Optionally apply small ν and filter (or filter only after full stage, but per-stage helps).

### 8) Incremental rollout
- Phase 1: Strong form + SAT boundary terms + small ν + filtering + positivity limiter.
- Phase 2: Switch interior divergence to skew-symmetric split form.
- Phase 3: Replace Rusanov with entropy-conserving flux + dissipation (optional).

### 9) Validation steps
- Start with 1D vertical hydrostatic column (Legendre) with wall/outflow to validate SAT correctness and hydrostatic preservation.
- 2D acoustic reflection test to check wall energy behavior.
- Then RTI with tiny perturbation, cfl small, monitor p, ρ positivity and boundary fluxes.

If you’re good with this plan, I’ll implement Phase 1 (SAT at boundaries + per-stage enforcement, tiny ν, weighted monitoring already in) and wire the config flags, then iterate to split-form.