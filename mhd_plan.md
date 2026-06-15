# Implementing Ideal MHD in phrike with Helmholtz (Leray) Projection for ∇·B = 0

## 0. Goal and guiding principle

Add ideal compressible magnetohydrodynamics (MHD) to phrike's 3D pseudo-spectral
(Fourier) solver, evolving the magnetic field **B** alongside the existing
Euler state, and maintain the solenoidal constraint **∇·B = 0 to machine
precision** at all stored times.

The key enabler: phrike's `Grid3D` is a **periodic Fourier** solver, so the
Helmholtz/Leray projection onto the divergence-free subspace is *exact and
trivial* in spectral space. For each wavevector **k**,

```
B̂_sol(k) = B̂(k) − k (k·B̂(k)) / |k|²        (k ≠ 0)
```

removes the longitudinal (monopole) component so that `k·B̂_sol(k) = 0` for every
mode. The discrete spectral divergence `∇·B = ifftn(i k·B̂)` is then identically
zero up to floating-point round-off (~1e-14, double). This is strictly cleaner
than hyperbolic divergence cleaning (Dedner) or constrained transport, which
only push ∇·B down to truncation error.

**Scope of the projection:** this is *compressible* MHD — velocity **u** is NOT
divergence-free. The projection is applied to **B only**. Do not project momentum.

---

## 1. Mathematical formulation (ideal MHD, μ₀ = 1)

State vector (8 components), chosen so the first 5 match `EulerEquations3D` exactly:

```
U = [ ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z ]
```

Keeping the hydro block identical means the existing momentum/energy indexing,
tracers (read ρ at index 0, velocity at 1–3), and much of the I/O reuse cleanly.

Total energy and total pressure:

```
E   = p/(γ−1) + ½ρ|u|² + ½|B|²
p*  = p + ½|B|²                       (thermal + magnetic pressure)
```

**Pressure recovery (critical difference from Euler):** the magnetic energy must
be subtracted before recovering thermal pressure:

```
p = (γ−1) ( E − ½ρ|u|² − ½|B|² )
```

Conservative form, `∂U/∂t + ∂_x F^x + ∂_y F^y + ∂_z F^z = 0`. With
`u·B = u_xB_x + u_yB_y + u_zB_z`:

**F^x**
```
ρu_x
ρu_x² + p* − B_x²
ρu_x u_y − B_x B_y
ρu_x u_z − B_x B_z
(E + p*)u_x − B_x (u·B)
0
u_x B_y − u_y B_x
u_x B_z − u_z B_x
```

**F^y**
```
ρu_y
ρu_y u_x − B_y B_x
ρu_y² + p* − B_y²
ρu_y u_z − B_y B_z
(E + p*)u_y − B_y (u·B)
u_y B_x − u_x B_y
0
u_y B_z − u_z B_y
```

**F^z**
```
ρu_z
ρu_z u_x − B_z B_x
ρu_z u_y − B_z B_y
ρu_z² + p* − B_z²
(E + p*)u_z − B_z (u·B)
u_z B_x − u_x B_z
u_z B_y − u_y B_z
0
```

The magnetic-flux rows are the antisymmetric `uB − Bu` tensor, i.e. the induction
equation `∂B/∂t + ∇·(uB − Bu) = 0  ⇔  ∂B/∂t = ∇×(u×B)`.

**This requires no change to `_compute_rhs_3d`'s structure** — it is still
`rhs = −(dx1(Fx) + dy1(Fy) + dz1(Fz))`. We only need a new `flux()` that returns
8-component `(Fx, Fy, Fz)`.

### Why the projection is still needed

Analytically `∇·(∇×(u×B)) ≡ 0`, so the *continuous* induction term is
divergence-free. But the discrete RHS forms the nonlinear products `uB`, `BB`,
`u·B` in physical space and applies the 2/3 dealias mask + exponential filter
between FFT derivative operations. These operations do **not** commute exactly
with the products, so a small ∇·B is generated each step (truncation/aliasing
level), and the time integrator accumulates it. The Helmholtz projection wipes it
out, restoring machine-precision div-free at every stored state.

---

## 2. Wave speeds and CFL

Add the **fast magnetosonic speed** to the signal speed. Per cell:

```
a²    = γ p / ρ                          (sound speed²)
b²    = |B|² / ρ                         (total Alfvén speed²)
c_An² = B_n² / ρ                         (normal Alfvén speed², n = x,y,z)
c_f,n = sqrt( ½ [ (a²+b²) + sqrt( (a²+b²)² − 4 a² c_An² ) ] )
```

CFL signal speed in direction n: `|u_n| + c_f,n`. For a simple, safe, isotropic
bound (mirroring the existing 3D `max(|ux|,|uy|,|uz|) + a`), use the fact that
`c_f ≤ sqrt(a² + b²)`:

```
max_wave_speed = max over grid of [ max(|u_x|,|u_y|,|u_z|) + sqrt(a² + b²) ]
```

This is conservative (slightly over-estimates dt restriction) and cheap. Expose
the exact directional formula too for tighter stepping later.

---

## 3. File-by-file implementation

### 3.1 `phrike/equations.py` — add `MHDEquations3D`

A dataclass mirroring `EulerEquations3D` (so `BaseProblem.run` and the solver call
it identically), with `gamma`. Methods, all supporting both NumPy and torch paths
exactly like the existing classes:

- `primitive(U) -> (rho, ux, uy, uz, p, Bx, By, Bz)` — note the **8-tuple** and the
  magnetic-energy subtraction in `p`.
- `conservative(rho, ux, uy, uz, p, Bx, By, Bz) -> U` (8 stacked components).
- `flux(U) -> (Fx, Fy, Fz)` — the tensors from §1.
- `max_wave_speed(U) -> float` — §2.
- `conserved_quantities(U) -> dict` — add `magnetic_energy = ½∑|B|²`,
  `total_energy`, `cross_helicity = ∑ u·B`, and (diagnostic) `div_b_max` if a grid
  is passed, else compute it in the solver. Keep `mass`, `momentum_*`, `energy`.

Sketch (NumPy path; add the `torch.stack` branch as in the Euler classes):

```python
@dataclass
class MHDEquations3D:
    gamma: float = 5.0 / 3.0   # MHD convention; configurable

    def primitive(self, U):
        rho = U[0]; ux = U[1]/rho; uy = U[2]/rho; uz = U[3]/rho
        E = U[4]; Bx = U[5]; By = U[6]; Bz = U[7]
        kinetic = 0.5 * rho * (ux*ux + uy*uy + uz*uz)
        magnetic = 0.5 * (Bx*Bx + By*By + Bz*Bz)
        p = (self.gamma - 1.0) * (E - kinetic - magnetic)
        return rho, ux, uy, uz, p, Bx, By, Bz

    def flux(self, U):
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        E = U[4]
        uB = ux*Bx + uy*By + uz*Bz
        pstar = p + 0.5 * (Bx*Bx + By*By + Bz*Bz)
        mx, my, mz = rho*ux, rho*uy, rho*uz
        Fx = np.stack([
            mx,
            mx*ux + pstar - Bx*Bx,
            mx*uy - Bx*By,
            mx*uz - Bx*Bz,
            (E + pstar)*ux - Bx*uB,
            np.zeros_like(rho),
            ux*By - uy*Bx,
            ux*Bz - uz*Bx,
        ], axis=0)
        # Fy, Fz analogously (see §1)
        return Fx, Fy, Fz
```

Numba kernels (`_flux_mhd3d_kernel`, etc.) are an optional later optimization; start
with vectorized NumPy/torch to get correctness first. Keep `_TORCH_AVAILABLE`
guards identical to the existing code.

### 3.2 `phrike/grid.py` — add projection / divergence / curl to `Grid3D`

Reuse the existing `fftn`/`ifftn`, `kx/ky/kz` (already 2π/L-scaled, present for both
NumPy and torch backends). Precompute `K2` once in `__post_init__`:

```python
# in Grid3D.__post_init__ (both numpy and torch)
self.k2 = (self.kx[None, None, :]**2
         + self.ky[None, :, None]**2
         + self.kz[:, None, None]**2)
self.k2_safe = where(self.k2 == 0, 1, self.k2)   # avoid /0 at k=0
```

Methods (NumPy + torch dual path, same pattern as `dx1`):

```python
def divergence(self, Vx, Vy, Vz):
    """Spectral ∇·V using the SAME masked operator as dx1/dy1/dz1."""
    Fx = self._apply_masks(self.fftn(Vx))
    Fy = self._apply_masks(self.fftn(Vy))
    Fz = self._apply_masks(self.fftn(Vz))
    div_hat = self.ikx*Fx + self.iky*Fy + self.ikz*Fz
    return self.ifftn(div_hat)

def project_solenoidal(self, Vx, Vy, Vz):
    """Helmholtz projection: remove the longitudinal part of (Vx,Vy,Vz).
    After this, k·V̂ = 0 for all k  ⇒  ∇·V = 0 to machine precision."""
    Fx = self.fftn(Vx); Fy = self.fftn(Vy); Fz = self.fftn(Vz)
    kdotV = self.kx[None,None,:]*Fx + self.ky[None,:,None]*Fy + self.kz[:,None,None]*Fz
    coef = kdotV / self.k2_safe          # k=0 mode: kdotV=0 there, stays 0
    Fx = Fx - self.kx[None,None,:]*coef
    Fy = Fy - self.ky[None,:,None]*coef
    Fz = Fz - self.kz[:,None,None]*coef
    return self.ifftn(Fx), self.ifftn(Fy), self.ifftn(Fz)
```

Notes:
- The projection uses *plain* `k` (not the masked `ik`); the masked divergence in
  `divergence()` (used for diagnostics) is still machine-zero afterwards because
  the mask only zeroes high modes and `k·V̂ = 0` holds on every retained mode.
- `curl(...)` (optional, §6) for the curl-form induction term.

### 3.3 `phrike/solver.py` — add `SpectralSolverMHD3D`

A dedicated solver (cleanest; CFL, monitoring, and the projection hook all differ).
It reuses the `_compute_rhs_3d` structure but:

1. **RHS:** `Fx,Fy,Fz = eqs.flux(U); rhs = −(dx1(Fx)+dy1(Fy)+dz1(Fz))` — identical
   shape, 8 components. Optional resistive/viscous terms (§6) added here.
2. **Project B after every RK substage and after the full step.** Wrap each stage:

```python
def _project_B(self, U):
    Bx, By, Bz = self.grid.project_solenoidal(U[5], U[6], U[7])
    # Energy bookkeeping: hold THERMAL pressure fixed, recompute E with B_sol
    # so the removed (unphysical monopole) magnetic energy does not leak into p.
    rho, ux, uy, uz, p, *_ = self.equations.primitive(U)
    U = U.clone() if is_torch(U) else U.copy()
    U[5], U[6], U[7] = Bx, By, Bz
    U[4] = p/(self.equations.gamma-1.0) + 0.5*rho*(ux*ux+uy*uy+uz*uz) \
           + 0.5*(Bx*Bx+By*By+Bz*Bz)
    return U
```

   Apply `_project_B` to each intermediate `U2,U3,U4` and to the final update in the
   RK2/RK4 steppers (write MHD variants `_rk{2,4}_step_mhd3d`). Projecting per
   substage keeps div(B) at machine precision *throughout* the step, not just at
   stored outputs. (Per-step-only is a cheaper fallback; within-substage div(B) is
   then truncation-level, which is usually acceptable.)
3. **Order of operations per step:** RK stages (each: flux → rhs → update →
   `_project_B`) → `_apply_physical_filters_3d` (spectral filter) → `_project_B`
   **last**, because filtering B re-introduces a longitudinal component. The stored
   state is therefore exactly solenoidal.
4. **`compute_dt`:** use the MHD `max_wave_speed` (§2) over
   `min(dx,dy,dz)`.
5. **Positivity:** clamp ρ and **thermal** p (recompute E with the *current* B), since
   strong fields can drive the thermal pressure negative. Reuse the
   `_positivity_clamp` idea but with MHD primitive/conservative.

Adaptive RK (rk23/45/78) works unchanged through `_compute_rhs_mhd_3d`; insert a
`_project_B` on the accepted state inside `_adaptive_step` / `_adaptive_run_step`.

### 3.4 `phrike/problems/problem_list/` — MHD problems

Add problem classes subclassing `BaseProblem`, each returning `Grid3D`,
`MHDEquations3D`, and `SpectralSolverMHD3D` from `get_solver_class()`. Initial B
**must be projected** before first use (`grid.project_solenoidal`). Suggested set:

- **`alfven3d.py`** — circularly polarized Alfvén wave. Exact nonlinear solution;
  the primary *convergence* and *div(B)* validation. Spatial: spectral
  (exponential) convergence; temporal: RK order.
- **`orszag_tang3d.py`** — Orszag–Tang vortex (2D fields in a 3D box). Classic
  qualitative MHD benchmark.
- **`mhd_turb3d.py`** — extend the existing `turb3d` machinery: reuse
  `turbulent_velocity_3d` for **u**, generate a banded random **B** with target
  spectrum and `b_rms`, then `project_solenoidal`. Natural fit for the active
  turbulence work and tracers.
- **`field_loop3d.py`** (optional) — magnetic field-loop advection; standard ∇·B
  stress test.

Register all in `phrike/problems/register.py` (e.g. `mhd_turb3d`, `alfven3d`,
`orszag_tang3d`).

### 3.5 `phrike/io.py` — snapshots for MHD

`save_solution_snapshot` currently branches 1D/2D/3D on grid attributes. Add an MHD
branch keyed on `U.shape[0] == 8` (or `hasattr(equations, "...")`/`len(primitive)`):
save `Bx, By, Bz` and a `div_b` diagnostic field (`grid.divergence(Bx,By,Bz)`) plus
`meta["mhd"] = True`. Keep `rho, ux, uy, uz, p` so existing viewers still work.

### 3.6 Monitoring — `phrike/problems/base.py`

`compute_velocity_stats` raises for `len(primitive_vars) ∉ {4,5}`. Add an
**8-tuple branch** (`rho, ux, uy, uz, p, Bx, By, Bz`) computing `v_mag` from the
velocity components and ignoring B. Add MHD diagnostics to the monitoring output:
`magnetic_energy`, `b_rms`, `max|∇·B|` (the headline number — should sit at ~1e-13),
`plasma_β = p/(½|B|²)`, `cross_helicity`. These come from
`equations.conserved_quantities` + `grid.divergence`.

### 3.7 Config — `configs/`

Add e.g. `configs/mhd_turb3d.yaml`, `configs/alfven3d.yaml`. Extend the existing
schema with:

```yaml
physics:
  gamma: 1.6666666666666667     # 5/3 for MHD
  mhd:
    enabled: true
    resistivity: 0.0            # η, explicit (0 = ideal)
    viscosity: 0.0              # ν, explicit (0 = ideal)
    divergence_clean: projection # projection | none
    project_each_substage: true

initial_conditions:
  # hydro keys as in turb3d, plus:
  b0: [0.0, 0.0, 1.0]           # mean field
  b_rms: 0.1                    # fluctuation amplitude
  b_kmin: 1.0
  b_kmax: 5.0
```

`gamma` is already read from `physics.gamma`; the `mhd` block is new and parsed in
the problem's `setup_common_parameters`/`create_equations`.

---

## 4. Initial conditions: generating a divergence-free B

1. Build a banded random vector field B (reuse the spectral-envelope code in
   `turbulent_velocity_3d`).
2. `Bx,By,Bz = grid.project_solenoidal(Bx,By,Bz)` → exactly solenoidal.
3. Add a uniform mean field `b0` (constant ⇒ k=0 only ⇒ already div-free).
4. Normalize to target `b_rms`.
5. Assemble `U = eqs.conservative(rho, ux, uy, uz, p, Bx, By, Bz)`.

For the Alfvén wave, set the analytic B and verify `max|∇·B| ~ 1e-15` at t=0 as a
unit test of the projection plumbing.

---

## 5. Dealiasing, filtering, torch/MPS

- Nonlinearities (`uB`, `BB`, `u·B`) are quadratic — the existing **2/3-rule**
  dealias mask is sufficient. The exponential filter applies to B components too.
- **MPS/torch:** projection uses `kx,ky,kz` and `fftn/ifftn`, all already torch-aware
  on the grid. Precompute `k2_safe` as a tensor on the correct device/dtype. MPS
  forces float32 (per existing grid logic) — div(B) floor will be ~1e-6 in single
  precision; for true "machine precision" run the **double / CPU or CUDA** path. Make
  the validation suite default to double precision.

---

## 6. Optional refinements (after the core works)

- **Curl-form induction:** compute the induction term as `∂B/∂t = curl(u×B)` via a
  spectral curl instead of the conservative flux rows. Then the *time derivative* of
  B is divergence-free by construction (`k·(k×X̂) ≡ 0`), so pre-projection div(B) is
  already near round-off and the projection becomes a tiny correction. Keep the
  conservative flux form as the default (matches phrike's `flux()` API); offer curl
  form behind a config flag.
- **Explicit resistivity / viscosity:** add `η∇²B` and `ν∇²u` (`−ηk²B̂`, `−νk²(ρu)̂`
  in spectral space). Trivial, and gives a physical dissipation scale for turbulence.
  `∇²` preserves the solenoidal property (`∇·∇²B = ∇²(∇·B) = 0`).
- **Vector-potential alternative (cross-check):** evolve **A** with **B = ∇×A**, which
  guarantees div(B)=0 by construction. More invasive (gauge handling, energy
  coupling); useful only as an independent verification of the projection approach.
- **Numba kernels** for the MHD flux once correctness is established.

---

## 7. Validation and acceptance criteria

| Test | What it checks | Pass criterion |
|------|----------------|----------------|
| Projection unit test | spectral plumbing | `max|∇·B|` ≤ 1e-13 (double) on random + analytic B |
| ∇·B over time (turb/OT) | constraint preservation | `max|∇·B|` stays ~1e-13, **resolution- and time-independent** |
| Circularly polarized Alfvén wave | accuracy | spectral spatial convergence; RK temporal order; phase speed = `B_0/√ρ` |
| Orszag–Tang vortex | nonlinear correctness | qualitative match to published density/pressure structure |
| Energy conservation (ideal, η=ν=0) | flux consistency | total energy conserved to filter/time tolerance |
| Field-loop advection | div control under advection | loop advects without ∇·B growth or shape distortion |

Headline metric: `max|∇·B|` printed every monitoring interval, sitting at
double-precision round-off independent of resolution and elapsed time.

---

## 8. Incremental rollout

1. **M1 — Equations:** `MHDEquations3D` (primitive/conservative/flux/wave speed/
   conserved quantities) + unit tests (flux of a known state, p recovery).
2. **M2 — Projection:** `Grid3D.project_solenoidal` + `divergence` + projection unit
   test (random and analytic B → `max|∇·B| ~ 1e-14`).
3. **M3 — Solver:** `SpectralSolverMHD3D` with per-substage projection, MHD CFL,
   positivity; reuse RK structure.
4. **M4 — Problem + config:** `alfven3d` first (validation), then `mhd_turb3d`.
   Wire I/O (Bx,By,Bz,div_b) and the monitoring 8-tuple branch.
5. **M5 — Validate:** run the §7 suite; confirm machine-precision ∇·B and Alfvén
   convergence.
6. **M6 — Optional:** resistivity/viscosity, curl-form induction, Numba kernels,
   tracers in MHD turbulence.

---

## 9. Risks and gotchas

- **Pressure recovery** must subtract ½|B|²; forgetting it corrupts `p`, sound speed,
  and CFL. Most common MHD bug.
- **Energy bookkeeping in projection:** recompute E from projected B holding *thermal*
  p fixed (don't let removed monopole energy leak into pressure). The change is tiny
  but doing it correctly keeps p consistent and positive.
- **Filter after projection re-introduces div(B):** project **last** in the step.
- **k=0 mode:** guard `|k|²` (`k2_safe`); the mean field is untouched (correctly).
- **MPS single precision** floors div(B) at ~1e-6 — validate in double on CPU/CUDA.
- **Monitoring arity:** `compute_velocity_stats` must learn the 8-tuple, or it raises.
- **CFL:** fast speed `sqrt(a²+b²)` can be ≫ sound speed in low-β regions → smaller dt;
  expect this, don't fight it.

---

## 10. References

- Brackbill & Barnes (1980), *The effect of nonzero ∇·B* — the projection/cleaning idea.
- Tóth (2000), *The ∇·B=0 constraint in shock-capturing MHD codes* — comparison of methods.
- Spectral-method MHD: divergence cleaning via Leray projection is exact in Fourier space.
- Orszag & Tang (1979); circularly polarized Alfvén wave as a nonlinear convergence test.
