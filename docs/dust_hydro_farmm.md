# Hydro FARMM: realizable moment dust with an active sub-grid reservoir

Design note for the hydro-only (unmagnetised) precursor of FARMM-Dust, the
field-aligned realizable moment method of
`~/PDY2/guiding_centre_multimoment_dust/multimoment_dust.tex` (with the corrections
of `multimoment_dust_audit.tex`). The magnetised version carries a parallel HyQMOM
hierarchy along **b̂**; here there is no field, so the "parallel" axis is the grid
axis in 1D and the multi-dimensional closure is the Gaussian ten-moment closure.
The SGS coupling coefficients follow the canonical MT25 contract
(`Obsidian/20-projects/mini-ramses/notes/MT25 stochastic GC dust — production theory contract.md`),
evaluated at zero magnetisation.

## 1. What is coupled to what

Three energy reservoirs exchange energy, and the scheme conserves their sum to
round-off apart from the gas solver's positivity floors:

| reservoir | field | gains from | loses to |
|---|---|---|---|
| resolved gas | `(ρ, ρu, E)` (pseudo-spectral) | SGS compression work (expansion), drag heating, SGS decay heating | spectral dissipation (to SGS), drag on dust mean motion |
| SGS turbulence | `e_s` (energy density, finite volume) | kinetic energy removed by the spectral dissipation; compression | decay to gas heat; work on dust random motion |
| dust | raw velocity moments (finite volume) | gas drag (mean); SGS forcing (random) | drag heating to gas |

Momentum is exchanged between the gas and dust mean motion by drag with
back-reaction. The SGS reservoir also pushes back on the resolved flow through the
turbulent pressure `p_s = (2/3) e_s`, so it is a mechanical reservoir and not only
a bookkeeping tally.

## 2. Gas

Unchanged PHRIKE Euler solver, except that the momentum flux carries `p + p_s`
and the energy flux carries `(E + p + p_s) u` when `sgs.pressure` is on. The
signal speed adds the SGS sound speed, `c² → γp/ρ + (10/9) e_s/ρ` (adiabatic SGS
energy scales as `ρ^{5/3}`). `p_s` is frozen over a step.

## 3. Dust kinetic model

Single grain species, distribution `F(x, v, t)`, lab frame:

    ∂F/∂t + v·∇F − ν ∂_v·[(v − u_g) F] = D ∇_v² F + ∇·(κ_d ∇F)

`ν = 1/t_s` is a constant or Epstein, `ν = ρ_g v_th / (ρ• a)` with
`v_th = (8 p / π ρ_g)^{1/2}`. Back-reaction is applied to the gas through the
momentum exchange.

### 3.1 Transported fields (raw lab-frame moments)

We transport **raw** moments, `ρ_d ⟨v^α⟩`, not central moments. The draft paper
transports central moments and documents the consequence (its Sec. "Central versus
raw moment transport"): the central scheme does not conserve the second moment, and
it destroys the variance created when numerical mixing combines streams. The
raw scheme conserves every transported moment to round-off, and it is realizable
by convexity, because the moment cone is convex in raw coordinates and a Rusanov
update is a convex combination. The cost is cancellation in `v = M2/M0 − m²` for
cold, fast dust, so the dust runs in float64 with explicit round-off tolerances (§6).

* **`gaussian`** closure (1D, 2D, 3D), 10 fields:
  `ρ, ρu_x, ρu_y, ρu_z, M_xx, M_yy, M_zz, M_xy, M_xz, M_yz` with `M_ij = ρ⟨v_i v_j⟩`.
  Third moments come from the Gaussian closure,
  `M_dij = ρ(u_d u_i u_j + u_d C_ij + u_i C_dj + u_j C_di)`, `C = M/ρ − u uᵀ`.
  All three velocity components are carried in every dimension. Components along
  unresolved directions are transported only along resolved ones.
* **`hyqmom`** closure (1D only), 8 fields:
  `M0..M4 = ρ⟨v_x^k⟩`, `ρu_y, ρu_z`, `P_⊥ = ρ⟨v_y² + v_z²⟩`. The `x` sector is the
  FARMM parallel hierarchy. `M5` comes from the n = 2 HyQMOM closure
  `C5 = q (5 r/v − 3 q²/v² − v)/2` (that is, `S5 = S3(5S4 − 3S3² − 1)/2`),
  `v, q, r` = central moments 2, 3, 4. The `(v_y, v_z)` distribution is an
  isotropic Gaussian independent of `v_x`, which is the exponential-in-μ closure
  of FARMM written for a hydro grain, so the mixed moments vanish.

### 3.2 Realizability

* `gaussian`: `ρ ≥ 0` and `C ⪰ 0`. All seven principal minors are non-negative,
  within tolerance.
* `hyqmom`: `ρ ≥ 0`, `v ≥ 0`, `K = r − v² − q²/v ≥ 0` (the fourth-order Hankel
  cone), and `σ_⊥² = (P_⊥/ρ − u_y² − u_z²)/2 ≥ 0`. A cell with `v ≤ tol·m²` is
  treated as cold (`C5 = 0`, `q`, `r` read as zero).
* `ρ = 0` requires every field to be zero.

### 3.3 Transport (finite volume on the collocation grid)

The dust points are PHRIKE's collocation points, treated as cell centres.

* **WENO5-Z faces** (Borges et al. 2008) of the reconstructed variables, with a
  fall-back to MC-limited linear faces wherever a WENO face leaves a positivity
  box (`ρ`, `v`, `H4`, `σ_⊥² ≥ 0`). Against the kinetic free-streaming
  references WENO5-Z is 6× (smooth packet) to 38× (crossing streams) more
  accurate than MC at 64 cells; from about 128 cells the remaining error is the
  closure's own, not the discretisation's. A kinetic flux with HLL in place of
  the node-wise upwinding changed nothing measurable on these tests, and for a
  stationary contact HLL is three times more dissipative: it upwinds every node
  with the outermost wave speeds. Each cell picks one of two reconstructions:
  * **Conserved (Zhang–Shu).** Face increments of the raw moments, scaled toward
    the cell average by powers of ½ until both faces are strictly realizable and
    their quadrature-node velocities lie within the neighbours' node range plus
    half its span. That is a relaxed kinetic maximum principle: it keeps second
    order at smooth extrema but stops light extrapolated faces from carrying
    spuriously fast nodes. Used for warm cells near the cone boundary
    (two-point mixtures, crossing streams) and for rough data.
  * **Primitive.** Gaussian: `(ρ, u, C)`, with the `C` increment halved until
    both faces are positive semi-definite. HyQMOM: `(ρ, u_x, v, S3, H4, u_y, u_z,
    σ_⊥²)`, where realizability is the box `v, H4, σ_⊥² ≥ 0`. HyQMOM uses this in the cone interior
    (`H4 ≥ ½` in the cell and both neighbours; a Gaussian has `H4 = 2`): `q, r`
    are small differences of `O(m³)` and `O(m⁴)` raw terms, and limiting raw
    components separately makes the face skewness and kurtosis noisy. Both
    closures also use it in **cold-like cells** (below), with the variance
    slopes set to zero so a cold cell's faces are Diracs at the face velocities.

  Why cold-like cells need it: a cold cell's raw `M2 = ρuu` has no room for a
  sub-cell velocity spread, so any realizable conserved reconstruction flattens
  its velocity, and cold dust drops to first order with an `O(1)` error at
  velocity stagnation points. Primitive faces are realizable by construction,
  but the cell average is no longer the mean of its faces, so the update can
  leave the cone by about the variance a linear velocity profile carries within
  a cell. That is resolved a posteriori (next bullets).

  Cells lighter than 10⁻⁶ of their neighbourhood maximum are first order in
  both reconstructions.
* **Sub-cell budget and cold-like cells.** The exact cell average of a smooth
  cold flow is not cold: a linear velocity profile carries variance `|Δu|²/12`.
  The unresolved budget is `b = (1/12) Σ_d min(|Δ₋u|², |Δ₊u|²)` over resolved
  axes; the minimum makes it vanish at discontinuities and extrema. A cell is
  cold-like (`subcell_eligible`) when its total variance is at most `12 b` and
  its one-sided velocity differences agree within a factor 2 on every axis
  (smooth flow). After each RK stage, a cone violation in a cold-like cell that
  is smaller than `b` is projected: HyQMOM cells become cold; Gaussian `C` has
  its negative eigenvalue clipped. Mass and momentum are unchanged; the kinetic
  energy change is returned to the gas (stage-1 changes with weight ½, because
  the dust energy is linear in `M2`). The solver counts these as
  `subcell_projections`; warm or rough cells are never projected.
* **MOOD.** Any other cell whose stage update leaves the cone is recomputed at
  first order, together with its neighbours, and the stage is redone. The
  all-first-order kinetic update is realizable under the CFL bound, so the loop
  terminates. The solver counts these as `mood_cells`.
* **Kinetic flux-vector splitting over a quadrature of the closure** (Desjardins,
  Fox & Villedieu 2008; Vikas et al. 2011): `F = F⁺(U_L) + F⁻(U_R)`, where `F±`
  sums `ρ w_α v_{α,d} φ(v_α)` over nodes moving right or left.
  * `gaussian`: six sigma points `u ± 3^{1/2} L_k`, weight ⅙, where `L` is
    the closed-form semi-definite Cholesky factor of `C`, pivoted so that the
    flux direction `d` comes first. Any square root reproduces the Gaussian
    moments through third order, so `F⁺ + F⁻` is the closure flux exactly. With
    this pivot the `d`-velocities are the three-point Gauss–Hermite rule:
    `u_d ± (3C_dd)^{1/2}` with weight ⅙ each, and `u_d` with weight ⅔. The node
    speed is therefore exactly `|u_d| + (3C_dd)^{1/2}`, no eigensolve is needed
    (4× faster than sigma points from `eigh`), and cold or rank-deficient `C`
    is handled exactly.
  * `hyqmom`: the three-node Gauss quadrature of the standardised moments. The
    n = 2 closure is exactly the third recurrence coefficient
    `a₂ = (a₀ + a₁)/2`, so the Jacobi matrix is
    `[[0, 1, 0], [1, S3, H4^{1/2}], [0, H4^{1/2}, S3/2]]`, and the quadrature
    reproduces the HyQMOM `M5` exactly.

  The update is a positive combination of nodes, so it is realizable under
  `Σ_d (dt/dx_d) max|v_{α,d}| ≤ ½`, and the signal speed is the exact node
  speed. **This deviates from the draft paper's Rusanov flux with split-state
  doubling.** In raw float64 moments, the split states `U ± F/a` subtract nearly
  equal quantities in cold streams, in two-point mixtures on the cone boundary,
  and in cells next to much heavier ones. Their round-off then grows without
  bound: in the crossing-stream test it stalled the scheme at 3500 substeps per
  0.01 time. The kinetic flux only ever subtracts a cell's own outflow from
  itself.
* SSP-RK3 (Shu–Osher) in time, with `dt` chosen from `C = 0.8` of the bound;
  each stage's projection energy enters the gas with the weight that stage
  carries in the final state (1/6, 2/3, 1). A stage that
  violates the bound or leaves the cone halves and redoes the substep. Dust
  transport is sub-cycled within the gas step.
* At the start of each substep, states that pass the round-off-tolerant cone
  test but not the exact one are projected onto the cone, and numerically cold
  HyQMOM cells (`v < 10⁻¹⁰` of its cancellation scale) are set exactly cold.
  Otherwise round-off in `q, r` accumulates across substeps until it reads as
  `K < 0`. Mass and momentum are untouched; the energy change (≲ `ρ·10⁻¹⁰ m²`) is
  returned to the gas, and the solver reports the counts.
* Eddy diffusion `∂_t U = ∇·(κ_d ∇U)`, with the same `κ_d` applied to every raw
  moment. Explicit, conservative, face-averaged `κ`, sub-cycled. For
  `dt Σ_d (κ₊+κ₋)/dx_d² ≤ 1` it is a convex combination, so it is realizable.
  `κ_d = c_κ σ_Δ Δ · x/(1+x)`: grains with small Stokes number ride the unresolved
  eddies, while the spreading of large-Stokes grains is already carried by their
  transported dispersion.

### 3.4 Local sources (exact, asymptotic preserving)

These act per cell over a duration `h` (half a step on each side of the
transport). Rates are frozen at the start of the update.

1. **SGS budget.** Write `σ_Δ² = 2e_s/(3ρ_g)`, `τ_Δ = Δ/σ_Δ`,
   `ε_Δ = σ_Δ³/Δ` and `x = ν τ_Δ`. The MT25 coefficient at zero magnetisation is
   `D = ε_Δ arctan x` (alternatives: `white`, `D = ε_Δ`; `finite_tau`,
   `D = ε_Δ/(1+x)`; `none`). The reservoir has two loss rates: dust work,
   `r_d = 3ρ_d D / e_s`, and decay, `r_ε = C_ε (e_s/ρ_g)^{1/2}/Δ`. Total loss is
   `L = e_s (1 − e^{−(r_d+r_ε)h})`, split in proportion to the rates. The dust
   receives `D_eff = L_d/(3ρ_d h)`, so the energy it receives equals what the
   reservoir loses, and `e_s` stays positive for any `h`.
2. **Mean drag with back-reaction.** The exact two-body solution: the
   centre-of-mass velocity is fixed, and `Δu = u_d − u_g` decays as
   `e^{−ν(1+ρ_d/ρ_g)h}`. Velocity components that the gas grid does not resolve
   relax toward zero with no back-reaction, which is an infinitely massive gas in
   those directions. The mean kinetic energy lost becomes gas heat.
3. **Dispersion (Ornstein–Uhlenbeck).** With `A = e^{−νh}`,
   `Σ = 2D_eff(1 − A²)/(2ν)`:
   * `gaussian`: `C' = A²C + ΣI`;
   * `hyqmom`: `v' = A²v + Σ`, `q' = A³q`, `r' = A⁴r + 6A²Σv + 3Σ²`,
     `σ_⊥²' = A²σ_⊥² + Σ`.

   This is the exact OU map of FARMM eq. (apupdate) with no geometric forcing,
   which is cone preserving (its eq. slackexact). The energy the reservoir loses,
   minus the change in dust random energy, is drag heat. Exactly:
   `Q_rand = L_d − ΔE_r ≥ 0`.
4. **Gas heat.** The gas gains `Q_mean + Q_rand + L_ε`, so the total energy is
   unchanged.

The drag is integrated exactly, so it never limits the timestep, and the terminal
states `u_d → u_g`, `v → D/ν` are recovered for any `νh` (FARMM's AP property).

Two refinements make the local step second order in `h` while keeping energy
exact:

* **Midpoint rates.** `r_d` and `r_ε` are evaluated at the midpoint
  `e_s e^{−(r_d+r_ε)h/2}` of a frozen-rate predictor. The exact loss
  `e_s(1 − e^{−(r_d+r_ε)h})` is still split in proportion to the rates, and the
  dust receives exactly its share.
* **Linear-in-time forcing (ETD2).** Within the step, `D(t) = D_a + s t`, with
  mean `D_eff` (so the dust receives exactly `L_d`) and slope `s` from
  `D(e_s(t_end)) − D(e_s(t_start))`, limited so that `D ≥ 0`. Then
  `Σ = 2[D_a φ(2ν, h) + s ψ(2ν, h)]`, with `ψ = (h − φ)/(2ν)`. With frozen
  forcing, a stiff dispersion tracks the step-averaged `D`, an `O(h)` lag. With
  ETD2 it tracks `D(t_end)/ν`, with the physical lag `−Ḋ/(2ν²)`. HyQMOM uses
  the same `Σ` in its OU map, which keeps Gaussian inputs Gaussian and the slack
  `K ≥ 0`.

Against an independent Radau integration of the uniform-box ODE, the coupled
solver converges at second order for `t_s = 10, 0.1, 10⁻⁴`
(`νΔt` up to about 200), with relative errors around 2×10⁻⁶ at 105 steps and
total energy conserved to 10⁻¹⁵.

## 4. SGS reservoir

    ∂e_s/∂t + ∇·(e_s u) = −(2/3) e_s ∇·u + P − ρ_g ε − 3ρ_d D (+ ∇·(κ_s ∇e_s))

* **Production `P`.** Each application of the spectral dissipation removes resolved
  kinetic energy `ΔK = K(U) − K(U_filtered)` with `K = |ρu|²/2ρ`. The filter
  conserves `∫E`, so without SGS that energy would become heat. We smooth `ΔK`
  with the spectral Gaussian `exp(−k²ℓ²/2)`, `ℓ = c_ℓ Δ`, take the positive part,
  renormalise to the global `∫ΔK`, and cap each cell at `f_cap ρ e_int`. That
  amount is transferred pointwise from `E` to `e_s`. This is the "spectral code
  knows exactly what its filter removes" advantage.
* **Transport and compression.** Conservative minmod-limited advection with the
  time-centred gas velocity, which is positive for `Σ_d (dt/dx_d)|u_d| ≤ ½`. The
  compression factor `exp(−(2/3)∇·u dt)` uses the spectral divergence, and its
  energy is exchanged pointwise with the gas `E`. The gas fluxes already did the
  `p_s` work on the gas, so the gas internal energy is unaffected in total.
* **Decay `ε`** = `C_ε k^{3/2}/Δ`, `k = e_s/ρ_g`, as part of the exact budget in §3.4.

## 5. Step

Strang splitting: local source (`dt/2`) → gas spectral step (records `ΔK` at
each dissipation half-step) → production transfer → SGS transport and compression
→ dust transport and eddy diffusion → local source (`dt/2`).
`dt = min(gas CFL with p_s, dust signal speed, SGS advection)`. Drag and the SGS
exchanges are exact, so they do not enter `dt`.

## 6. Numerics caveats

* Dust and SGS run on NumPy float64. Raw-moment cancellation makes realizability
  tests tolerance-based, relative to the magnitudes of the cancelled terms:
  `λ_min(C) ≥ −10⁻¹²(|u|² + Σ|C_ij|)`; for HyQMOM, `v ≥ −10⁻¹⁰ s_v` and
  `K ≥ −10⁻¹²·(linearised error of K)`. Here `s_v, s_q, s_r` are the sums of
  the absolute values of the terms subtracted in forming `v, q, r` from raw
  moments. The Gaussian test uses eigenvalues because minors of a near-singular
  `C` carry round-off of order ε, which no additive determinant tolerance can absorb.
* The Gaussian closure is a poor model of colliding cold streams, as expected.
  It thermalises them into a γ = 3 gas whose shocks accelerate without bound down
  steep density tails. HyQMOM keeps the variance within the kinetic bound. This is
  closure physics, not numerics: energy is conserved to 10⁻¹⁶ in both.
* Momentum in unresolved velocity components is not conserved, because the gas
  cannot hold it. Energy is.
* The white-noise OU representation over-estimates the long-time diffusivity of
  large-Stokes grains relative to Reeks' result (the MT25 model limitation).
* **Stiff-drag splitting.** Drag is exact, but gas and dust are transported by
  different schemes between the two local half-steps. While `ν dt ≫ 1` their
  velocities separate by `O(dt²)` per step and the drag pulls them back, so the
  coupled scheme is first order in that regime and recovers second order once
  `ν dt ≲ 1`. In the dusty wave at `ε = 1`, `t_s/T = 10⁻³` the L1 error is
  about 0.6% of the mode amplitude at `N = 512` (order 1.6 on the finest pair).
  Mixture-inertia (asymptotic-preserving) transport of the coupled pair would
  remove this; it is not implemented.
* **Cold dust order.** With primitive faces, sub-cell projection and WENO5-Z,
  smooth cold dust converges at about order 2 in L1 in the dusty wave (1.5 with
  minmod faces; first order with conserved faces only).
* **Velocity tails.** A single Gaussian per cell cannot hold the heavy tails of
  short-`t_s` dust in turbulence, where a cell mixes grains that still carry
  neighbouring cells' velocities. Against 256 particles per cell, the stacked
  kurtosis is about 6 at `t_s/τ_Δ = 0.05` and the dispersion differs by about
  30% (20% with minmod faces, whose extra numerical mixing happened to mimic
  the particles' nearest-grid-point spreading). At `t_s/τ_Δ = 1` the dispersion
  agrees within 5% and the kurtosis is 3.3;
  at `t_s/τ_Δ = 20`, within 0.4% and 3.0.
  HyQMOM carries the fourth moment and represents these tails in 1D; a
  multi-dimensional quadrature closure would be needed in 2D/3D.
* **2D turbulence feeds little SGS energy.** The 2D cascade is inverse, so the
  filter drains little kinetic energy and the reservoir stays small. The 3D
  tests are the meaningful ones for the production channel.

## 7. Charged dust in MHD (FARMM field-aligned phase)

`phrike.dusty_mhd.DustyMHDSolver` couples the same finite-volume dust and SGS
transport to `SpectralSolverMHD`. The gas state in **every** dimension is
`[rho, mx, my, mz, E, Bx, By, Bz]`: all three momenta exchange with the dust.
`E` includes thermal, kinetic and magnetic energy (`|B|²/2`). Epstein drag uses
thermal pressure after subtracting both kinetic and magnetic energy.
`dust.charge_to_mass` is signed, in code units, and defaults to zero. Euler
problems retain the hydro source map, including their unresolved-component
convention. No magnetic force is applied to an Euler state.

### Exact mean and covariance maps

Define `delta = u_d - u_g`, `eps = rho_d/rho_g`, `Omega = (q/m)|B|`, and
`bhat = B/|B|`. The local two-body equations are

```text
du_d/dt = -nu delta + Omega delta cross bhat
du_g/dt = -eps du_d/dt.
```

For a frozen field over duration `h`, the centre-of-mass velocity is unchanged.
With `f = 1 + eps`, split the slip into parallel and perpendicular parts:

```text
delta_parallel' = exp(-f nu h) delta_parallel
delta_perp' = exp(-f nu h) [cos(f Omega h) delta_perp
                           + sin(f Omega h) delta_perp cross bhat].
```

The reduced mass density is `mu = rho_d/f`, and drag heat is
`Q_mean = (mu/2)|delta|² [1 - exp(-2 f nu h)]`. Lorentz rotation does no work
on the relative motion. With back-reaction disabled, `f=1` and gas momentum is
fixed; the change in dust kinetic energy is exchanged with gas energy, as in
the hydro source. A zero field or zero charge removes the rotation. At zero
drag, the slip norm is conserved; for `nu h` large it relaxes to zero for any
gyrofrequency.

Dust fluctuations have no cell-level back-reaction in their damping rate.
The Gaussian covariance follows the exact map

```text
C' = exp(-2 nu h) R C R^T + Sigma_tensor,
R v = cos(Omega h) v + [1-cos(Omega h)] bhat (bhat dot v)
      + sin(Omega h) v cross bhat.
```

Pure gyration preserves the covariance trace and eigenvalues. Rotation and
non-negative Gaussian forcing preserve positive semidefiniteness. For an
isotropic coefficient, this reduces to the hydro OU map.

### Gyrotropic MT25 forcing and reservoir accounting

`sgs.d_model: mt25_magnetized` is the default when MHD dust has nonzero charge.
`mt25` remains isotropic and ignores magnetisation; `white`, `finite_tau` and
`none` retain their existing meanings. Using the infinite-inertial-range MT25
contract, with the same `sigma`, `tau`, `eps_D = sigma³/Delta` as above,
`x = nu tau` and `y = |Omega| tau`, the coefficients are

```text
D_parallel = eps_D atan(x)
D_perp = (eps_D/2) [pi - atan((1-y)/x) - atan((1+y)/x)]
D_tensor = D_parallel bhat bhat^T + D_perp (I - bhat bhat^T).
```

The code evaluates complementary `atan2` angles to avoid cancellation for
small `x`. At `x=0`, `D_perp/eps_D` is zero below `y=1`, `pi/4` at resonance,
and `pi/2` above it. At `y=0`, both coefficients equal `eps_D atan(x)`. When
`x` greatly exceeds both `1` and `y`, both approach `(pi/2) eps_D`. Thus weak
drag above resonance retains perpendicular forcing; at exactly zero drag
there is no stationary variance in that forced direction.

Because `D_tensor` commutes with `R`, constant forcing gives
`Sigma_tensor = 2 D_tensor phi(2 nu, h)`. The existing midpoint-rate reservoir
subcycling and positive, linear-in-time ETD2 forcing are applied separately to
the parallel and perpendicular coefficients. Their time-integrated work is
split from the same reservoir loss in proportion to the midpoint coefficients.
The dust work rate is `rho_d (D_parallel + 2 D_perp)`; the exact loss `L_dust`
equals the dust random-energy gain plus drag heat, including the componentwise
ETD2 correction. This keeps gas + SGS + dust energy consistent. For a held
reservoir, each directional variance is exactly
`v_i(t) = v_i(0) exp(-2 nu t) + (D_i/nu)[1-exp(-2 nu t)]`, continuously
extended to `v_i(0)+2 D_i t` at `nu=0`.

### Field-aligned HyQMOM and scope

HyQMOM is restricted to 1D straight fields along x:
`|By|, |Bz| <= 1e-12 |B|` in every cell, otherwise the solver raises. Its
parallel `M0..M4` hierarchy is unchanged by rotation; transverse mean slip
rotates, while isotropic transverse dispersion is unchanged by rotation.
The OU map uses `Sigma_parallel` in `v, q, r` and `Sigma_perp` in
`sigma_perp²`. The existing layout contains **eight** fields: five parallel
moments, two transverse momenta, and one transverse energy. No redundant ninth
field is introduced. A transverse Alfvén perturbation is an oblique field,
so that problem uses the Gaussian closure.

The Gaussian closure supports arbitrary fields in 1D/2D/3D, including exact
gyro-rotation in 2D/3D. This phase performs **no guiding-centre averaging**.
There is no gyro-resolution timestep limit: `Omega dt` is integrated exactly
in each frozen local map. Accuracy of splitting with spatially evolving fields
still depends on the timestep; exact local gyration is not an accuracy claim
for arbitrarily large steps in an inhomogeneous flow.

The supported gas equations are ideal/resistive MHD with fixed RK2/RK4 on
NumPy float64 periodic Fourier grids. Hall, ambipolar and shearing-sheet
sources raise. `SGSMHDEquations1D/2D/3D` add `p_sgs` to diagonal momentum
fluxes and `p_sgs u_d` to directional gas energy fluxes (`u_d` here means the
gas velocity in direction `d`). Their signal-speed bound is
`max_d |u_d| + sqrt[(gamma p + |B|² + (5/3)p_sgs)/rho]`.

The gas step mirrors `_reference_fixed_step`: spectral dissipation half-step,
MHD RK substages and magnetic projection, then accepted-state stabilization
and positivity limiting. Kinetic loss `K=|m|²/(2 rho)` is measured at both
filter halves and transferred to SGS with the existing smoothing/cap rule.
**Magnetic energy removed by the spectral filter remains in E as gas heat.**
The existing magnetic projection holds thermal pressure fixed; divergence-free
periodic evolution requires only round-off projection corrections. An SGS
magnetic reservoir is not modelled.

Registered examples are `configs/charged_dust_box.yaml` (uniform arbitrary
field and slip), `configs/dusty_alfven1d.yaml` (cold charged dust, SGS off),
and `configs/dusty_orszag_tang2d.yaml` (charged OT with SGS). Box gas and dust
velocities and `initial_conditions.magnetic_field` each have three components.
The magnetic validation scripts are `charged_gyration.py`, `dusty_alfven.py`
and `dusty_orszag_tang.py`, registered with `scripts/dust_validation/run_all.py`.
The OT example and validation use an explicitly broad order-two spectral
filter (cutoff rate 500, onset zero). Their conservation and realizability
checks qualify that filtered problem, rather than shock resolution or an
ideal-MHD accuracy limit.
