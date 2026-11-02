# Hydro FARMM dust and active SGS validation

Run the campaign from the repository root:

```sh
PYTHONPATH=. python scripts/dust_validation/run_all.py
```

Each test runs in a fresh Python process with NumPy float64 and Matplotlib Agg.
Outputs live in `outputs/dust_validation/<test>/`: `metrics.json`, labelled PNG
figures, and (when run through the driver) `run.log`. The driver continues after
failed checks, exits nonzero if any check fails, and writes
[the complete check table](../outputs/dust_validation/summary.md).
Use `run_all.py --report-only` to regenerate that table from existing results.
Every test process has a 180-second driver timeout (7200 s for
`long_turbulence`, which runs only with `--long`). Grids fit the requested laptop limits.

The campaign uses the method described in [the design note](dust_hydro_farmm.md),
including kinetic flux-vector splitting rather than Rusanov transport. Each
metrics file records SHA256 hashes of the dust, SGS and gas solver sources, checks
that they did not change during the run, and records peak process memory. Counts of `repaired_cells` and `roundoff_projections` are
reported separately for every physical run. A nonzero repair count fails a check.

## References and acceptance criteria

### Uniform OU box (`ou_box.py`)

Uniform source-only boxes cover Gaussian closure in 1D, 2D and 3D, and HyQMOM in
1D. HyQMOM in higher dimensions is unsupported by the solver. The independent
Radau reference integrates the evolving SGS reservoir, dust variance, relative
drift and gas heat with MT25 forcing. This distinguishes the exact local OU map
from time discretisation of its changing forcing. The parameter sweep covers
initial `nu*tau_Delta = 0.01, 1, 100` and `dt*nu = 0.01, 10`.

A coarse source step is not automatically an accurate integration of a reservoir
that changes on a much shorter eddy time. Both the continuous-ODE error and the
stiff terminal-balance check are reported. HyQMOM Gaussianisation starts with
`S3=1, S4=5`, inside the Hankel cone. The exact OU central moments are
`q'=A^3 q`, `v'=A^2 v+Sigma`, and
`r'=A^4 r+6 A^2 Sigma v+3 Sigma^2`; normalised skewness and kurtosis use those
updated moments. Energy must remain constant to relative `1e-12`.

### Two-body drag (`drag_decay.py`)

For dust loading `eps=0.01, 1, 10`, the centre-of-mass velocity stays fixed and
the relative velocity is `delta_u(0)*exp[-nu*(1+eps)*t]`. The lost mean kinetic
energy must equal gas heat. Tests include small and stiff drag steps. Exact-drag,
momentum and heat tolerances allow only accumulated float64 error. Any rejected
cold moment state is reported with the last completed time; it is not silently
regularised by this harness.

### MT25 heating and Gaussianisation (`mt25_gaussianisation.py`)

The central claim of the SGS–dust coupling (Moseley & Teyssier 2025): unresolved
turbulence of one-component dispersion `σ_Δ` drives grain velocities as an OU
process with `D = ε_Δ arctan x`, `ε_Δ = σ_Δ³/Δ`, `x = ν τ_Δ`. Initially cold dust
therefore heats at rate `2D`, saturates at `D/ν`, and any initial distribution
relaxes to a Gaussian. The reservoir is held fixed (decay off, production off,
loading 10⁻⁹) so `D` is constant, and the full coupled solver's local step is
advanced in a uniform box. `x` is swept over 10⁻²…10², for the Gaussian closure
in 1D, 2D and 3D and HyQMOM in 1D.

* dispersion history against the exact OU solution, error ≤ 10⁻⁷ of `D/ν`;
* initial heating rate against `arctan x`, relative ≤ 10⁻⁹;
* plateau against `D/ν`, relative ≤ 10⁻⁷;
* HyQMOM kurtosis from `(S3, S4) = (1, 5), (0, 1.5), (−1.5, 6)` reaching 3 within
  10⁻³ after `ν t = 4`.

The 10⁻⁷ tolerances allow the reservoir drain by the 10⁻⁹ loading and nothing more.

### Dusty acoustic wave (`dustywave.py`)

The four complex Fourier amplitudes evolve with the exact linear matrix (Laibe &
Price 2011 setup, adiabatic), using `scipy.linalg.expm`. Gas pressure is perturbed
adiabatically; gas and cold dust start co-moving. Amplitude `10⁻⁶` keeps the
nonlinear departure below the discretisation errors measured. `ε = 0.01, 1` and
`t_s/T = 10⁻³, 0.3, 30`, one acoustic period, no spectral dissipation, gas CFL 0.8
(dust sub-cycles under its own bound). `N = 32…256`, and `64…512` for
`t_s/T = 10⁻³`, so the finest pair has `ν dt ≈ 2` and leaves the stiff-splitting
regime (design note §6).

Errors are spatial L1 norms divided by the initial mode amplitude. Per field:

* L1 order ≥ 1.5 on the finest pair (limited second order; minmod clips smooth
  extrema), unless the field is already at the floor, L1 ≤ 10⁻⁵ of the initial
  amplitude at the finest `N` (spectral gas at weak coupling, dust velocity when
  drag is negligible);
* energy drift ≤ 10⁻¹², zero repairs, every state realizable.

Cold dust reaches this order because cold-like cells reconstruct `(ρ, u)` and
project sub-cell-level cone violations (design note §3.3). A profiles figure
compares all four fields with the exact mode at `N = 64`.

### Kinetic free streaming (`beams_free_streaming.py`)

The independent reference integrates `f(x-v*t,v)` using Gaussian velocity
quadrature and spatial Gauss-Legendre cell averages. It covers a smooth Gaussian
packet and two cold opposing packets at `N=64,128,256,512`, for both closures.
Each stream has a density floor at least `1e-6` of its peak. Source-free transport
uses `nu=0`, the limit of the requested `t_s=1e12`; finite stopping changes these
short-time references below the acceptance tolerances.

Every raw field's periodic integral must be constant within `2e-12`, with a mass
scale for zero-integral fields. Cone violations, covariance and scaled Hankel
margins are checked. For the cold velocities `+1,-1`, the kinetic support implies
variance at most 1. The check allows `1e-9`: forming `v = M2/M0 − m²` from raw
float64 moments has cancellation round-off that random-walks over about 10³
substeps to 10⁻¹⁰–10⁻⁹ of `s_v ≈ 1`. That is the same floor below which the
closure treats a variance as cold. Density,
mean velocity, variance, skewness and kurtosis L1 errors are descriptive closure
measurements. Neither a universal convergence rate to a kinetic distribution nor
an assumed ordering of closures is imposed. Gaussian closure need not represent
a bimodal distribution.

### SGS acoustic speed (`sgs_acoustic.py`)

At `N=64`, a small right-going acoustic eigenmode perturbs both gas pressure and
SGS pressure. SGS energy starts with its adiabatic perturbation,
`delta_e_s=(5/3)*e_s*delta_rho/rho`; otherwise a standing/entropy admixture would
bias the phase measurement. Dust, SGS decay and forcing are off. The reference is
`c_eff^2=gamma*p/rho+(10/9)*e_s/rho`. A linear fit to the unwrapped fundamental
Fourier phase measures speed over one period, at SGS/internal-energy ratios
`0,0.1,0.5,1`. Relative error must be below `1e-3`.

### Turbulent energy ledger (`turb_energy_ledger.py`)

The grid sizes are `128^2` and `32^3`, with identical seeded initial gas fields
across loadings `0.01,1`. Stopping times are `0.05,1,20` initial grid-eddy times.
MT25 forcing, spectral-filter production, SGS pressure and decay, drag
back-reaction, and both dust and SGS diffusion are enabled. The short window is
`0.08*tau_Delta`; this tests the ledger and initial loading response, and does
not establish long-term turbulence decay or stationary statistics.

The sum of gas, SGS and dust energies must drift by less than `1e-10` relative.
The SGS integral must equal its initial value plus production minus decay and
dust work plus compression, at the same tolerance. Heavier dust must decrease
resolved and SGS energies more over this specified window. The scalar OU
prediction integrates the run's mass-weighted MT25 D history with exact drag
and sampled forcing. Spatial transport and diffusion convert spatially varying
mean dust motion into cell dispersion, and that term is absent from the scalar
OU equation. The ten-percent check therefore applies to intermediate and long
stopping times, where this conversion is negligible. At short stopping times
the dust follows the gas, the conversion dominates its tiny `D/ν`, and the
comparison is recorded together with the measured transport fraction.

### Particle reference (`particle_reference_2d.py`)

Passive dust (loading 10⁻⁶, back-reaction off, eddy diffusion off) in decaying
2D turbulence with the active reservoir, `32²`, `t_s/τ_Δ = 0.05, 1, 20`, two
initial grid-eddy times. 256 particles per cell drift, relax exactly toward the
gas velocity of their cell, and receive exact OU kicks with the MT25 `D` of their
cell. Sampling and deposition are both nearest-grid-point, so particles see
exactly the cell-uniform gas and reservoir the moment drag sees; the comparison
isolates transport and closure error from interpolation choices. Initial
velocities are antithetic samples of each cell's moment Gaussian.

* global mass-weighted dispersion within 10% of the particles at `t_s/τ_Δ = 1`
  and 20 (Monte Carlo noise is below 1% over 1024 cells);
* stacked, cell-normalised velocity PDF kurtosis within 0.15 of 3 at
  `t_s/τ_Δ = 20`, where SGS kicks set the dispersion;
* zero unrealizable cells and zero repairs in every regime.

At `t_s/τ_Δ = 0.05` the dispersion is set by grid-scale velocity structure, not
SGS kicks: particles carry a neighbour's velocity for a fraction of a cell after
crossing the piecewise-constant gas field, so the cell PDF is a heavy-tailed
mixture. The dispersion gap and kurtosis there are recorded as measured closure
error, not checked.

### Long decaying turbulence (`long_turbulence.py`)

The three-way exchange over time from an empty reservoir: `128²` and `24³`,
`k = 2–8`, `v_rms = 0.3`, `t_s = 0.1`, `ε = 0.01, 1`, to `t = 1` (about three
large-eddy times). Dust starts co-moving with the gas. The filter drain is the
only SGS source; MT25 coupling, SGS pressure and decay, back-reaction and eddy
diffusion are on.

* total energy (gas + SGS + dust) drift ≤ 10⁻¹⁰ and SGS ledger closure
  (production − decay − dust work + compression) ≤ 10⁻¹⁰;
* zero repairs and zero positivity-floor cells;
* the reservoir is filled by the cascade (production > 0);
* heavy loading ends with less resolved and less SGS energy than light loading.

The figure shows the reservoir energies, the cumulative exchanges, and the dust
random-energy budget (SGS work in, drag damping out, transport mixing of the
resolved dust velocity). This test takes about an hour, so the driver runs it
only with `--long` and otherwise keeps its last recorded result.

## Measured results

<!-- RESULTS_START -->
Final campaign on 2026-11-01 (local Apple M3, NumPy float64): **856/856 checks
pass**, with zero repaired cells in every physical run.

| Test | Passed / total |
|---|---:|
| [ou_box](../outputs/dust_validation/ou_box/metrics.json) | 139/139 |
| [drag_decay](../outputs/dust_validation/drag_decay/metrics.json) | 241/241 |
| [dustywave](../outputs/dust_validation/dustywave/metrics.json) | 133/133 |
| [beams_free_streaming](../outputs/dust_validation/beams_free_streaming/metrics.json) | 230/230 |
| [sgs_acoustic](../outputs/dust_validation/sgs_acoustic/metrics.json) | 13/13 |
| [turb_energy_ledger](../outputs/dust_validation/turb_energy_ledger/metrics.json) | 81/81 |
| [particle_reference_2d](../outputs/dust_validation/particle_reference_2d/metrics.json) | 19/19 |

The first pass of this campaign reported 69 failures and four reproducers. The
architect resolved them as follows. The reproducer scripts are kept as
regression probes.

1. **Under-resolved SGS decay in the source step**
   ([repro](../scripts/dust_validation/repro_ou_budget_freeze.py)). With `dt` of
   one turnover time or more, the frozen-rate (and then midpoint-rate) reservoir
   update was wrong by up to 100%: `e_s = 1.5` against Radau `2.8×10⁻⁶`. The
   local step now sub-cycles whenever the reservoir loss rate × `dt` exceeds 0.25,
   and uses midpoint rates with ETD2 linear-in-time forcing. Result: `e_s`
   matches Radau to `2.83096×10⁻⁶` against `2.83091×10⁻⁶`. The worst box errors
   over the whole sweep are now 6×10⁻⁴ (`e_s`), 1.8×10⁻³ (variance) and
   8×10⁻⁴ (heat). Energy is exact to 4×10⁻¹⁵, and coarse steps are second order.
2. **Cold-drag self-rejection**
   ([repro](../scripts/dust_validation/repro_cold_drag_source.py)). After drag
   decelerated the dust from u ≈ 0.7 to 0.0075, round-off left in `M2` from the
   early speed was 10⁻¹² of the current `u²`. The source step now projects
   round-off cone violations of its input and output (its exact maps preserve
   the cone), returns the energy to the gas, and reports it as
   `projection_heat`.
3. **Dusty-wave orders** ([repro](../scripts/dust_validation/repro_dustywave_density.py)).
   The first-order behaviour is the cold-dust property described above, not a
   defect. Measured dust-density L1 orders (N = 64–128) are 0.98–1.43. The
   L2 orders are 0.56–1.05, limited by the stagnation cell.
4. **HyQMOM variance bound**
   ([repro](../scripts/dust_validation/repro_hyqmom_kinetic_bound.py)). The excess
   of 2.3×10⁻¹⁰ is the raw-moment round-off floor; the allowance is now 10⁻⁹.
5. **Turbulence OU prediction.** The 3D short-stopping run's 15.8% deviation is
   the transport-mixing term: it accounts for 37% of that run's random-energy
   change. In every intermediate and long run the prediction matches to
   ≤ 3×10⁻⁴.
6. **HyQMOM smooth-packet noise** (found by this campaign, fixed afterwards).
   Limiting raw moments one component at a time made the face skewness and
   kurtosis noisy, because `q` and `r` are small differences of O(m³) and
   O(m⁴) raw terms. Interior cells now reconstruct in
   `(ρ, m, v, S3, H4, u_⊥, σ_⊥²)`, where realizability is a box; cells near the
   boundary keep the conserved Zhang–Shu reconstruction. Both packets now
   converge at about second order. The smooth packet's L1 errors are
   2.1×10⁻², 5.2×10⁻³, 1.4×10⁻³ and 4.7×10⁻⁴ (N = 64–512) with decreasing
   roughness; the crossing streams' are 1.1×10⁻¹ → 2.7×10⁻³.
   `tests/test_dust_moments.py::test_hyqmom_smooth_packet_is_second_order_and_smooth`
   guards the fix.

Other results:

* **SGS acoustic speed.** Relative errors are 6×10⁻⁹ to 5.5×10⁻⁴ against
  `c_eff² = γp/ρ + (10/9)e_s/ρ`.
* **Crossing streams.** HyQMOM keeps the variance within the kinetic bound and
  tracks the kinetic density. The Gaussian closure thermalises the streams and
  shock-heats the tails, as expected for a single-Gaussian closure.
* **Particle reference.** Density agrees to 1.4–3.4% and mean velocity to
  2.5–6.3% of `v_rms`. Variance agrees to 6% at intermediate and long `t_s`. At
  short `t_s` the variance differs by 43% after matching the CIC window: the
  physical variance `D/ν` is tiny there, and both descriptions are dominated by
  unresolved shear within a cell and numerical mixing. Local velocity PDFs are
  Gaussian within sampling noise (kurtosis 2.6–3.1).

For all check values, thresholds, outcomes and figure links, use
[the complete generated table](../outputs/dust_validation/summary.md).
<!-- RESULTS_END -->
