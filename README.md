# PHRIKE

[![CI](https://github.com/ermoseley/phrike/actions/workflows/ci.yml/badge.svg)](https://github.com/ermoseley/phrike/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PHRIKE is an experimental pseudo-spectral research code for compressible
hydrodynamics and ideal magnetohydrodynamics (MHD). It is a compact platform
for investigating spectral discretizations, explicit adaptive integration, and
portable NumPy/PyTorch backends—not a production CFD package.

## What is implemented

- 1D, 2D, and 3D hydrodynamic and ideal-MHD problem setups.
- Fourier and Legendre bases, plus a 1D Chebyshev–Lobatto basis, with hard
  2/3 Fourier dealiasing, 3/2 polynomial flux overintegration, optional
  timestep-aware spectral dissipation, and artificial viscosity for
  exploratory shock problems.
- 1D fixed/Dirichlet, outflow, and wall boundaries for hydro and MHD,
  including independent endpoint and per-variable settings.
- 2D Fourier-x/SBP-y grids with periodic x and approximately open y faces,
  plus midpoint-resolvent ambipolar diffusion for 2D MHD.
- Fixed-step RK2 and RK4, and embedded RK23, RK45, and rk78 (Fehlberg 7(8)).
- NumPy CPU execution plus optional PyTorch CPU, CUDA, and Apple Metal/MPS
  backends.
- Lagrangian tracer particles with tracer-density reconstruction.

## Quick start

```bash
git clone https://github.com/ermoseley/phrike.git
cd phrike
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# A CPU smoke run; outputs are kept outside the repository.
python -m phrike acoustic1d --config configs/acoustic1d.yaml \
  --backend cpu --outdir /tmp/phrike-acoustic

# Run the test suite. Tests pass an explicit backend.
python -m pytest -q
```

`python -m phrike --help` lists every registered problem and option. Video
output is off by default; pass `--video` to render frames and encode them with
ffmpeg.

For an optional accelerator backend, install the Torch extra and select a
backend explicitly:

```bash
python -m pip install -e ".[torch]"
python -m phrike alfven1d --config configs/alfven1d.yaml --backend metal
# Use --backend cuda on an NVIDIA system.
```

If PyTorch is installed and the backend is left on `auto`, Apple Silicon
uses Metal/MPS. Metal/MPS executes in single precision.

## Example problems

| Problem | Name | Example configuration |
| --- | --- | --- |
| Sod shock tube | `sod` | `configs/sod_chebyshev_1d.yaml` |
| Shu–Osher shock–entropy interaction | `shu_osher1d` | `configs/shu_osher1d.yaml` |
| Acoustic wave | `acoustic1d` | `configs/acoustic1d.yaml` |
| Gaussian wave packets | `gaussian_wave1d`, `gaussian_wave2d`, `gaussian_wave2d_y` | `configs/gaussian_*.yaml` |
| Hydrostatic equilibrium | `hse_1d` | `configs/hse_1d_test.yaml` |
| Rayleigh–Taylor instability | `rti_1d`, `rti` | `configs/rti_1d_athena.yaml`, `configs/rti_athena.yaml` |
| Kelvin–Helmholtz instability | `khi2d` | `configs/khi2d.yaml` |
| Taylor–Green vortex | `tgv3d` | `configs/tgv3d.yaml` |
| Decaying 3D turbulence | `turb3d` | `configs/turb3d.yaml` |
| Circularly polarized Alfvén wave | `alfven1d`, `alfven3d` | `configs/alfven1d.yaml`, `configs/alfven3d.yaml` |
| Brio–Wu MHD shock tube | `brio_wu1d` | `configs/brio_wu1d.yaml` |
| Orszag–Tang vortex | `orszag_tang2d` | `configs/orszag_tang2d.yaml` |
| Ambipolar-diffusion shear instability | `adsi2d` | `configs/adsi2d.yaml` |
| Decaying 3D MHD turbulence | `mhd_turb3d` | `configs/mhd_turb3d.yaml` |

For the 1D shock examples:

```bash
python -m phrike sod --config configs/sod_chebyshev_1d.yaml --backend cpu
python -m phrike brio_wu1d --config configs/brio_wu1d.yaml --backend cpu
```

The validated shock combination is Chebyshev with fixed endpoints.
`--basis legendre` selects the other polynomial basis. `--bc outflow` and
`--bc wall` are experimental: their endpoint states drift before any wave
arrives, most strongly on Legendre grids, and PHRIKE warns when they are used.

See [1D boundary configuration and validation](docs/validation.rst) for the
endpoint semantics, fixed-value overrides, and measured shock-profile errors.

The single-layer ambipolar-diffusion shear instability example retains all
three velocity and magnetic components on a two-dimensional grid. Its open
ideal-MHD boundary flux and zero tangential diffusive electric field are
separate closures; read [validation and limitations](docs/validation.rst)
before interpreting growth rates or boundary-sensitive results.

## Spectral dissipation and filtering

The shipped Fourier configurations use a hard 2/3 dealias projection. Optional
smooth damping is configured separately with
`integration.spectral_dissipation`: set `enabled`, an even `order`, and exactly
one of `rate` or `e_folding_time_at_cutoff`. `rate` is the damping rate at
the cutoff wavenumber; the e-folding time of that mode is `1/rate`. The
exponential is integrated in simulation time with a Strang half-step on
either side of the Runge–Kutta step. Optional `onset_fraction` confines damping to the corresponding upper
fraction of the retained band (for example, `0.8` leaves the lower 80% exactly
untouched). Legendre and Chebyshev grids instead use the independently named
`integration.modal_filter`; its `alpha` remains a dimensionless modal-filter
strength.

## Validation and scope

The automated suite checks adaptive Runge–Kutta tableaus and controller order,
basis operators, boundary handling, shock-tube profiles, backend selection and
error handling, and Torch CPU tensor behavior. It is a unit and regression
suite, not a full numerical-validation campaign; measured results and their
limits are recorded in [docs/validation.rst](docs/validation.rst). The
equations that are integrated, including the CFL definition and the
unit convention, are in [docs/equations.rst](docs/equations.rst). YAML
keys and the accepted integrator names are in
[docs/configuration.rst](docs/configuration.rst).

Use the code as experimental research software. In particular:

- Claims of performance, convergence, and accelerator parity require a
  hardware- and configuration-specific measurement.
- Pseudo-spectral shock problems require filtering/viscosity choices to be
  validated for the case at hand.
- Non-periodic polynomial workflows are still exploratory; the 3D Legendre path
  is not implemented.
- CUDA and Metal/MPS hardware are not exercised in continuous integration.
  A separate job installs a CPU-only PyTorch wheel and runs the Torch tests.

## Repository layout

- `phrike/` — solver, equations, grids, and problem definitions.
- `configs/` — version-controlled YAML configurations.
- `tests/` — automated checks.
- `scripts/` — post-processing and development diagnostics
  (see [scripts/README.md](scripts/README.md)).
- `docs/` — Sphinx documentation source.

## Citing

If you use PHRIKE in published work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## Contributing

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) and
include the command(s) used to validate a change.

## License

PHRIKE is released under the [MIT License](LICENSE).
