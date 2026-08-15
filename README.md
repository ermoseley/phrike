# PHRIKE

[![CI](https://github.com/ermoseley/phrike/actions/workflows/ci.yml/badge.svg)](https://github.com/ermoseley/phrike/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PHRIKE is an experimental pseudo-spectral research code for compressible
hydrodynamics and ideal magnetohydrodynamics (MHD). It is a compact platform
for investigating spectral discretizations, explicit adaptive integration, and
portable NumPy/PyTorch backends—not a production CFD package.

## What is implemented

- 1D, 2D, and 3D hydrodynamic and ideal-MHD problem setups.
- Fourier and Legendre bases, with spectral filtering and optional artificial
  viscosity for exploratory shock problems.
- Fixed-step RK2/RK4 and embedded RK23, RK45, and Fehlberg RKF78 integration.
- NumPy CPU execution plus optional PyTorch CPU, CUDA, and Apple Metal/MPS
  backends.
- YAML-configured example problems, including acoustic waves, Kelvin-Helmholtz,
  Alfvén waves, and Orszag-Tang.

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

# Run the checked test suite.
python -m pytest -q
```

For an optional accelerator backend, install the Torch extra and select a
backend explicitly:

```bash
python -m pip install -e ".[torch]"
python -m phrike alfven1d --config configs/alfven1d.yaml --backend metal
# Use --backend cuda on an NVIDIA system.
```

## Validation and scope

The automated suite checks adaptive Runge-Kutta tableaus and controller order,
backend selection/error handling, and Torch CPU tensor behavior. It is a smoke
and unit-test suite, not a full numerical-validation campaign.

Use the code as experimental research software. In particular:

- Claims of performance, convergence, and accelerator parity require a
  hardware- and configuration-specific measurement.
- Pseudo-spectral shock problems require filtering/viscosity choices to be
  validated for the case at hand.
- Non-periodic Legendre workflows are still exploratory; the 3D Legendre path
  is not implemented.
- CUDA and Metal/MPS hardware are not exercised in continuous integration.

See [the documentation source](docs/) for installation, a walkthrough, and
development checks.

## Repository layout

- `phrike/` — solver, equations, grids, and problem definitions.
- `configs/` — version-controlled YAML configurations.
- `tests/` — focused automated checks.
- `scripts/` — source-tree post-processing and development diagnostics.
- `docs/` — Sphinx documentation source.

## Contributing

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) and
include the command(s) used to validate a change.

## License

PHRIKE is released under the [MIT License](LICENSE).
