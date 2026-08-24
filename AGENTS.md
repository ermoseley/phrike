# AGENTS.md

## Cursor Cloud specific instructions

PHRIKE is a Python pseudo-spectral CFD/MHD solver exposed as a CLI (`phrike`) and a
Python API (`phrike.run_simulation`). There are no web servers, databases, or
long-running background services — simulations run as one-off processes.

### Environment
- Dependencies are installed into a virtualenv at `.venv/` (gitignored). The startup
  update script creates it and runs `pip install -e ".[dev]"`.
- Run tools via the venv without activating, e.g. `.venv/bin/phrike ...`,
  `.venv/bin/pytest`, `.venv/bin/ruff ...` (or `source .venv/bin/activate` first).
- `ffmpeg` is available system-wide and is only needed for `--video` output.

### Running simulations (the core functionality)
- Standard run: `.venv/bin/phrike <problem> --config configs/<file>.yaml`
  (e.g. `.venv/bin/phrike sod --config configs/sod.yaml`). Problems and matching
  configs are listed by `.venv/bin/phrike --help` and live in `configs/`.
- Outputs (`.npz` snapshots, `.png` field plots, `.mp4` videos) are written to
  `outputs/` (gitignored), overridable with `--outdir`.
- Video is off by default; enable with `--video`. Note: several configs (e.g.
  `configs/sod.yaml`) default `video.codec` to `h264_videotoolbox`, which is
  macOS-only. On Linux pass `--video-codec libx264`.
- GPU/`--backend torch` requires the optional `[torch]` extra (not installed by
  default); the default `numpy` CPU backend works out of the box.

### Lint / test caveats
- Lint: `.venv/bin/ruff check phrike/` runs but the existing codebase currently has
  many pre-existing violations (it exits non-zero). This is the repo's current state,
  not a setup problem.
- Tests: there is no real test suite. `pyproject.toml` excludes a `tests/` dir that
  does not exist, and the root-level `test_*.py` files are standalone scripts (not
  pytest-compatible), so `.venv/bin/pytest` errors on collection. Validate changes by
  running actual simulations via the CLI/API instead.
