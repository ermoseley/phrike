# Supporting scripts

The tools in this directory are source-tree utilities, not installed console
commands. Run them from a PHRIKE checkout after installing the project.

The main `phrike` command remains the canonical way to run a problem. In
particular, pass `--video` to render a run and encode its frames; this replaces
the retired one-off video scripts.

```bash
python -m phrike orszag_tang2d --config configs/orszag_tang2d.yaml --backend cpu --video
```

## Post-processing saved 3-D output

`postprocess.py` handles the maintained snapshot workflows:

```bash
# Inspect the output directory selected by the configuration.
python -m scripts.postprocess list --config configs/turb3d.yaml

# Plot a central density projection and slice from a saved snapshot.
python -m scripts.postprocess plot outputs/turb3d/snapshot_t0.100000.npz \
  --config configs/turb3d.yaml

# Reconstruct tracer density from the latest tracer snapshot in that output directory.
python -m scripts.postprocess tracer-density --config configs/turb3d.yaml
```

The snapshot metadata supplies the domain dimensions. Uniform tracer lattices
retain unwrapped coordinates and topology for the periodic simplex-cell density
estimator; random tracers use the configured histogram grid.

## Development diagnostics

```bash
# Benchmark the cost of tracer advection on a chosen backend.
python -m scripts.benchmark_tracers --backend numpy --grid 64 --steps 20

# Validate tracer density against a smooth compressive flow on Apple Metal.
python -m scripts.validate_tracers --backend torch --device mps --grid 32

# Run the full 1-D circularly polarized Alfvén-wave diagnostic.
python -m scripts.validate_alfven1d
```

The benchmark intentionally calls an internal solver step and is therefore a
developer measurement, not part of the stable public API or test suite.

## Retired root helpers

| Retired helper | Supported replacement |
| --- | --- |
| `visualize_turb3d_snapshot.py` | `python -m scripts.postprocess plot ...` |
| `analyze_turb3d_lpsse.py` | `python -m scripts.postprocess tracer-density ...` |
| `phrike2vid.py`, `create_video_python.py` | Run the problem with `python -m phrike ... --video` |
| `run_orszag_tang2d.py` | `python -m phrike orszag_tang2d --config configs/orszag_tang2d.yaml` |
| ad-hoc `test_*.py` demos | Version-controlled configs and `python -m pytest -q` |
