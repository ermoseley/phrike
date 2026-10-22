# Contributing to PHRIKE

PHRIKE is active experimental research software. Small, well-tested changes
are preferred over broad refactors.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Python 3.10 or newer matches the interpreters used in continuous
integration.

Before opening a pull request, run:

```bash
python -m pytest -q
python -m sphinx -W -b html docs docs/_build/html
python -m build --wheel --outdir dist
```

On Apple Silicon with PyTorch installed, the default backend is
Metal/MPS in single precision, including when a configuration asks for
double precision. Tests pass an explicit backend into the grid
(`backend="numpy"`, or a Torch CPU device). `python -m pytest -q` on a
machine with Metal available is not the same run as the default CI job,
which does not install PyTorch. CI also has a job that installs a
CPU-only PyTorch wheel and runs the suite. Lint is not a required check.

Please keep generated outputs, Numba caches, local run directories, and
machine-specific configuration out of commits. For numerical changes, state
the problem, configuration, backend, and validation evidence in the pull
request description.
