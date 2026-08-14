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

Before opening a pull request, run:

```bash
python -m pytest -q
python -m sphinx -W -b html docs docs/_build/html
python -m build --wheel --outdir dist
```

Please keep generated outputs, Numba caches, local run directories, and
machine-specific configuration out of commits. For numerical changes, state
the problem, configuration, backend, and validation evidence in the pull
request description.
