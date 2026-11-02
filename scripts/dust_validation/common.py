"""Small shared reporting and solver setup for the standalone campaign scripts."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/phrike-dust-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
from pathlib import Path
import subprocess
import time
import resource
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [ROOT/'phrike'/name for name in ('dust_moments.py', 'dust_sources.py',
                'dusty_hydro.py', 'dusty_mhd.py', 'sgs.py', 'solver.py', 'equations.py')]


def source_hashes():
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in SOURCE_FILES}


def solver(ndim=1, n=32, *, closure='gaussian', eps=0.01, ts=1., sgs_on=False,
           dust_on=True, filtering=False, cfl=0.3, **kwargs):
    from phrike.grid import Grid1D, Grid2D, Grid3D
    from phrike.dusty_hydro import (DustyHydroSolver, SGSEulerEquations1D,
                                  SGSEulerEquations2D, SGSEulerEquations3D)
    from phrike.problems.problem_list.dusty_box import dust_state
    opts = dict(backend='numpy', precision='double', fft_workers=1,
                filter_params={'spectral_dissipation': dict(enabled=filtering, p=4,
                               rate=10., onset_fraction=0.)})
    grid = (Grid1D(n, 1., **opts) if ndim == 1 else
            Grid2D(n, n, 1., 1., **opts) if ndim == 2 else
            Grid3D(n, n, n, 1., 1., 1., **opts))
    eq = (SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D)[ndim-1](1.4)
    dust_cfg = dict(enabled=dust_on, closure=closure, backreaction=True,
                    stopping={'model': 'constant', 't_s': ts})
    sgs_cfg = dict(enabled=sgs_on, pressure=True, production='none', decay=False,
                   d_model='mt25')
    dust_cfg.update(kwargs.pop('dust_config', {}))
    sgs_cfg.update(kwargs.pop('sgs_config', {}))
    s = DustyHydroSolver(grid, eq, cfl=cfl, dust_config=dust_cfg, sgs_config=sgs_cfg, **kwargs)
    rho = np.ones(s.shape)
    s.U_gas = eq.conservative(rho, *[np.zeros(s.shape) for _ in range(ndim)], rho)
    if dust_on:
        s.U_dust = dust_state(s.closure, eps*rho, np.zeros((3, *s.shape)))
    return s


def mhd_solver(ndim=1, n=32, *, closure='gaussian', eps=0.01, ts=1., charge=1.,
               field=(1., 0., 0.), sgs_on=False, filtering=False, cfl=0.8,
               dust_config=None, sgs_config=None):
    from phrike.grid import Grid1D, Grid2D, Grid3D
    from phrike.dusty_mhd import (DustyMHDSolver, SGSMHDEquations1D,
                                SGSMHDEquations2D, SGSMHDEquations3D)
    from phrike.problems.problem_list.dusty_box import dust_state
    opts = dict(backend='numpy', precision='double', fft_workers=1,
                filter_params={'spectral_dissipation': dict(enabled=filtering, p=4,
                               rate=10., onset_fraction=0.)})
    if isinstance(filtering, dict):
        opts['filter_params']['spectral_dissipation'] = filtering
    grid = (Grid1D(n, 1., **opts) if ndim == 1 else
            Grid2D(n, n, 1., 1., **opts) if ndim == 2 else
            Grid3D(n, n, n, 1., 1., 1., **opts))
    eq = (SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D)[ndim-1](1.4)
    dc = dict(enabled=True, closure=closure, backreaction=True, charge_to_mass=charge,
              stopping={'model': 'constant', 't_s': ts})
    sc = dict(enabled=sgs_on, pressure=True, production='none', decay=False,
              d_model='mt25_magnetized')
    dc.update(dust_config or {})
    sc.update(sgs_config or {})
    s = DustyMHDSolver(grid, eq, cfl=cfl, dust_config=dc, sgs_config=sc)
    rho, zero = np.ones(s.shape), np.zeros(s.shape)
    s.U_gas = eq.conservative(rho, zero, zero, zero, rho,
                              *(float(b)*rho for b in field))
    s.U_dust = dust_state(s.closure, eps*rho, np.zeros((3, *s.shape)))
    return s


class Report:
    def __init__(self, name):
        self.name, self.start = name, time.perf_counter()
        self.out = ROOT/'outputs'/'dust_validation'/name
        self.out.mkdir(parents=True, exist_ok=True)
        self.checks, self.runs, self.figures, self.notes = [], [], [], []
        self.worker_count = 0
        self.initial_source_sha256 = source_hashes()

    def check(self, name, value, threshold, *, op='<=', reason=''):
        value, threshold = float(value), float(threshold)
        passed = bool(np.isfinite(value) and {'<=': value <= threshold,
                       '>=': value >= threshold, '==': value == threshold,
                       '<': value < threshold, '>': value > threshold}[op])
        self.checks.append(dict(name=name, value=value, threshold=threshold, op=op,
                                passed=passed, reason=reason))
        print(f"{'PASS' if passed else 'FAIL'} {name}: {value:.9g} {op} {threshold:.9g}", flush=True)
        return passed

    def figure(self, fig, name):
        fig.savefig(self.out/name, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(name)

    def finish(self, *, complete=True):
        final_hashes = source_hashes()
        if complete:
            self.check('source files changed during run',
                       sum(final_hashes[k] != v for k, v in self.initial_source_sha256.items()),
                       0, op='==', reason='Results require a stable source revision.')
        provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                          cwd=ROOT, text=True).strip(),
                          source_sha256=final_hashes, initial_source_sha256=self.initial_source_sha256,
                          numpy=np.__version__)
        def convert(x):
            if isinstance(x, np.ndarray): return x.tolist()
            if isinstance(x, np.generic): return x.item()
            raise TypeError(type(x).__name__)
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        units = 1024**2 if sys.platform == 'darwin' else 1024
        peak_mib = peak_rss/units
        child_peak_mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/units
        payload = dict(test=self.name, complete=complete, runtime_seconds=time.perf_counter()-self.start,
                       peak_memory_MiB=peak_mib,
                       child_peak_memory_MiB=child_peak_mib, worker_count=self.worker_count,
                       memory_upper_bound_MiB=peak_mib+self.worker_count*child_peak_mib,
                       checks=self.checks, runs=self.runs, figures=self.figures,
                       notes=self.notes, provenance=provenance)
        (self.out/'metrics.json').write_text(json.dumps(payload, indent=2, default=convert)+'\n')
        return int(any(not c['passed'] for c in self.checks))
