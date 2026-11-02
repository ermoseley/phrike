"""Reproduce a small supported-velocity bound violation, without changing PHRIKE.

PYTHONPATH=. python scripts/dust_validation/repro_hyqmom_kinetic_bound.py
The exact f has only v=+/-1, so variance<=1; measured overshoot is ~2e-10.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/phrike-dust-mpl')
import numpy as np
from phrike.dust_moments import HyqmomClosure, transport


def main():
    n = 256
    z, w = np.polynomial.legendre.leggauss(6)
    x = (np.arange(n)[:, None]+0.5+z/2)/n
    beams = []
    for centre in (0.28, 0.72):
        profile = 1e-6+sum(np.exp(-0.5*((x-centre+j)/0.055)**2)
                           for j in (-2, -1, 0, 1, 2))
        beams.append(profile @ (w/2))
    rho = beams[0]+beams[1]
    m = (beams[0]-beams[1])/rho
    variance = 1-m*m
    q = -2*m+2*m**3
    r = 1+2*m*m-3*m**4
    closure = HyqmomClosure()
    u = np.zeros((3, n))
    u[0] = m
    U = closure.from_primitives(rho, u, variance, q, r, np.zeros(n))
    repaired = projections = 0
    for _ in range(4):
        U, stats = transport(U, closure, .22/4, (1/n,))
        repaired += stats['repaired_cells']
        projections += stats['roundoff_projections']
        if np.max(closure.primitives(U)['v']) > 1+1e-10:
            break
    maximum = np.max(closure.primitives(U)['v'])
    print(f'max variance={maximum:.17g}; kinetic bound=1; tolerance=1e-10; '
          f'repaired_cells={repaired}; roundoff_projections={projections}')
    assert maximum <= 1+1e-10, 'HyQMOM exceeds the supported-velocity variance bound'


if __name__ == '__main__':
    main()
