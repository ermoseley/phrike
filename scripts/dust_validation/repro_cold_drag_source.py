"""Repeated exact source maps reject a cold Gaussian state near zero mean.

PYTHONPATH=. python scripts/dust_validation/repro_cold_drag_source.py
No spatial transport is invoked; repaired_cells=roundoff_projections=0.
"""
import numpy as np
from phrike.dust_moments import GaussianClosure
from phrike.dust_sources import local_coupling_step

c = GaussianClosure(1)
rho = np.ones(1)
u = np.array([[.7],[0.],[0.]])
dust = c.from_primitives(.01*rho,u,np.zeros((3,3,1)))
gas = np.array([[1.],[-.3],[2.545]])
es = np.zeros(1)
for j in range(500):
    try:
        gas,dust,es,_ = local_coupling_step(gas,dust,es,.01,closure=c,
            gamma=1.4,nu=1.,Delta=.25,ndim=1,d_model='none',sgs_decay=False)
    except ValueError:
        p = c.primitives(dust)
        print(f'step={j}, u_d={p["u"][0,0]:.12g}, C_xx={p["C"][0,0,0]:.12g}; '
              'repaired_cells=0, roundoff_projections=0')
        raise
print('No cold source rejection on this revision.')
