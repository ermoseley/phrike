"""Minimal source-only reproducer: stiff SGS midpoint predictor freezes decay.

PYTHONPATH=. python scripts/dust_validation/repro_ou_budget_freeze.py
No transport is invoked: repaired_cells=roundoff_projections=0.
"""
import numpy as np
from scipy.integrate import solve_ivp
from phrike.dust_moments import GaussianClosure
from phrike.dust_sources import local_coupling_step

c = GaussianClosure(1)
rho = np.ones(1)
dust = c.from_primitives(.1*rho, np.zeros((3,1)), .04*np.eye(3)[:,:,None])
gas = np.array([[1.], [0.], [2.5]])
nu, delta, h = .04, .25, 250.
_, _, es, ledger = local_coupling_step(gas, dust, 1.5*rho, h, closure=c,
    gamma=1.4, nu=nu, Delta=delta, ndim=1)


def rhs(t,y):
    e = max(y[0],0.)
    sigma = np.sqrt(2*e/3)
    D = sigma**3/delta*np.arctan(nu*delta/sigma) if sigma else 0.
    return [-.3*D-e**1.5/delta]


ref = solve_ivp(rhs,(0,h),[1.5],method='Radau',rtol=1e-11,atol=1e-14)
print(f'nu*tau_Delta={nu*delta:g}, h*nu={h*nu:g}, '
      f'es numerical={es[0]:.12g}, Radau={ref.y[0,-1]:.12g}, '
      f'L_decay={ledger["L_decay"][0]:.12g}; '
      'repaired_cells=0, roundoff_projections=0')
# The physical reservoir drains; midpoint rates instead vanish after its predictor.
assert abs(es[0]-ref.y[0,-1])/1.5 < .01, 'SGS reservoir freezes at a stiff source step'
