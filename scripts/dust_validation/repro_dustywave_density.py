"""Reproduce poor cold-dust density convergence without changing solver code.

PYTHONPATH=. python scripts/dust_validation/repro_dustywave_density.py
A nonzero exit reproduces density order below the declared second-order envelope.
"""
from common import solver, np
from scipy.linalg import expm
from phrike.problems.problem_list.dusty_box import dust_state

c, k, A, eps = np.sqrt(1.4), 2*np.pi, 1e-4, 0.01
T = 1/c
nu = 1/(0.3*T)
M = np.array([[0,-1j*k,0,0],[-1j*k*c*c,-eps*nu,0,eps*nu],
              [0,0,0,-1j*k*eps],[0,nu,0,-nu]],dtype=complex)
y = expm(M*T)@(A*np.array([1,c,eps,c]))
errors = []
for n in (64,128):
    s = solver(n=n,eps=eps,ts=1/nu,cfl=.8)
    z = np.cos(k*s.grid.x)
    s.U_gas = s.equations.conservative(1+A*z,A*c*z,1+c*c*A*z)
    u = np.zeros((3,n)); u[0] = A*c*z
    s.U_dust = dust_state(s.closure,eps*(1+A*z),u)
    while s.t < T-1e-14:
        s.step(min(s.gas_solver._stability_limit(s.U_gas),T-s.t))
    ref = eps+np.real(y[2]*np.exp(1j*k*s.grid.x))
    error = np.sqrt(np.mean((s.U_dust[0]-ref)**2))/(A*eps)
    errors.append(error)
    print(f'N={n}, relative_L2={error:.12g}, stats={s.transport_stats}',flush=True)
order = np.log2(errors[0]/errors[1])
print(f'{"FAIL" if order<1.5 else "PASS"} dust density order: {order:.9g} >= 1.5')
raise SystemExit(int(order<1.5))
