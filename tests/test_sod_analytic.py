import numpy as np

from hydra.sod_analytic import sod_sample


def test_sod_sample_shapes():
    x = np.linspace(0, 1, 100)
    left = {"rho": 1.0, "u": 0.0, "p": 1.0}
    right = {"rho": 0.125, "u": 0.0, "p": 0.1}
    sol = sod_sample(x, t=0.1, x0=0.5, left=left, right=right, gamma=1.4)
    assert set(sol.keys()) == {"rho", "u", "p"}
    assert sol["rho"].shape == x.shape


