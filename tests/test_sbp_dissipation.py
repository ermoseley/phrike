import numpy as np
import pytest

from phrike.grid import Grid2D


def _grid(ny=33, strength=0.2, **kwargs):
    return Grid2D(Nx=8, Ny=ny, Lx=1.0, Ly=1.0, basis_y="sbp",
                  filter_params={"sbp_dissipation": {"enabled": True,
                                                     "strength": strength}}, **kwargs)


def test_configuration_and_disabled_operator():
    grid = Grid2D(Nx=8, Ny=9, Lx=1.0, Ly=1.0, backend="numpy")
    assert grid.sbp_dissipation_strength == grid.sbp_dissipation_rate_bound(2) == 0
    np.testing.assert_array_equal(grid.sbp_dissipation(np.ones((9, 8)), 2), 0)
    with pytest.raises(ValueError, match="requires Fourier x and SBP y"):
        Grid2D(Nx=8, Ny=9, Lx=1.0, Ly=1.0, backend="numpy",
               filter_params={"sbp_dissipation": {"enabled": True, "strength": 0.2}})
    for bad in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="strength"):
            _grid(strength=bad, backend="numpy")
        with pytest.raises(ValueError, match="speed"):
            _grid(backend="numpy").sbp_dissipation_rate_bound(bad)


def test_weighted_conservation_dissipation_and_locality():
    grid = _grid(backend="numpy")
    rng = np.random.default_rng(4)
    f, g = rng.normal(size=(2, 3, grid.Ny, grid.Nx))
    Qf, Qg = grid.sbp_dissipation(f, 1.5), grid.sbp_dissipation(g, 1.5)
    w = grid.wy[:, None]
    np.testing.assert_allclose((Qf*w).sum(axis=-2), 0, atol=2e-14)
    np.testing.assert_allclose((f*Qg*w).sum(), (g*Qf*w).sum(), atol=2e-14)
    assert (f*Qf*w).sum() < 0
    y = np.asarray(grid.y)[:, None]
    for polynomial in (np.ones_like(y), y, y*y):
        np.testing.assert_array_equal(grid.sbp_dissipation(polynomial, 1.5), 0)
    impulse = np.zeros((grid.Ny, 1)); impulse[0] = 1
    np.testing.assert_array_equal(grid.sbp_dissipation(impulse, 1.5)[4:], 0)
    checker = ((-1.0)**np.arange(grid.Ny))[:, None]
    Qchecker = grid.sbp_dissipation(checker, 1.5)
    np.testing.assert_allclose(Qchecker[4:-4],
                               -grid.sbp_dissipation_strength*1.5/grid.dy*checker[4:-4])
    damped = checker+0.5/grid.sbp_dissipation_rate_bound(1.5)*Qchecker
    assert (damped*damped*w).sum() < (checker*checker*w).sum()
    assert -(f*Qf*w).sum() <= grid.sbp_dissipation_rate_bound(1.5)*(f*f*w).sum()


def test_smooth_refinement_has_fifth_interior_and_second_boundary_order():
    interior, boundary = [], []
    for ny in (17, 33, 65):
        grid = _grid(ny, backend="numpy")
        y = np.asarray(grid.y)[:, None]
        Qf = grid.sbp_dissipation(y**3+y**6, 1.0)
        interior.append(np.max(abs(Qf[4:-4])))
        boundary.append(abs(Qf[0, 0]))
    np.testing.assert_allclose(np.array(interior[:-1])/interior[1:], 32, rtol=1e-6)
    np.testing.assert_allclose(np.array(boundary[:-1])/boundary[1:], 4, rtol=0.03)


def test_torch_cpu_matches_numpy_and_preserves_input():
    torch = pytest.importorskip("torch")
    grid = _grid(backend="numpy")
    torch_grid = _grid(backend="torch", torch_device="cpu")
    f = np.random.default_rng(7).normal(size=(3, grid.Ny, grid.Nx))
    tensor = torch.as_tensor(f.copy())
    actual = torch_grid.sbp_dissipation(tensor, 1.2)
    np.testing.assert_allclose(actual.numpy(), grid.sbp_dissipation(f, 1.2), atol=2e-15)
    np.testing.assert_array_equal(tensor.numpy(), f)
    assert actual.dtype == tensor.dtype and actual.device == tensor.device
