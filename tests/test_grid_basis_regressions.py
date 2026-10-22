"""Regressions for spectral grids and polynomial bases."""

import warnings

import numpy as np
import pytest

import phrike.grid as grid_module
from phrike.basis.fourier import FourierBasis1D
from phrike.basis.legendre import LegendreLobattoBasis1D, LegendreLobattoBasis2D
from phrike.equations import EulerEquations2D
from phrike.grid import Grid1D, Grid2D, Grid3D


def test_dealias_mask_keeps_integer_cutoff_mode():
    for n in (10, 20, 43, 61, 69):
        grid = Grid1D(N=n, Lx=1.0, backend="numpy")
        wavenumber = np.rint(np.fft.fftfreq(n) * n).astype(int)
        cutoff = (n - 1) // 3
        mask = np.asarray(grid.dealias_mask)
        assert np.all(mask[np.abs(wavenumber) <= cutoff] == 1.0)
        assert np.all(mask[np.abs(wavenumber) > cutoff] == 0.0)

    grid = Grid2D(Nx=10, Ny=8, Lx=1.0, Ly=1.0, backend="numpy")
    kx = np.rint(np.fft.fftfreq(10) * 10).astype(int)
    ky = np.rint(np.fft.fftfreq(8) * 8).astype(int)
    expect = (
        (np.abs(ky)[:, None] <= (8 - 1) // 3)
        & (np.abs(kx)[None, :] <= (10 - 1) // 3)
    ).astype(float)
    np.testing.assert_array_equal(np.asarray(grid.dealias_mask), expect)

    grid = Grid3D(Nx=10, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0, backend="numpy")
    kz = np.rint(np.fft.fftfreq(4) * 4).astype(int)
    ky = kz
    kx = np.rint(np.fft.fftfreq(10) * 10).astype(int)
    expect_3d = (
        (np.abs(kz)[:, None, None] <= (4 - 1) // 3)
        & (np.abs(ky)[None, :, None] <= (4 - 1) // 3)
        & (np.abs(kx)[None, None, :] <= (10 - 1) // 3)
    ).astype(float)
    np.testing.assert_array_equal(np.asarray(grid.dealias_mask), expect_3d)


def test_fourier_point_evaluation_rejects_batches_and_drops_unretained_modes():
    grid = Grid1D(N=64, Lx=2.0 * np.pi, backend="numpy")
    x = np.asarray(grid.x)
    with pytest.raises(ValueError, match=r"shape"):
        grid.evaluate_fourier_at_points_1d(np.vstack((np.sin(x), np.cos(x))), x)
    kept = np.sin(21.0 * x)
    dropped = np.sin(22.0 * x)
    # FINUFFT is used when present; its requested tolerance is 1e-6.
    np.testing.assert_allclose(
        grid.evaluate_fourier_at_points_1d(kept, x), kept, atol=1e-5
    )
    assert np.max(np.abs(grid.evaluate_fourier_at_points_1d(dropped, x))) < 1e-5


def test_mixed_fourier_legendre_is_rejected_and_sbp_still_builds():
    with pytest.raises(NotImplementedError, match="Fourier/Legendre"):
        Grid2D(
            Nx=8, Ny=8, Lx=1.0, Ly=1.0,
            basis_x="legendre", basis_y="fourier", backend="numpy",
        )
    with pytest.raises(NotImplementedError, match="Fourier/Legendre"):
        Grid2D(
            Nx=8, Ny=8, Lx=1.0, Ly=1.0,
            basis_x="fourier", basis_y="legendre", backend="numpy",
        )
    grid = Grid2D(
        Nx=8, Ny=9, Lx=1.0, Ly=2.0, basis_y="sbp", backend="numpy",
        bc_config={
            "left": "periodic", "right": "periodic",
            "bottom": "open", "top": "outflow",
        },
    )
    assert grid._bc_boundary_map["top"] == "open"
    _, mesh_y = grid.xy_mesh()
    np.testing.assert_allclose(grid.dy1(mesh_y**2), 2.0 * mesh_y, atol=1e-12)


def test_unknown_and_incomplete_legendre_boundaries_raise():
    def build(**kwargs):
        return Grid2D(
            Nx=6, Ny=6, Lx=1.0, Ly=1.0,
            basis_x="legendre", basis_y="legendre", backend="numpy", **kwargs,
        )

    with pytest.raises(ValueError, match="Unknown 2D boundary condition"):
        build(bc="outflow")
    with pytest.raises(ValueError, match="Unknown 2D boundary condition"):
        build(bc="bogus")
    with pytest.raises(ValueError, match="without values"):
        build(bc_config={
            "left": "dirichlet", "right": "neumann",
            "bottom": "neumann", "top": "neumann",
        })
    with pytest.raises(ValueError, match="missing density"):
        build(bc_config={
            side: {"momentum_x": 0.0, "momentum_y": 0.0, "pressure": 1.0}
            for side in ("left", "right", "bottom", "top")
        })
    with pytest.raises(ValueError, match="must specify"):
        build(bc_config={"left": "neumann", "right": "neumann"})
    build(bc="reflective")


def test_legendre_wall_copies_energy_and_does_not_mutate_input():
    grid = Grid2D(
        Nx=6, Ny=6, Lx=1.0, Ly=1.0,
        basis_x="legendre", basis_y="legendre", backend="numpy", bc="dirichlet",
    )
    x, _ = grid.xy_mesh()
    equations = EulerEquations2D(gamma=1.4)
    state = equations.conservative(
        np.full_like(x, 2.0), np.full_like(x, 0.3), np.full_like(x, 0.1),
        np.full_like(x, 1.5),
    )
    original = state.copy()
    bounded = grid.apply_boundary_conditions(state, None)
    np.testing.assert_array_equal(state, original)
    assert np.all(bounded[1, 1:-1, 0] == 0.0)
    assert np.all(bounded[2, 0, :] == 0.0)
    np.testing.assert_array_equal(bounded[3, 1:-1, 0], original[3, 1:-1, 1])


def test_dirichlet_pressure_is_converted_to_total_energy():
    spec = {
        "density": 2.0, "momentum_x": 2.0, "momentum_y": 0.0, "pressure": 3.0,
    }
    grid = Grid2D(
        Nx=6, Ny=6, Lx=1.0, Ly=1.0,
        basis_x="legendre", basis_y="legendre", backend="numpy",
        bc_config={side: spec for side in ("left", "right", "bottom", "top")},
    )
    x, _ = grid.xy_mesh()
    equations = EulerEquations2D(gamma=1.4)
    state = equations.conservative(
        np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x),
    )
    original = state.copy()
    with pytest.raises(ValueError, match="requires equations"):
        grid.apply_boundary_conditions(state, None)
    bounded = grid.apply_boundary_conditions(state, equations)
    again = grid.apply_boundary_conditions(state, None)
    np.testing.assert_array_equal(state, original)
    # E = 3 / 0.4 + (2^2) / (2 * 2) = 7.5 + 1.
    np.testing.assert_allclose(bounded[3, :, 0], 8.5, atol=1e-12)
    np.testing.assert_allclose(bounded[0, :, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(bounded[1, :, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(bounded[3, 2, 2], 1.0 / 0.4, atol=1e-12)
    np.testing.assert_allclose(again[3, :, 0], 8.5, atol=1e-12)


def test_numpy_fourier_single_warns_and_legendre_single_does_not():
    grid_module._numpy_fourier_single_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = Grid1D(N=16, Lx=2.0 * np.pi, backend="numpy", precision="single")
        derivative = grid.dx1(np.sin(np.asarray(grid.x)))
        Grid1D(N=8, Lx=1.0, backend="numpy", precision="single")
    notes = [str(item.message) for item in caught if issubclass(item.category, UserWarning)]
    assert sum("float64" in note for note in notes) == 1
    assert grid.x.dtype == np.float64
    assert derivative.dtype == np.float64

    grid_module._numpy_fourier_single_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legendre = Grid1D(
            N=8, Lx=1.0, basis="legendre", bc="wall",
            backend="numpy", precision="single",
        )
    assert not any("Fourier" in str(item.message) for item in caught)
    assert legendre.x.dtype == np.float32


def test_mps_double_request_warns_once():
    pytest.importorskip("torch")
    grid_module._mps_double_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, precision = grid_module._torch_dtypes("double", "mps")
        grid_module._torch_dtypes("double", "mps")
    assert precision == "single"
    assert len(caught) == 1
    assert "float32" in str(caught[0].message)


def test_grid3d_mesh_dtype_follows_coordinates():
    torch = pytest.importorskip("torch")
    grid = Grid3D(
        Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0,
        backend="torch", torch_device="cpu", precision="single",
    )
    mesh_x, mesh_y, mesh_z = grid.xyz_mesh()
    assert mesh_x.dtype == mesh_y.dtype == mesh_z.dtype == grid.x.dtype
    assert mesh_x.dtype == torch.float32
    assert mesh_x.shape == (4, 4, 4)


def test_fourier_filter_follows_wavenumber():
    basis = FourierBasis1D(32, 2.0 * np.pi)
    wave = np.cos(basis.nodes())
    np.testing.assert_allclose(basis.apply_spectral_filter(wave), wave, atol=1e-6)
    coefficient = np.zeros(32)
    coefficient[16] = 1.0
    nyquist = basis.inverse(coefficient)
    filtered = basis.apply_spectral_filter(nyquist)
    np.testing.assert_allclose(filtered, nyquist * np.exp(-36.0), atol=1e-12)


def test_legendre_1d_accepts_rank3_and_float64_on_single():
    basis = LegendreLobattoBasis1D(9, 2.0)
    values = np.random.default_rng(0).normal(size=(2, 3, 9))
    np.testing.assert_allclose(basis.inverse(basis.forward(values)), values, atol=1e-10)
    np.testing.assert_allclose(
        basis.dx(values),
        np.stack([basis.dx(layer) for layer in values]),
        atol=1e-12,
    )
    np.testing.assert_allclose(basis.d2x(values)[1], basis.d2x(values[1]), atol=1e-10)
    np.testing.assert_allclose(
        basis.apply_spectral_filter(values, alpha=0.0), values, atol=1e-10
    )

    single = LegendreLobattoBasis1D(9, 2.0, precision="single")
    field = np.sin(np.pi * np.asarray(single.nodes(), dtype=np.float64) / 2.0)
    promoted = single.forward(field)
    native = single.forward(field.astype(np.float32))
    np.testing.assert_allclose(promoted, np.asarray(native, dtype=np.float64), rtol=1e-5, atol=1e-5)


def test_legendre_2d_forward_inverse_and_numpy_fallback(monkeypatch):
    import phrike.basis.legendre as legendre_module

    basis = LegendreLobattoBasis2D(9, 5, 1.0, 2.0)
    values = np.random.default_rng(1).normal(size=(4, 5, 9))
    np.testing.assert_allclose(basis.inverse(basis.forward(values)), values, atol=1e-10)
    _, nodes_y = basis.nodes()
    field_y = nodes_y[:, None] + np.zeros((5, 9))
    np.testing.assert_allclose(basis.dy(field_y**2), 2.0 * field_y, atol=1e-9)

    sample = np.random.default_rng(2).normal(size=(3, 5, 9))
    derivative = basis.dy(sample)
    monkeypatch.setattr(legendre_module, "_NUMBA_AVAILABLE", False)
    fallback = basis.dy(sample)
    filtered = basis.apply_spectral_filter(sample, alpha=0.0)
    np.testing.assert_allclose(fallback, derivative, atol=1e-12)
    assert fallback.shape == sample.shape
    np.testing.assert_allclose(filtered, sample, atol=1e-10)
    np.testing.assert_allclose(basis.dy(field_y**2), 2.0 * field_y, atol=1e-9)


def test_legendre_mps_filter_matches_numpy():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS unavailable")
    reference = LegendreLobattoBasis1D(33, 1.0, precision="single")
    nodes = np.asarray(reference.nodes(), dtype=np.float32)
    field = np.sin(2.0 * np.pi * nodes)
    expected = np.asarray(reference.apply_spectral_filter(field))
    for _ in range(4):
        basis = LegendreLobattoBasis1D(33, 1.0, precision="single")
        tensor = torch.tensor(field, dtype=torch.float32, device="mps")
        first = basis.apply_spectral_filter(tensor)
        second = basis.apply_spectral_filter(tensor)
        assert torch.isfinite(first).all()
        np.testing.assert_allclose(first.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(second.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)

    basis_2d = LegendreLobattoBasis2D(9, 7, 1.0, 1.0, precision="single")
    nodes_x, nodes_y = basis_2d.nodes()
    yy, xx = np.meshgrid(
        np.asarray(nodes_y, dtype=np.float32),
        np.asarray(nodes_x, dtype=np.float32),
        indexing="ij",
    )
    field_2d = np.sin(np.pi * xx) * np.cos(np.pi * yy)
    expected_2d = np.asarray(basis_2d.apply_spectral_filter(field_2d))
    tensor_2d = torch.tensor(field_2d, dtype=torch.float32, device="mps")
    filtered_2d = basis_2d.apply_spectral_filter(tensor_2d)
    assert torch.isfinite(filtered_2d).all()
    np.testing.assert_allclose(
        filtered_2d.cpu().numpy(), expected_2d, rtol=1e-4, atol=1e-4
    )
