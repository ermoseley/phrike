import numpy as np
import pytest

from phrike.basis import ChebyshevLobattoBasis1D, LegendreLobattoBasis1D


@pytest.mark.parametrize("N", [1, 2, 9, 10])
def test_legendre_highest_mode_roundtrip_and_noop_filter(N):
    basis = LegendreLobattoBasis1D(N, 2.5)
    modes = np.eye(N)
    values = basis.inverse(modes)
    np.testing.assert_allclose(basis.forward(values), modes, atol=3.0e-13)
    np.testing.assert_allclose(
        basis.apply_spectral_filter(values, alpha=0.0), values, atol=3.0e-13
    )


@pytest.mark.parametrize("N", [2, 9, 10, 33])
def test_chebyshev_nodes_quadrature_and_full_modal_roundtrip(N):
    basis = ChebyshevLobattoBasis1D(N, 2.5)
    x = basis.nodes()
    weights = basis.quadrature_weights()
    assert x[0] == 0.0
    assert x[-1] == 2.5
    assert np.all(np.diff(x) > 0.0)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights.sum(), 2.5, atol=2.0e-14)
    for degree in range(N):
        np.testing.assert_allclose(
            weights @ (x / 2.5)**degree, 2.5 / (degree + 1), atol=2.0e-14
        )

    # Every individual mode, especially the endpoint-normalized highest mode.
    modes = np.eye(N)
    np.testing.assert_allclose(basis.forward(basis.inverse(modes)), modes, atol=1.0e-14)
    values = np.random.default_rng(12).normal(size=(2, 3, N))
    np.testing.assert_allclose(basis.inverse(basis.forward(values)), values, atol=2.0e-14)


def test_chebyshev_polynomial_derivatives_and_modal_filter():
    basis = ChebyshevLobattoBasis1D(17, 2.5)
    x = basis.nodes()
    values = np.stack([x**4 - 2*x, 3*x**3 + 2])
    np.testing.assert_allclose(basis.dx(values), [4*x**3 - 2, 9*x**2], atol=2.0e-11)
    np.testing.assert_allclose(basis.d2x(values), [12*x**2, 18*x], atol=2.0e-9)
    modes = np.zeros((2, 17))
    modes[:, 0] = 2.0
    modes[:, -1] = [1.0, -0.5]
    filtered = basis.apply_spectral_filter(basis.inverse(modes), p=8, alpha=3.0)
    modes[:, -1] *= np.exp(-3.0)
    np.testing.assert_allclose(filtered, basis.inverse(modes), atol=1.0e-14)


@pytest.mark.parametrize("N,Lx", [(1, 1), (3.5, 1), (3, 0), (3, -1), (3, np.inf), (3, np.nan)])
def test_chebyshev_invalid_geometry(N, Lx):
    with pytest.raises(ValueError):
        ChebyshevLobattoBasis1D(N, Lx)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_chebyshev_torch_roundtrip_derivatives_and_filter(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal/MPS is unavailable")
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2.0e-4 if device == "mps" else 2.0e-11
    basis = ChebyshevLobattoBasis1D(9, 2.5)
    values = np.stack([basis.nodes()**3, 1.0 + basis.nodes()**2])
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    for operation in (basis.dx, basis.d2x, basis.apply_spectral_filter):
        actual = operation(tensor)
        assert actual.dtype == tensor.dtype
        assert actual.device == tensor.device
        np.testing.assert_allclose(actual.cpu().numpy(), operation(values), atol=tolerance)
    modes = torch.eye(9, dtype=dtype, device=device).reshape(1, 9, 9)
    np.testing.assert_allclose(
        basis.forward(basis.inverse(modes)).cpu().numpy(),
        modes.cpu().numpy(), atol=tolerance,
    )
