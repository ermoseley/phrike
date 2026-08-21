from pathlib import Path

import numpy as np
import pytest

from scripts.reproduce_khi_fig2_metal import (
    SPECTRAL_DISSIPATION_ONSET_FRACTION,
    SpectralNavierStokesDye,
    cutoff_e_folding_time,
    numerical_signature,
    restrict_to_common_modes,
    validate_checkpoint_signature,
)


def test_khi_spectral_viscosity_preserves_low_band_and_damps_cutoff_one_efold():
    torch = pytest.importorskip("torch")
    nx = 24
    model = SpectralNavierStokesDye(nx, torch.device("cpu"))
    cutoff = (nx - 1) // 3
    low_mode = int(np.floor(SPECTRAL_DISSIPATION_ONSET_FRACTION * cutoff))
    x = torch.arange(nx, dtype=torch.float32) / nx
    low = torch.sin(2.0 * np.pi * low_mode * x)
    high = torch.cos(2.0 * np.pi * cutoff * x)
    field = 2.0 + (low + high)[None, :].expand(2 * nx, -1)
    state = field[None, ...]

    damped = model.dissipate(state, cutoff_e_folding_time(nx))

    np.testing.assert_allclose(float(damped.mean()), 2.0, atol=2.0e-6)
    low_amplitude = 2.0 * torch.mean((damped[0] - 2.0) * low[None, :])
    high_amplitude = 2.0 * torch.mean((damped[0] - 2.0) * high[None, :])
    np.testing.assert_allclose(float(low_amplitude), 1.0, rtol=2.0e-5)
    np.testing.assert_allclose(float(high_amplitude), np.exp(-1.0), rtol=2.0e-5)


def test_common_mode_restriction_removes_unshared_modes_and_resamples_nodes():
    source_nx = 24
    source_ny = 2 * source_nx
    target_nx = 12
    target_ny = 2 * target_nx
    source_x = np.arange(source_nx) / source_nx
    source_y = 2.0 * np.arange(source_ny) / source_ny
    y, x = np.meshgrid(source_y, source_x, indexing="ij")
    shared = np.sin(2.0 * np.pi * (3.0 * x + 2.0 * y / 2.0))
    unshared = 0.4 * np.cos(2.0 * np.pi * (6.0 * x + y / 2.0))

    target_x = np.arange(target_nx) / target_nx
    target_y = 2.0 * np.arange(target_ny) / target_ny
    target_y_mesh, target_x_mesh = np.meshgrid(
        target_y, target_x, indexing="ij"
    )
    expected = np.sin(
        2.0 * np.pi * (3.0 * target_x_mesh + 2.0 * target_y_mesh / 2.0)
    )
    np.testing.assert_allclose(
        restrict_to_common_modes(shared + unshared, target_nx),
        expected,
        atol=3.0e-14,
    )


def test_checkpoint_signature_rejects_old_or_mismatched_numerics():
    expected = numerical_signature(64, 0.7)
    validate_checkpoint_signature(
        {"numerical_signature": expected}, expected, Path("checkpoint.pt")
    )
    with pytest.raises(RuntimeError, match="obsolete numerical signature"):
        validate_checkpoint_signature(
            {"numerics_version": "hard-two-thirds-v1"},
            expected,
            Path("checkpoint.pt"),
        )
    with pytest.raises(RuntimeError, match="obsolete numerical signature"):
        validate_checkpoint_signature(
            {"numerical_signature": numerical_signature(64, 0.35)},
            expected,
            Path("checkpoint.pt"),
        )
