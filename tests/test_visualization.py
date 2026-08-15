import numpy as np
import pytest

from phrike.visualization import (
    central_column_mean,
    central_slice,
    projection_axis_labels,
    projection_extent,
)


FIELD = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ("z", FIELD.mean(axis=0)),
        ("y", FIELD.mean(axis=1)),
        ("x", FIELD.mean(axis=2)),
    ],
)
def test_central_column_mean_full_domain(axis, expected):
    np.testing.assert_allclose(central_column_mean(FIELD, axis=axis), expected)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [("z", FIELD[1]), ("y", FIELD[:, 1, :]), ("x", FIELD[:, :, 2])],
)
def test_central_slice_midpoint(axis, expected):
    np.testing.assert_array_equal(central_slice(FIELD, axis=axis), expected)


@pytest.mark.parametrize(
    ("axis", "extent", "labels"),
    [
        ("z", (0.0, 2.0, 0.0, 3.0), ("x", "y")),
        ("y", (0.0, 2.0, 0.0, 4.0), ("x", "z")),
        ("x", (0.0, 3.0, 0.0, 4.0), ("y", "z")),
    ],
)
def test_projection_coordinates(axis, extent, labels):
    assert projection_extent((2.0, 3.0, 4.0), axis=axis) == extent
    assert projection_axis_labels(axis=axis) == labels


def test_projection_helpers_reject_invalid_axis():
    with pytest.raises(ValueError, match="axis must be x, y, or z"):
        central_slice(FIELD, axis="r")
