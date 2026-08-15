import numpy as np

from phrike.problems.problem_list.turb3d import turbulent_velocity_3d


def test_turbulent_velocity_supports_non_cubic_grids():
    shape = (2, 3, 4)
    coordinates = np.zeros(shape)

    state = turbulent_velocity_3d(
        coordinates,
        coordinates,
        coordinates,
        kmin=1.0,
        kmax=2.0,
    )

    assert state.shape == (5, *shape)
