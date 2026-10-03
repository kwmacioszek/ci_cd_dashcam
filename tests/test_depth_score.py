import numpy as np

from perception.depth import near_score


def test_uses_lower_half_of_box():
    depth = np.zeros((100, 100))
    depth[:50] = 0.1  # góra: tło
    depth[50:] = 0.9  # dół: zderzak/stopy
    assert near_score(depth, np.array([10, 20, 60, 80])) == 0.9


def test_box_is_clipped_to_image():
    depth = np.full((100, 100), 0.5)
    assert near_score(depth, np.array([-50, -50, 500, 500])) == 0.5


def test_box_outside_image_returns_none():
    depth = np.full((100, 100), 0.5)
    assert near_score(depth, np.array([200, 200, 300, 300])) is None


def test_degenerate_box_returns_none():
    depth = np.full((100, 100), 0.5)
    assert near_score(depth, np.array([10, 10, 10, 50])) is None
