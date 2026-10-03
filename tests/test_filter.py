import numpy as np

from perception.detect import filter_detections

SHAPE = (480, 640)
ROAD_IDS = {1, 3}  # np. person, car


def run(boxes, scores, labels):
    return filter_detections(
        np.array(boxes, dtype=float).reshape(-1, 4),
        np.array(scores, dtype=float),
        np.array(labels),
        SHAPE,
        ROAD_IDS,
    )


def test_keeps_road_class():
    b, s, lab = run([[100, 100, 160, 160]], [0.9], [3])
    assert len(b) == 1 and lab[0] == 3


def test_drops_non_road_class():
    b, _, _ = run([[100, 100, 160, 160]], [0.9], [99])
    assert len(b) == 0


def test_drops_giant_box():
    b, _, _ = run([[0, 0, 600, 400]], [0.9], [3])
    assert len(b) == 0


def test_drops_own_car_mask_at_bottom_of_frame():
    b, _, _ = run([[200, 400, 300, 470]], [0.9], [3])
    assert len(b) == 0


def test_same_object_with_two_labels_keeps_higher_score():
    b, s, lab = run([[100, 100, 160, 160], [101, 101, 161, 161]], [0.7, 0.9], [1, 3])
    assert len(b) == 1 and lab[0] == 3 and s[0] == 0.9


def test_empty_input():
    b, s, lab = run([], [], [])
    assert len(b) == len(s) == len(lab) == 0
