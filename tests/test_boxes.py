import numpy as np
import pytest

from perception.boxes import iou_matrix, nms


def test_iou_identical_is_one():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10]], dtype=float)
    # przecięcie 50, suma 150
    assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_iou_empty_inputs_have_correct_shape():
    a = np.zeros((0, 4))
    b = np.array([[0, 0, 1, 1]], dtype=float)
    assert iou_matrix(a, b).shape == (0, 1)
    assert iou_matrix(b, a).shape == (1, 0)


def test_nms_keeps_highest_score_of_overlapping_pair():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.6, 0.9, 0.8])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2]


def test_nms_empty():
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5)) == 0
