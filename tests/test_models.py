"""Testy dymne na prawdziwych wagach (pobierane z Hugging Face). Uruchamiane osobno: pytest -m model."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.detect import ROAD

pytestmark = pytest.mark.model

FRAME = Path(__file__).parent / "fixtures" / "frame.jpg"


@pytest.fixture(scope="module")
def frame():
    bgr = cv2.imread(str(FRAME))
    assert bgr is not None, f"nie wczytano {FRAME}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def detector():
    from perception.detect import Detector

    return Detector(device="cpu")


@pytest.fixture(scope="module")
def depth_model():
    from perception.depth import DepthEstimator

    return DepthEstimator(device="cpu")


def test_detector_output_contract(detector, frame):
    boxes, scores, labels = detector(frame, thr=0.5)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert len(boxes) == len(scores) == len(labels)
    assert len(boxes) >= 1, "na klatce z drogą powinno być przynajmniej jedno auto/osoba"
    assert ((scores >= 0.5) & (scores <= 1)).all()
    assert {detector.id2label[int(lab)] for lab in labels} <= ROAD
    assert (boxes[:, 0] < boxes[:, 2]).all() and (boxes[:, 1] < boxes[:, 3]).all()
    h, w = frame.shape[:2]
    assert boxes[:, [0, 2]].min() >= 0 and boxes[:, [0, 2]].max() <= w
    assert boxes[:, [1, 3]].min() >= 0 and boxes[:, [1, 3]].max() <= h


def test_depth_output_contract(depth_model, frame):
    d = depth_model(frame)
    assert d.shape == frame.shape[:2]
    assert d.min() == pytest.approx(0.0, abs=1e-6) and d.max() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(d).all()
