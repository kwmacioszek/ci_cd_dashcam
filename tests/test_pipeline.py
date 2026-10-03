"""Test spinania etapów na atrapach modeli: szybki, bez wag i bez sieci."""

from pathlib import Path

import numpy as np
import pytest

from perception.pipeline import run
from perception.video import load_frames

FIXTURE = Path(__file__).parent / "fixtures" / "clip.mp4"


class FakeDetector:
    """Jeden samochód jadący w prawo, niezależnie od treści obrazu."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        x = 10.0 + 5 * self.calls
        self.calls += 1
        return np.array([[x, 100, x + 80, 160.0]]), np.array([0.9]), np.array([3])


def fake_depth(img):
    return np.full(img.shape[:2], 0.7)


def test_run_produces_one_stable_track_with_depth():
    frames = [np.zeros((240, 320, 3), np.uint8)] * 10
    res = run(frames, FakeDetector(), fake_depth)
    assert {r.id for r in res.records} == {1}
    assert all(r.near == pytest.approx(0.7) for r in res.records)
    assert res.detections_per_frame == [1] * 10
    assert len(res.seconds_per_frame) == 10


def test_run_without_depth_has_no_near_score():
    frames = [np.zeros((240, 320, 3), np.uint8)] * 5
    res = run(frames, FakeDetector(), depth=None)
    assert res.records and all(r.near is None for r in res.records)


def test_load_frames_from_fixture():
    frames, fps = load_frames(FIXTURE, max_frames=10, stride=2, width=320)
    assert len(frames) == 10
    assert frames[0].shape[1] == 320 and frames[0].shape[2] == 3
    assert fps == pytest.approx(7.5)


def test_load_frames_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_frames("/nie/ma/takiego.mp4")
