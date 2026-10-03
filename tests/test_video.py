"""Testy wczytywania wideo: syntetyczny klip generowany w locie, bez zależności od sieci i wag."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.video import load_frames

FIXTURE = Path(__file__).parent / "fixtures" / "clip.mp4"
FPS = 10.0
N_FRAMES = 20
SIZE = (640, 360)  # (szerokość, wysokość)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """Klip MJPG: klatka i ma jasność 10*i, żeby dało się sprawdzić, które klatki zostały wzięte."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, SIZE)
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 10 * i, np.uint8))
    writer.release()
    return path


def test_stride_decimates_frames_and_fps(clip: Path):
    frames, fps = load_frames(clip, stride=4, width=320)
    assert len(frames) == N_FRAMES // 4
    assert fps == pytest.approx(FPS / 4)


def test_stride_picks_every_nth_frame(clip: Path):
    frames, _ = load_frames(clip, stride=5, width=320)
    means = [float(f.mean()) for f in frames]
    # kodek stratny, więc tolerancja; kolejne wybrane klatki to 0, 50, 100, 150
    assert means == pytest.approx([0, 50, 100, 150], abs=4)


def test_max_frames_limits_output(clip: Path):
    frames, _ = load_frames(clip, max_frames=3, stride=1, width=320)
    assert len(frames) == 3


def test_resize_keeps_aspect_ratio(clip: Path):
    frames, _ = load_frames(clip, max_frames=1, stride=1, width=320)
    assert frames[0].shape == (180, 320, 3)
    assert frames[0].dtype == np.uint8


def test_frames_are_rgb_not_bgr(tmp_path: Path):
    path = tmp_path / "red.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (64, 64))
    for _ in range(3):
        writer.write(
            np.dstack([np.zeros((64, 64)), np.zeros((64, 64)), np.full((64, 64), 255)]).astype(np.uint8)
        )
    writer.release()  # w BGR kanał 2 to czerwony
    frames, _ = load_frames(path, max_frames=1, stride=1, width=64)
    r, g, b = frames[0].reshape(-1, 3).mean(axis=0)
    assert r > 200 and g < 60 and b < 60


def test_fixture_clip_loads(  # prawdziwy plik z repo, tak jak go dostanie potok
):
    frames, fps = load_frames(FIXTURE, max_frames=5, stride=1, width=320)
    assert len(frames) == 5
    assert fps > 0


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_frames("/nie/ma/takiego.mp4")


def test_accepts_str_and_path(clip: Path):
    a, _ = load_frames(clip, max_frames=2, stride=1, width=160)
    b, _ = load_frames(str(clip), max_frames=2, stride=1, width=160)
    assert len(a) == len(b) == 2


def test_corrupt_file_returns_no_frames(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"to nie jest wideo")
    frames, fps = load_frames(bad)
    assert frames == []
    assert fps == pytest.approx(30.0 / 4)
