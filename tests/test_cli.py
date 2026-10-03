"""Testy CLI i eksportu wyników bez pobierania wag modeli."""

import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from perception import cli
from perception.pipeline import Record


@pytest.fixture
def models(monkeypatch):
    # Tracker potwierdza tor po trzech detekcjach; czwarta klatka daje też ślad toru.
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    loader = Mock(return_value=(frames, 12.5))
    monkeypatch.setattr(cli, "load_frames", loader)
    detector = Mock(
        return_value=(np.array([[8.0, 8.0, 32.0, 32.0]]), np.array([0.9]), np.array([3])),
        id2label={3: "car"},
        device="cpu",
    )
    detector_factory = Mock(return_value=detector)
    depth_factory = Mock(return_value=lambda img: np.full(img.shape[:2], 0.75))
    monkeypatch.setattr("perception.detect.Detector", detector_factory)
    monkeypatch.setattr("perception.depth.DepthEstimator", depth_factory)
    return loader, detector_factory, depth_factory


@pytest.mark.parametrize("no_depth", [False, True])
def test_main_exports_json_and_videos(tmp_path, capsys, models, no_depth):
    loader, detector_factory, depth_factory = models
    video = tmp_path / "input.mp4"
    json_path = tmp_path / "records.json"
    outputs = [tmp_path / "boxes.mp4", tmp_path / "tracks.mp4"]
    args = [
        str(video),
        "--device",
        "cpu",
        "--max-frames",
        "4",
        "--stride",
        "2",
        "--width",
        "64",
        "--json",
        str(json_path),
        "--out",
        str(outputs[0]),
        "--out-tracks",
        str(outputs[1]),
    ]
    if no_depth:
        args.append("--no-depth")
    else:
        outputs.append(tmp_path / "depth.mp4")
        args.extend(["--out-depth", str(outputs[-1])])

    assert cli.main(args) == 0

    loader.assert_called_once_with(video, 4, 2, 64)
    detector_factory.assert_called_once_with(device="cpu")
    if no_depth:
        depth_factory.assert_not_called()
    else:
        depth_factory.assert_called_once_with(device="cpu")
    assert json.loads(json_path.read_text()) == [
        {
            "frame": index,
            "id": 1,
            "label": "car",
            "box": [8.0, 8.0, 32.0, 32.0],
            "near": None if no_depth else 0.75,
        }
        for index in range(2, 4)
    ]
    for path in outputs:
        capture = cv2.VideoCapture(str(path))
        try:
            assert capture.isOpened()
            assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 4
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.5)
            for _ in range(4):
                ok, frame = capture.read()
                assert ok and frame.shape == (64, 64, 3)
            assert frame.any()
        finally:
            capture.release()
    assert "Unikalne tory: 1" in capsys.readouterr().out


@pytest.mark.parametrize("missing", [False, True])
def test_main_reports_unreadable_input(monkeypatch, capsys, missing):
    loader = Mock(side_effect=FileNotFoundError) if missing else Mock(return_value=([], 30.0))
    monkeypatch.setattr(cli, "load_frames", loader)

    assert cli.main(["missing.mp4"]) == 2

    message = "Brak pliku" if missing else "Nie udało się wczytać klatek"
    assert message in capsys.readouterr().err


def test_main_rejects_depth_output_when_depth_disabled(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["input.mp4", "--no-depth", "--out-depth", "depth.mp4"])
    assert exc.value.code == 2
    assert "--out-depth wymaga głębi" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--out", "--out-tracks", "--out-depth"])
def test_main_reports_invalid_output_extension(tmp_path, capsys, models, option):
    path = tmp_path / "output.avi"

    assert cli.main(["input.mp4", option, str(path), "--device", "cpu"]) == 2

    assert not path.exists()
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err


def test_write_video_releases_writer_on_open_failure(tmp_path, monkeypatch):
    writer = Mock()
    writer.isOpened.return_value = False
    monkeypatch.setattr(cli.cv2, "VideoWriter", Mock(return_value=writer))

    with pytest.raises(RuntimeError, match="Nie można otworzyć pliku wyjściowego"):
        cli.write_video(tmp_path / "output.mp4", [np.zeros((64, 64, 3), dtype=np.uint8)], 10.0)

    writer.release.assert_called_once()
    writer.write.assert_not_called()


def test_write_video_adds_mp4_extension_and_preserves_rgb(tmp_path):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    path = cli.write_video(tmp_path / "red", [frame], 10.0)

    assert path == tmp_path / "red.mp4"
    capture = cv2.VideoCapture(str(path))
    try:
        ok, bgr = capture.read()
        assert ok
        blue, green, red = bgr.reshape(-1, 3).mean(axis=0)
        assert red > 200 and green < 20 and blue < 20
    finally:
        capture.release()


def test_draw_handles_unknown_label_without_modifying_input():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    record = Record(0, 7, 999, np.array([8, 8, 32, 32]), None)

    rendered = cli.draw(frame, [record], {})

    assert rendered.shape == frame.shape
    assert rendered.any()
    assert not frame.any()
