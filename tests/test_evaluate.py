"""Logika bramki jakości (bez modeli): sprawdzamy, że próg faktycznie blokuje."""

import numpy as np

from perception.gate import check, compute_metrics
from perception.pipeline import Record, RunResult

THRESHOLDS = {
    "min_mean_detections": 2.0,
    "min_mean_track_length": 5.0,
    "max_unique_tracks": 10,
    "max_ms_per_frame": 500,
}
GOOD = {"mean_detections": 3.0, "mean_track_length": 12.0, "unique_tracks": 6, "ms_per_frame": 200.0}


def test_good_metrics_pass():
    assert check(GOOD, THRESHOLDS) == []


def test_each_threshold_blocks():
    bad = {
        "mean_detections": 0.5,
        "mean_track_length": 1.0,
        "unique_tracks": 50,
        "ms_per_frame": 5000.0,
    }
    for metric, value in bad.items():
        failures = check({**GOOD, metric: value}, THRESHOLDS)
        assert len(failures) == 1 and failures[0].startswith(metric)


def test_missing_threshold_is_ignored():
    assert check({**GOOD, "ms_per_frame": 1e9}, {"min_mean_detections": 1.0}) == []


def test_compute_metrics():
    box = np.zeros(4)
    res = RunResult(
        records=[Record(0, 1, 3, box, None), Record(1, 1, 3, box, None), Record(1, 2, 3, box, None)],
        detections_per_frame=[2, 4],
        seconds_per_frame=[0.1, 0.3],
    )
    m = compute_metrics(res)
    assert m["frames"] == 2
    assert m["mean_detections"] == 3.0
    assert m["unique_tracks"] == 2
    assert m["mean_track_length"] == 1.5
    assert m["ms_per_frame"] == 200.0
