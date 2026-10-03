"""Spina etapy: detekcja -> śledzenie -> (opcjonalnie) głębia. Modele są wstrzykiwane jako funkcje."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from perception.depth import near_score
from perception.tracker import ByteTrackLite

DetectFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
DepthFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Record:
    frame: int
    id: int
    label: int
    box: np.ndarray
    near: float | None


@dataclass
class RunResult:
    records: list[Record] = field(default_factory=list)
    detections_per_frame: list[int] = field(default_factory=list)
    seconds_per_frame: list[float] = field(default_factory=list)
    depth_maps: list[np.ndarray] = field(default_factory=list)  # uint8 0..255, puste bez głębi


def run(
    frames: Sequence[np.ndarray],
    detect: DetectFn,
    depth: DepthFn | None = None,
    tracker: ByteTrackLite | None = None,
) -> RunResult:
    tracker = tracker or ByteTrackLite()
    result = RunResult()
    for fi, img in enumerate(frames):
        t0 = time.perf_counter()
        boxes, scores, labels = detect(img)
        active = tracker.update(boxes, scores, labels)
        dm = depth(img) if depth is not None else None
        if dm is not None:
            result.depth_maps.append((dm * 255).astype(np.uint8))
        for t in active:
            near = near_score(dm, t.box) if dm is not None else None
            result.records.append(Record(fi, t.id, t.label, t.box.copy(), near))
        result.detections_per_frame.append(len(boxes))
        result.seconds_per_frame.append(time.perf_counter() - t0)
    return result
