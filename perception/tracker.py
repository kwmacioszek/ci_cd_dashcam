import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.boxes import iou_matrix


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w, h])


def z_to_xyxy(z: np.ndarray) -> np.ndarray:
    return np.array([z[0] - z[2] / 2, z[1] - z[3] / 2, z[0] + z[2] / 2, z[1] + z[3] / 2])


class Track:
    F = np.eye(8) + np.eye(8, k=4)  # model stałej prędkości
    H = np.eye(4, 8)
    Q = np.diag([1, 1, 1, 1, 0.1, 0.1, 0.05, 0.05]).astype(float)
    R = np.diag([4, 4, 9, 9]).astype(float)

    def __init__(self, track_id: int, box: np.ndarray, score: float, label: int):
        self.x = np.zeros(8)
        self.x[:4] = xyxy_to_z(box)
        self.P = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(float)
        self.label, self.score = int(label), float(score)
        self.id = track_id
        self.hits, self.since = 1, 0
        self.history = [self.x[:2].copy()]

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.since += 1
        return self.box

    def update(self, box: np.ndarray, score: float) -> None:
        y = xyxy_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.score, self.hits, self.since = float(score), self.hits + 1, 0
        self.history.append(self.x[:2].copy())

    @property
    def box(self) -> np.ndarray:
        return z_to_xyxy(self.x[:4])


class ByteTrackLite:
    def __init__(
        self,
        high: float = 0.6,
        low: float = 0.3,
        iou1: float = 0.3,
        iou2: float = 0.5,
        max_age: int = 10,
        min_hits: int = 3,
    ):
        self.high, self.low, self.iou1, self.iou2 = high, low, iou1, iou2
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def _match(
        self, tracks: list[Track], boxes: np.ndarray, labels: np.ndarray, iou_thr: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = iou_matrix(np.array([t.box for t in tracks]), boxes)
        same = np.array([[t.label == lab for lab in labels] for t in tracks])
        cost = 1 - iou * same
        rows, cols = linear_sum_assignment(cost)
        pairs = [
            (int(i), int(j)) for i, j in zip(rows, cols, strict=True) if iou[i, j] * same[i, j] >= iou_thr
        ]
        matched_t, matched_d = {p[0] for p in pairs}, {p[1] for p in pairs}
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in matched_t],
            [j for j in range(len(boxes)) if j not in matched_d],
        )

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> list[Track]:
        """Jedna klatka. Zwraca potwierdzone tory zaktualizowane w tej klatce."""
        for t in self.tracks:
            t.predict()
        hi = scores >= self.high
        lo = (scores >= self.low) & (scores < self.high)
        bh, sh, lh = boxes[hi], scores[hi], labels[hi]
        bl, sl, ll = boxes[lo], scores[lo], labels[lo]

        pairs, unmatched_tracks, unmatched_high = self._match(self.tracks, bh, lh, self.iou1)
        for i, j in pairs:
            self.tracks[i].update(bh[j], sh[j])
        rest = [self.tracks[i] for i in unmatched_tracks]
        pairs2, _, _ = self._match(rest, bl, ll, self.iou2)
        for i, j in pairs2:
            rest[i].update(bl[j], sl[j])
        for j in unmatched_high:
            self.tracks.append(Track(self._next_id, bh[j], sh[j], lh[j]))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.since <= self.max_age]
        return [t for t in self.tracks if t.since == 0 and t.hits >= self.min_hits]
