import numpy as np

from perception.tracker import ByteTrackLite


def det(*boxes, score=0.9, label=1):
    b = np.array(boxes, dtype=float).reshape(-1, 4)
    return b, np.full(len(b), score), np.full(len(b), label)


def moving_box(frame, x0=100.0, vx=4.0, size=60.0):
    x = x0 + vx * frame
    return [x, 200, x + size, 200 + size]


def test_track_needs_min_hits_before_confirmation():
    tr = ByteTrackLite(min_hits=3)
    assert tr.update(*det(moving_box(0))) == []
    assert tr.update(*det(moving_box(1))) == []
    assert len(tr.update(*det(moving_box(2)))) == 1


def test_id_is_stable_for_moving_object():
    tr = ByteTrackLite()
    ids = set()
    for f in range(30):
        for t in tr.update(*det(moving_box(f))):
            ids.add(t.id)
    assert ids == {1}


def test_two_separate_objects_get_different_ids():
    tr = ByteTrackLite()
    last = []
    for f in range(10):
        last = tr.update(*det(moving_box(f, x0=50), moving_box(f, x0=500, vx=-3)))
    assert sorted(t.id for t in last) == [1, 2]


def test_track_survives_short_gap_with_same_id():
    tr = ByteTrackLite(max_age=10)
    for f in range(8):
        tr.update(*det(moving_box(f)))
    for _ in range(3):  # 3 klatki bez detekcji (zasłonięcie)
        tr.update(*det())
    out = tr.update(*det(moving_box(11)))
    assert [t.id for t in out] == [1]


def test_track_is_dropped_after_max_age_and_new_object_gets_new_id():
    tr = ByteTrackLite(max_age=3)
    for f in range(6):
        tr.update(*det(moving_box(f)))
    for _ in range(5):
        tr.update(*det())
    assert tr.tracks == []
    out = []
    for f in range(20, 25):
        out = tr.update(*det(moving_box(f)))
    assert [t.id for t in out] == [2]


def test_ids_are_per_tracker_not_global():
    a, b = ByteTrackLite(), ByteTrackLite()
    a.update(*det(moving_box(0)))
    b.update(*det(moving_box(0)))
    assert a.tracks[0].id == b.tracks[0].id == 1


def test_different_class_does_not_steal_track():
    tr = ByteTrackLite()
    for f in range(5):
        tr.update(*det(moving_box(f), label=1))
    # w tym samym miejscu pojawia się obiekt innej klasy: nie może przejąć toru klasy 1
    out = tr.update(*det(moving_box(5), label=3))
    assert all(t.label != 3 for t in out)
    assert len(tr.tracks) == 2


def test_detection_goes_to_track_of_same_class_even_if_other_class_overlaps_more():
    tr = ByteTrackLite()
    car, person = [100, 200, 160, 260], [110, 200, 170, 260]
    tr.update(
        np.array([car, person], dtype=float), np.array([0.9, 0.9]), np.array([3, 1])
    )  # tor 1 = auto (klasa 3), tor 2 = osoba (klasa 1)
    # detekcja osoby leży dokładnie na aucie (IoU z autem = 1.0, z osobą ~0.7)
    tr.update(*det(car, label=1))
    by_id = {t.id: t for t in tr.tracks}
    assert len(tr.tracks) == 2, "detekcja nie może założyć nowego toru"
    assert by_id[2].hits == 2 and by_id[1].hits == 1


def test_low_score_detection_extends_existing_track():
    tr = ByteTrackLite(high=0.6, low=0.3)
    for f in range(6):
        tr.update(*det(moving_box(f), score=0.9))
    out = tr.update(*det(moving_box(6), score=0.4))  # słaba detekcja, ale tor już istnieje
    assert [t.id for t in out] == [1]


def test_low_score_detection_does_not_start_new_track():
    tr = ByteTrackLite(high=0.6, low=0.3)
    tr.update(*det(moving_box(0), score=0.4))
    assert tr.tracks == []


def test_kalman_predicts_forward_motion():
    tr = ByteTrackLite()
    for f in range(10):
        tr.update(*det(moving_box(f, vx=5)))
    track = tr.tracks[0]
    x_before = track.box[0]
    track.predict()
    assert track.box[0] > x_before
