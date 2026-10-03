
import statistics

from perception.pipeline import RunResult


def compute_metrics(result: RunResult) -> dict[str, float]:
    ids = {r.id for r in result.records}
    lengths = [sum(1 for r in result.records if r.id == i) for i in ids]
    return {
        "frames": len(result.detections_per_frame),
        "mean_detections": round(statistics.fmean(result.detections_per_frame), 3),
        "unique_tracks": len(ids),
        "mean_track_length": round(statistics.fmean(lengths), 2) if lengths else 0.0,
        "ms_per_frame": round(1000 * statistics.median(result.seconds_per_frame), 1),
    }


def check(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Zwraca listę naruszeń (pusta = bramka zaliczona)."""
    rules = [
        ("mean_detections", "min_mean_detections", lambda v, t: v >= t, ">="),
        ("mean_track_length", "min_mean_track_length", lambda v, t: v >= t, ">="),
        ("unique_tracks", "max_unique_tracks", lambda v, t: v <= t, "<="),
        ("ms_per_frame", "max_ms_per_frame", lambda v, t: v <= t, "<="),
    ]
    failures = []
    for metric, key, ok, op in rules:
        if key in thresholds and not ok(metrics[metric], thresholds[key]):
            failures.append(f"{metric} = {metrics[metric]} (wymagane {op} {thresholds[key]})")
    return failures


def markdown_report(metrics: dict[str, float], thresholds: dict[str, float], failures: list[str]) -> str:
    status = "✅ zaliczona" if not failures else "❌ niezaliczona"
    lines = [f"### Bramka jakości: {status}", "", "| metryka | wartość | próg |", "|---|---|---|"]
    keys = {
        "mean_detections": "min_mean_detections",
        "mean_track_length": "min_mean_track_length",
        "unique_tracks": "max_unique_tracks",
        "ms_per_frame": "max_ms_per_frame",
    }
    for metric, key in keys.items():
        lines.append(f"| {metric} | {metrics[metric]} | {thresholds.get(key, '-')} |")
    lines += [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"
