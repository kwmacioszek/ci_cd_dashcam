import argparse
import json
import os
import sys
from pathlib import Path

from perception.gate import check, compute_metrics, markdown_report
from perception.pipeline import run
from perception.video import load_frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="tests/fixtures/clip.mp4")
    ap.add_argument("--thresholds", default="scripts/thresholds.json")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--device", default=None, help="cpu/mps/cuda; domyślnie auto")
    args = ap.parse_args()

    from perception.depth import DepthEstimator
    from perception.detect import Detector

    frames, _ = load_frames(args.video, args.max_frames, args.stride, args.width)
    detector = Detector(device=args.device)
    depth = None if args.no_depth else DepthEstimator(device=args.device)
    run(frames[:2], detector, depth)
    metrics = compute_metrics(run(frames, detector, depth))
    metrics["device"] = detector.device

    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    thresholds = json.loads(Path(args.thresholds).read_text())
    failures = check(metrics, thresholds)
    report = markdown_report(metrics, thresholds, failures)
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
