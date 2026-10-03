import argparse
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from perception.cli import draw
from perception.pipeline import Record, run
from perception.video import load_frames

_PROGRESS = gr.Progress()
TABLE_HEADERS = ["klatka", "id", "klasa", "x1", "y1", "x2", "y2", "near"]


@lru_cache(maxsize=2)
def _models(device: str | None, use_depth: bool):
    """Modele ładują się raz na proces, kolejne przebiegi je współdzielą."""
    from perception.depth import DepthEstimator
    from perception.detect import Detector

    return Detector(device=device), (DepthEstimator(device=device) if use_depth else None)


def _write_browser_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Zapisuje mp4 odtwarzalny w przeglądarce (H.264), a gdy kodek niedostępny, mp4v."""
    h, w = frames[0].shape[:2]
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            break
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def process(
    video: str | None,
    max_frames: int,
    stride: int,
    width: int,
    use_depth: bool,
    device: str,
    progress=_PROGRESS,
):
    if not video:
        raise gr.Error("Wgraj plik mp4.")

    progress(0, desc="Wczytuję klatki")
    frames, fps = load_frames(video, int(max_frames), int(stride), int(width))
    if not frames:
        raise gr.Error("Nie udało się wczytać klatek z pliku.")

    progress(0.05, desc="Ładuję modele (pierwszy raz trwa dłużej)")
    detector, depth = _models(None if device == "auto" else device, use_depth)

    n = len(frames)
    seen = 0

    def detect(img: np.ndarray):
        nonlocal seen
        seen += 1
        progress(0.1 + 0.8 * seen / n, desc=f"Klatka {seen}/{n}")
        return detector(img)

    res = run(frames, detect, depth)

    by_frame: dict[int, list[Record]] = {}
    for r in res.records:
        by_frame.setdefault(r.frame, []).append(r)

    progress(0.95, desc="Zapisuję wideo")
    out = Path(tempfile.mkdtemp()) / "wynik.mp4"
    _write_browser_video(
        out, [draw(f, by_frame.get(i, []), detector.id2label) for i, f in enumerate(frames)], fps
    )

    rows = [
        [
            r.frame,
            r.id,
            detector.id2label.get(r.label, str(r.label)),
            *(round(float(v), 1) for v in r.box),
            None if r.near is None else round(float(r.near), 3),
        ]
        for r in res.records
    ]
    ms = 1000 * float(np.median(res.seconds_per_frame))
    tracks = len({r.id for r in res.records})
    summary = f"Klatek: {n}, unikalnych torów: {tracks}, mediana {ms:.0f} ms/klatkę ({detector.device})"
    return str(out), summary, rows


def build():
    with gr.Blocks(title="perception") as demo:
        gr.Markdown("# perception: DETR → śledzenie → głębia\nWgraj film z dashcama (mp4).")
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="Wejście (mp4)", sources=["upload"])
                max_frames = gr.Slider(1, 600, value=60, step=1, label="Maks. klatek")
                stride = gr.Slider(1, 15, value=4, step=1, label="Co N-ta klatka")
                width = gr.Slider(320, 1920, value=960, step=32, label="Szerokość po przeskalowaniu")
                use_depth = gr.Checkbox(value=True, label="Estymacja głębi (near)")
                device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label="Urządzenie")
                btn = gr.Button("Uruchom", variant="primary")
            with gr.Column():
                out = gr.Video(label="Wynik")
                info = gr.Textbox(label="Podsumowanie", interactive=False)
        table = gr.Dataframe(headers=TABLE_HEADERS, label="Rekordy", interactive=False)
        btn.click(
            process,
            [inp, max_frames, stride, width, use_depth, device],
            [out, info, table],
        )
        if Path("tests/fixtures/clip.mp4").exists():
            gr.Examples([["tests/fixtures/clip.mp4"]], inputs=inp)
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="perception-ui", description=__doc__)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="publiczny link gradio.live")
    args = ap.parse_args(argv)
    build().queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
