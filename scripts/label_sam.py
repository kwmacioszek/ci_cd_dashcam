"""Streamlit: poprawianie etykiet YOLO z pomocą SAM 3 (klik -> ramka, tekst -> kandydaci).

  make label            # albo: uv run --extra ml --extra train --extra label streamlit run scripts/label_sam.py

Narzędzia (klikasz w obraz):
  SAM: punkt      klik w obiekt -> SAM3 (tracker) zwraca maskę, z niej powstaje ramka
  Ramka: 2 klik.  lewy-górny i prawy-dolny róg; opcjonalnie dopracowana przez SAM3
  Usuń            klik w ramkę usuwa ją
Dodatkowo: prompt tekstowy SAM3 ("car", "traffic light") podpowiada kandydatów, które akceptujesz.
Autozapis: każda zmiana ramek trafia od razu do etykiet zbioru (<zbiór>/<podział>/labels/), a pierwotny
plik jest raz kopiowany do runs/labels_orig/<podział>/. Bez opcji "Zapis w zbiorze" zapis idzie do osobnego
katalogu (domyślnie runs/labels_fixed), a oryginalny zbiór zostaje nietknięty.
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envfile import load_env  # noqa: E402
from eval_base import AUG_MARK, read_gt  # noqa: E402

SHOW_W = 1100
BACKUP = Path("runs/labels_orig")
TOOLS = ["SAM: punkt", "Ramka: 2 klik.", "Usuń"]
load_env()


@st.cache_resource
def _device() -> str:
    from perception.detect import pick_device

    return pick_device()


@st.cache_resource
def _tracker():
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    return (
        Sam3TrackerProcessor.from_pretrained("facebook/sam3"),
        Sam3TrackerModel.from_pretrained("facebook/sam3").to(_device()).eval(),
    )


@st.cache_resource
def _concept():
    from transformers import Sam3Model, Sam3Processor

    return (
        Sam3Processor.from_pretrained("facebook/sam3"),
        Sam3Model.from_pretrained("facebook/sam3").to(_device()).eval(),
    )


def sam_box(img: Image.Image, point=None, box=None) -> list[float] | None:
    """Ramka xyxy z maski SAM3 dla punktu (x, y) albo ramki (x1, y1, x2, y2)."""
    import torch

    proc, model = _tracker()
    kw = {"input_points": [[[list(point)]]], "input_labels": [[[1]]]} if point else {"input_boxes": [[list(box)]]}
    inp = proc(images=img, return_tensors="pt", **kw).to(_device())
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    mask = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
    m = np.asarray(mask).reshape(-1, *np.asarray(mask).shape[-2:])[0] > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def sam_text(img: Image.Image, prompt: str, thr: float, min_side: int = 8) -> list[list[float]]:
    import torch

    proc, model = _concept()
    inp = proc(images=img, text=prompt, return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_instance_segmentation(
        out, threshold=thr, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist()
    )[0]
    boxes = r["boxes"].cpu().numpy().reshape(-1, 4)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_side) & ((boxes[:, 3] - boxes[:, 1]) >= min_side)
    return [b.tolist() for b in boxes[keep]]


def label_path(stem: str, data: Path, out: Path) -> Path:
    """Gdzie zapisujemy etykiety: w zbiorze (autozapis) albo w osobnym katalogu."""
    if st.session_state.in_dataset:
        return data / st.session_state.split / "labels" / (stem + ".txt")
    return out / (stem + ".txt")


def is_fixed(stem: str, data: Path, out: Path) -> bool:
    if st.session_state.in_dataset:
        return (BACKUP / st.session_state.split / (stem + ".txt")).is_file()
    return (out / (stem + ".txt")).is_file()


def load_boxes(img_path: Path, size: tuple[int, int], data: Path, out: Path) -> list[dict]:
    fixed = out / (img_path.stem + ".txt")
    label = data / st.session_state.split / "labels" / (img_path.stem + ".txt")
    if not st.session_state.in_dataset and fixed.is_file():
        label = fixed
    b, c = read_gt(label, *size)
    return [{"cls": int(k), "box": [float(v) for v in x]} for x, k in zip(b, c, strict=True)]


def save_boxes(path: Path, boxes: list[dict], size: tuple[int, int]) -> None:
    w, h = size
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = (min(max(v, 0), lim) for v, lim in zip(b["box"], (w, h, w, h), strict=True))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{b['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def signature(boxes: list[dict]) -> list:
    return [(b["cls"], *(round(v, 2) for v in b["box"])) for b in boxes]


def persist(force: bool = False) -> None:
    """Zapisuje ramki bieżącego obrazu, jeśli zmieniły się od ostatniego zapisu/wczytania."""
    ss = st.session_state
    cur = ss.get("cur")
    if not cur or not (force or cur["sig"] != signature(ss.boxes)):
        return
    path: Path = cur["label"]
    if cur["backup"] and path.is_file() and not cur["backup"].is_file():  # pierwotny plik raz, nigdy nie nadpisujemy
        cur["backup"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, cur["backup"])
    elif cur["backup"] and not path.is_file() and not cur["backup"].is_file():
        cur["backup"].parent.mkdir(parents=True, exist_ok=True)
        cur["backup"].write_text("")  # oryginał bez etykiet: pusty znacznik
    save_boxes(path, ss.boxes, cur["size"])
    cur["sig"] = signature(ss.boxes)


def color(c: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[(c * 47) % 180, 220, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def render(img, boxes, names, scale, cands, p1, hide_labels):
    canvas = np.array(img.resize((SHOW_W, int(img.height * scale))))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v * scale) for v in b["box"])
        col = color(b["cls"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        if not hide_labels:
            cv2.putText(canvas, f"{i}:{names[b['cls']]}", (x1, max(y1 - 4, 12)), 0, 0.55, col, 2)
    for j, cb in enumerate(cands):
        x1, y1, x2, y2 = (int(v * scale) for v in cb)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(canvas, f"?{j}", (x1, max(y1 - 4, 12)), 0, 0.6, (255, 255, 0), 2)
    if p1:
        cv2.drawMarker(canvas, (int(p1[0] * scale), int(p1[1] * scale)), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    return canvas


def main() -> None:
    st.set_page_config(page_title="Etykietowanie + SAM3", layout="wide")
    ss = st.session_state
    ss.setdefault("idx", 0)
    ss.setdefault("ver", 0)
    ss.setdefault("p1", None)
    ss.setdefault("cands", [])
    ss.setdefault("boxes_for", None)
    persist()  # zmiany z poprzedniego przebiegu (też przy zmianie obrazu), zanim wczytamy kolejny obraz

    with st.sidebar:
        data = Path(st.text_input("Zbiór (YOLO)", "data/polish-traffic-12k"))
        ss.split = st.selectbox("Podział", ["valid", "train", "test"])
        ss.in_dataset = st.checkbox("Zapis w zbiorze (autozapis, nadpisuje etykiety)", True)
        out = Path(st.text_input("Katalog zapisu", "runs/labels_fixed", disabled=ss.in_dataset))
        names = yaml.safe_load((data / "data.yaml").read_text())["names"]
        imgs = sorted(p for p in (data / ss.split / "images").glob("*") if AUG_MARK not in p.stem)
        if not imgs:
            st.error("Brak obrazów.")
            st.stop()
        only_todo = st.checkbox("Ukryj już poprawione", False)
        if only_todo:
            imgs = [p for p in imgs if not is_fixed(p.stem, data, out)] or imgs
        ss.idx = min(ss.idx, len(imgs) - 1)
        ss.idx = int(st.number_input(f"Obraz (1..{len(imgs)})", 1, len(imgs), ss.idx + 1)) - 1
        tool = st.radio("Narzędzie", TOOLS)
        cls = names.index(st.selectbox("Klasa nowej ramki", names))
        refine = st.checkbox("Dopracuj ramkę SAM-em", True, disabled=tool != TOOLS[1])
        hide = st.checkbox("Ukryj napisy", False)
        st.divider()
        prompt = st.text_input("Prompt tekstowy SAM3", "car")
        thr = st.slider("Próg", 0.1, 0.9, 0.5, 0.05)
        cls_txt = names.index(st.selectbox("Klasa kandydatów", names, key="cls_txt"))

    path = imgs[ss.idx]
    img = Image.open(path).convert("RGB")
    key = (str(path), ss.split)
    if ss.boxes_for != key:  # nowy obraz: wczytaj etykiety, wyczyść stan roboczy
        ss.boxes = load_boxes(path, img.size, data, out)
        ss.boxes_for, ss.cands, ss.p1 = key, [], None
        lp = label_path(path.stem, data, out)
        ss.cur = {
            "label": lp,
            "size": img.size,
            "sig": signature(ss.boxes),
            "backup": BACKUP / ss.split / (path.stem + ".txt") if ss.in_dataset else None,
        }
    scale = SHOW_W / img.width

    left, right = st.columns([3, 1])
    with left:
        st.caption(f"{path.name}  ·  {img.width}×{img.height}  ·  {'poprawiony' if is_fixed(path.stem, data, out) else 'oryginał'}")
        click = streamlit_image_coordinates(
            render(img, ss.boxes, names, scale, ss.cands, ss.p1, hide), key=f"img{ss.idx}_{ss.ver}", width=SHOW_W
        )

    with right:
        c1, c2, c3 = st.columns(3)
        if c1.button("◀"):
            ss.idx = max(ss.idx - 1, 0)
            st.rerun()
        if c3.button("▶"):
            ss.idx = min(ss.idx + 1, len(imgs) - 1)
            st.rerun()
        if c2.button("💾 Zapisz", type="primary"):
            persist(force=True)
            st.toast(f"Zapisano {ss.cur['label']}")
        if st.button("SAM3: znajdź po tekście"):
            with st.spinner("SAM3..."):
                ss.cands = sam_text(img, prompt, thr)
            st.rerun()
        if ss.cands:
            take = st.multiselect("Akceptuj kandydatów (?N)", range(len(ss.cands)), default=[])
            if st.button("Dodaj wybranych"):
                ss.boxes += [{"cls": cls_txt, "box": ss.cands[j]} for j in take]
                ss.cands = []
                st.rerun()
        if ss.p1 and st.button("Anuluj pierwszy róg"):
            ss.p1 = None
            st.rerun()

        edited = st.data_editor(
            [{"cls": names[b["cls"]], "x1": b["box"][0], "y1": b["box"][1], "x2": b["box"][2], "y2": b["box"][3]} for b in ss.boxes],
            column_config={"cls": st.column_config.SelectboxColumn("klasa", options=names, required=True)},
            num_rows="dynamic",
            key=f"tbl{ss.idx}_{len(ss.boxes)}_{ss.ver}",
            hide_index=False,
        )
        new = [
            {"cls": names.index(r["cls"]), "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]}
            for r in edited
            if r.get("cls") and all(r.get(k) is not None for k in ("x1", "y1", "x2", "y2"))
        ]
        if new != ss.boxes:
            ss.boxes = new
            ss.ver += 1
            st.rerun()

    if click:  # współrzędne w układzie wyświetlanego obrazu -> oryginał
        x, y = click["x"] / scale, click["y"] / scale
        done = True
        if tool == TOOLS[0]:
            with st.spinner("SAM3..."):
                box = sam_box(img, point=(x, y))
            if box:
                ss.boxes.append({"cls": cls, "box": box})
            else:
                st.warning("SAM3 nie zwrócił maski, kliknij inaczej.")
        elif tool == TOOLS[1]:
            if ss.p1 is None:
                ss.p1, done = (x, y), True
            else:
                x1, y1, x2, y2 = min(ss.p1[0], x), min(ss.p1[1], y), max(ss.p1[0], x), max(ss.p1[1], y)
                box = [x1, y1, x2, y2]
                if refine:
                    with st.spinner("SAM3..."):
                        box = sam_box(img, box=box) or box
                ss.boxes.append({"cls": cls, "box": box})
                ss.p1 = None
        else:
            hit = [
                i for i, b in enumerate(ss.boxes) if b["box"][0] <= x <= b["box"][2] and b["box"][1] <= y <= b["box"][3]
            ]
            if hit:  # najmniejsza ramka pod kursorem
                area = lambda i: (ss.boxes[i]["box"][2] - ss.boxes[i]["box"][0]) * (ss.boxes[i]["box"][3] - ss.boxes[i]["box"][1])  # noqa: E731
                ss.boxes.pop(min(hit, key=area))
        if done:
            ss.ver += 1  # nowy klucz komponentu zeruje zapamiętany klik
            st.rerun()


main()
