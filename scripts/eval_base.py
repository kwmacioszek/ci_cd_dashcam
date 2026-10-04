"""Ewaluacja bazowego DETR-a (facebook/detr-resnet-50, bez dotrenowania) na walidacji polskiego zbioru.

Model zna tylko klasy COCO, więc klasy zbioru mapujemy na najbliższe klasy COCO (--map), a resztę
(znaki zakazu, ostrzegawcze, ograniczenia prędkości, przejścia) zostawiamy bez odpowiednika. GT to wyłącznie
etykiety zbioru (bez pseudo-etykiet). Trzy warianty:

  A  samo mapowanie klas (światła czerwone i zielone scalone w jedną klasę Traffic-Light)
  B  A + filtr domenowy z perception.detect.filter_detections (boksy-olbrzymy, maska maski, NMS)
  C  B + kolor światła z HSV wycinka (Red / Green zamiast jednej klasy)

  python scripts/fetch_val.py                       # najpierw dane (HF_TOKEN z .env)
  python scripts/eval_base.py --json runs/base_val.json
  python scripts/eval_base.py --limit 50            # test dymny
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envfile import load_env  # noqa: E402

from perception.detect import MODEL_ID, filter_detections  # noqa: E402

AUG_MARK = "_aug_out_"
# klasa zbioru -> klasa COCO (None = DETR jej nie zna)
COCO_MAP = {
    "Car": "car",
    "Truck": "truck",
    "Motorcycle": "motorcycle",
    "Pedestrian": "person",
    "Red-Traffic-Light": "traffic light",
    "Green-Traffic-Light": "traffic light",
    "Prohibition-Sign": "stop sign",  # B-20 STOP należy do grupy znaków zakazu
}
LIGHTS = ("Red-Traffic-Light", "Green-Traffic-Light")


def read_gt(label: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO (cls cx cy w h, ewentualnie wielokąt) -> boksy xyxy w pikselach i klasy."""
    boxes, cls = [], []
    if label.is_file():
        for line in label.read_text().split("\n"):
            t = line.split()
            if len(t) < 5:
                continue
            v = np.array(t[1:], dtype=float)
            if len(t) == 5:
                cx, cy, bw, bh = v
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            else:  # segmentacja: obwiednia wielokąta
                xs, ys = v[0::2], v[1::2]
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
            cls.append(int(t[0]))
    return np.array(boxes).reshape(-1, 4), np.array(cls, dtype=int)


def light_color(img: np.ndarray, box: np.ndarray) -> str:
    """Czerwone czy zielone światło: więcej jasnych, nasyconych pikseli odpowiedniego odcienia."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = img[max(y1, 0) : max(y2, 1), max(x1, 0) : max(x2, 1)]
    if crop.size == 0:
        return "Red-Traffic-Light"
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat > 90) & (val > 150)
    red = bright & ((hue < 10) | (hue > 165))
    green = bright & (hue > 40) & (hue < 95)
    return "Green-Traffic-Light" if green.sum() > red.sum() else "Red-Traffic-Light"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def prf(preds, gts, classes, thr, iou_thr=0.5):
    """Precision / recall / F1 per klasa przy progu pewności thr (zachłanne dopasowanie po score)."""
    out = {}
    for c in classes:
        tp = fp = fn = 0
        for p, g in zip(preds, gts, strict=True):
            pb, ps, pl = p
            gb, gl = g
            pb, ps = pb[(pl == c) & (ps >= thr)], ps[(pl == c) & (ps >= thr)]
            gb = gb[gl == c]
            used = set()
            if len(pb) and len(gb):
                iou = iou_matrix(pb, gb)
                for i in np.argsort(-ps):
                    j = int(np.argmax(np.where([k in used for k in range(len(gb))], -1, iou[i])))
                    if iou[i, j] >= iou_thr and j not in used:
                        used.add(j)
                        tp += 1
                    else:
                        fp += 1
            else:
                fp += len(pb)
            fn += len(gb) - len(used)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {
            "precision": pr,
            "recall": rc,
            "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def map_metrics(preds, gts, n_classes):
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    t = torch.tensor
    m.update(
        [
            {
                "boxes": t(b, dtype=torch.float32).reshape(-1, 4),
                "scores": t(s, dtype=torch.float32),
                "labels": t(lab),
            }
            for b, s, lab in preds
        ],
        [{"boxes": t(b, dtype=torch.float32).reshape(-1, 4), "labels": t(lab)} for b, lab in gts],
    )
    r = m.compute()
    classes = r["classes"].tolist()
    per = dict(zip(classes, r["map_per_class"].tolist(), strict=True))
    return {
        "map": float(r["map"]),
        "map50": float(r["map_50"]),
        "map75": float(r["map_75"]),
        "mar100": float(r["mar_100"]),
        "per_class_ap": {c: per.get(c, -1.0) for c in range(n_classes)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/polish-traffic-12k"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device")
    ap.add_argument("--thr", type=float, default=0.5, help="próg pewności dla precision/recall")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-aug", action="store_true", help="pomiń kopie augmentowane (_aug_out_)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import torch
    import yaml
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor

    from perception.detect import pick_device

    load_env()  # HF_TOKEN z .env (wagi / ewentualnie prywatny zbiór)
    names = yaml.safe_load((args.data / "data.yaml").read_text())["names"]
    idx = {n: i for i, n in enumerate(names)}
    merged = [n for n in names if n not in LIGHTS] + ["Traffic-Light"]  # klasy wariantów A i B
    midx = {n: i for i, n in enumerate(merged)}
    to_merged = np.array([midx["Traffic-Light" if n in LIGHTS else n] for n in names])

    imgs = sorted(
        p
        for p in (args.data / args.split / "images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.no_aug:
        imgs = [p for p in imgs if AUG_MARK not in p.stem]
    if args.limit:
        imgs = imgs[: args.limit]

    device = args.device or pick_device()
    proc = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(args.model, attn_implementation="eager").to(device).eval()  # type: ignore[arg-type]
    id2label = {int(i): str(n) for i, n in model.config.id2label.items()}
    coco2idx = {n: i for i, n in id2label.items()}
    # klasa COCO -> klasa zbioru (dla wariantu C światła rozstrzyga kolor)
    coco_to_ds = {coco2idx[v]: idx[k] for k, v in COCO_MAP.items() if k not in LIGHTS}
    coco_to_ds[coco2idx["traffic light"]] = idx[LIGHTS[0]]
    road_ids = set(coco_to_ds)
    print(f"{len(imgs)} obrazów z {args.data / args.split}, urządzenie {device}", flush=True)

    raw, gts, shapes, pics = [], [], [], []
    t0 = time.time()
    for n, path in enumerate(imgs, 1):
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        gts.append(read_gt(args.data / args.split / "labels" / (path.stem + ".txt"), w, h))
        with torch.no_grad():
            inp = proc(images=img, return_tensors="pt").to(device)
            res = proc.post_process_object_detection(
                model(**inp), threshold=0.05, target_sizes=torch.tensor([[h, w]])
            )[0]
        raw.append((res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()))
        shapes.append((h, w))
        pics.append(img if len(imgs) <= 1200 else None)
        if n % 100 == 0:
            print(f"  {n}/{len(imgs)}  ({(time.time() - t0) / n:.2f}s/obraz)", flush=True)
    dt = (time.time() - t0) / len(imgs)

    def to_ds(b, s, l, shape, img, color):  # noqa: E741
        keep = np.array([int(x) in road_ids for x in l], dtype=bool)
        b, s, l = b[keep], s[keep], l[keep]  # noqa: E741
        ds = np.array([coco_to_ds[int(x)] for x in l], dtype=int)
        if color and len(b):
            for k in np.where(ds == idx[LIGHTS[0]])[0]:
                ds[k] = idx[light_color(img, b[k])]
        return b, s, ds

    variants = {}
    for key, filt, color in (("A", False, False), ("B", True, False), ("C", True, True)):
        preds = []
        for (b, s, l), shape, img in zip(raw, shapes, pics, strict=True):  # noqa: E741
            if filt:
                b, s, l = filter_detections(b, s, l, shape, road_ids, nms_iou=0.7)  # noqa: E741
            preds.append(to_ds(b, s, l, shape, img, color))
        if color:
            p_eval, g_eval, classes, cnames = preds, gts, [idx[n] for n in names], names
        else:
            p_eval = [(b, s, to_merged[lab]) for b, s, lab in preds]
            g_eval = [(b, to_merged[lab]) for b, lab in gts]
            classes, cnames = list(range(len(merged))), merged
        m = map_metrics(p_eval, g_eval, len(cnames))
        pr = prf(p_eval, g_eval, classes, args.thr)
        n_gt = {c: int(sum((g[1] == c).sum() for g in g_eval)) for c in classes}
        m["per_class"] = {
            cnames[i]: {"ap": m["per_class_ap"][c], "n_gt": n_gt[c], **pr[c]} for i, c in enumerate(classes)
        }
        del m["per_class_ap"]
        # metryki zbiorcze tylko po klasach, które DETR w ogóle zna
        known = [cnames[i] for i, c in enumerate(classes) if cnames[i] in {"Traffic-Light", *COCO_MAP}]
        tp = sum(m["per_class"][n]["tp"] for n in known)
        fp = sum(m["per_class"][n]["fp"] for n in known)
        fn_all = sum(v["fn"] for v in m["per_class"].values())
        m["micro_precision_known"] = tp / (tp + fp) if tp + fp else 0.0
        m["micro_recall_all"] = tp / (tp + fn_all) if tp + fn_all else 0.0
        variants[key] = m
        print(f"\n== Wariant {key}: mAP {m['map']:.3f}  AP50 {m['map50']:.3f}  mAR100 {m['mar100']:.3f}")
        for n, v in m["per_class"].items():
            print(f"  {n:<22} AP {v['ap']:.3f}  P {v['precision']:.2f}  R {v['recall']:.2f}  GT {v['n_gt']}")

    out = {
        "model": args.model,
        "split": args.split,
        "images": len(imgs),
        "sec_per_image": dt,
        "device": device,
        "conf_threshold_pr": args.thr,
        "coco_map": COCO_MAP,
        "variants": variants,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nZapisano {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
