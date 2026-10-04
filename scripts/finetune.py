
import argparse
import json
import math
import random
import sys
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from perception.boxes import iou_matrix

BASE_MODEL = "facebook/detr-resnet-50"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NO_ARRAY = np.zeros((0, 5))
DEFAULT_ALIAS = {"pedestrian": "person"}  # nazwa w zbiorach drogowych -> klasa COCO
AUG_MARK = "_aug_out_"  # kopie augmentowane z Roboflow (w tym zbiorze także w valid/test)


@dataclass
class Item:
    path: Path
    gt: np.ndarray  # (n, 5): id klasy, x1, y1, x2, y2 (znormalizowane do 0..1)
    pseudo: np.ndarray = field(default_factory=lambda: NO_ARRAY)


# ---------- czyste funkcje (testowalne bez modeli) ----------


def norm_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def build_label_map(
    names: list[str], coco_id2label: dict[int, str], alias: dict[str, str] | None = None
) -> tuple[dict[int, int], list[str]]:
    """Klasy zbioru -> id modelu. Nazwa zgodna z COCO dostaje stare id, reszta nowe (od len(coco))."""
    by_name = {norm_name(n): i for i, n in coco_id2label.items() if n != "N/A"}
    alias = {norm_name(k): norm_name(v) for k, v in {**DEFAULT_ALIAS, **(alias or {})}.items()}
    n_old = len(coco_id2label)
    mapping: dict[int, int] = {}
    new_names: list[str] = []
    for src_id, name in enumerate(names):
        key = alias.get(norm_name(name), norm_name(name))
        if key in by_name:
            mapping[src_id] = by_name[key]
        else:
            mapping[src_id] = n_old + len(new_names)
            new_names.append(name)
    return mapping, new_names


def parse_yolo_label(text: str, mapping: dict[int, int]) -> np.ndarray:
    """Linie `cls cx cy w h` (lub wielokąt `cls x1 y1 x2 y2 ...`) -> (n, 5) z xyxy w 0..1."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) not in mapping:
            continue
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        if x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
            rows.append([mapping[int(float(parts[0]))], x1, y1, x2, y2])
    return np.array(rows, dtype=float).reshape(-1, 5)


def merge_boxes(gt: np.ndarray, pseudo: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Etykiety ze zbioru wygrywają: pseudo-skrzynka nakładająca się na GT (IoU >= próg) jest odrzucana."""
    if len(pseudo) == 0:
        return gt
    if len(gt):
        ious = iou_matrix(pseudo[:, 1:], gt[:, 1:])
        pseudo = pseudo[ious.max(axis=1) < iou_thr]
    return np.concatenate([gt, pseudo]) if len(gt) else pseudo


def drop_augmented(paths: list[Path]) -> list[Path]:
    """Bez kopii augmentowanych: ich bliźniaki z oryginałem zawyżałyby mAP na walidacji."""
    return [p for p in paths if AUG_MARK not in p.name]


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# ---------- wczytywanie zbioru YOLO ----------


def _split_dir(root: Path, cfg: dict, split: str) -> Path:
    keys = ["val", "valid", "test"] if split == "val" else [split]
    bases = [root, Path(cfg["path"]) if "path" in cfg else root]
    cands = [b / cfg[k] for k in keys if k in cfg for b in bases]
    cands += [root / "images" / keys[0], root / keys[0] / "images", root / keys[0]]
    cands += [root / "images" / k for k in keys] + [root / k / "images" for k in keys]
    for c in cands:
        if c.is_dir():
            return c
    raise SystemExit(f"Nie znalazłem katalogu '{split}' w {root} (sprawdź data.yaml).")


def extract_shards(root: Path) -> int:
    """Rozpakowuje `root/data/*.tar` do `root` (układ YOLO: train/, valid/, test/). Zwraca liczbę shardów."""
    shards = sorted((root / "data").glob("*.tar"))
    for tar in shards:
        with tarfile.open(tar) as tf:
            tf.extractall(root, filter="data")  # filter="data": bez ścieżek wychodzących poza root
    return len(shards)


def fetch_hf_dataset(repo: str, dest: Path) -> Path:
    """Pobiera zbiór z HF i rozpakowuje shardy; ponowne uruchomienie nic nie pobiera (znacznik .extracted)."""
    marker = dest / ".extracted"
    if not marker.is_file():
        from huggingface_hub import snapshot_download

        print(f"Pobieram {repo} do {dest} (kilkanaście GB, wznawialne)...", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest)
        print(f"Rozpakowano shardów: {extract_shards(dest)}", flush=True)
        marker.write_text(repo)
    return dest


def load_yolo(root: Path, split: str) -> tuple[list[str], list[Path], Path]:
    import yaml

    yamls = sorted(root.rglob("data.yaml")) or sorted(root.rglob("*.yaml"))
    if not yamls:
        raise SystemExit(f"Brak data.yaml w {root}: potrzebuję listy nazw klas.")
    root = yamls[0].parent
    cfg = yaml.safe_load(yamls[0].read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    img_dir = _split_dir(root, cfg, split)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [str(n) for n in names], images, img_dir


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def build_items(images: list[Path], mapping: dict[int, int], limit: int | None) -> list[Item]:
    items = []
    for p in images[:limit]:
        lp = label_path(p)
        gt = parse_yolo_label(lp.read_text(), mapping) if lp.is_file() else NO_ARRAY
        items.append(Item(p, gt))
    return items


# ---------- część z modelami ----------


def pick_device() -> str:
    from perception.detect import pick_device as pd

    return pd()


class ImageSet:
    """Zwraca (PIL, Item); przetwarzanie wstępne dzieje się w collate, czyli w workerach."""

    def __init__(self, items: list[Item]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        return Image.open(self.items[i].path).convert("RGB"), self.items[i]


def add_pseudo_labels(
    items: list[Item], cache: Path, device: str, thr: float, batch: int, workers: int, short: int
) -> None:
    """Pseudo-etykiety starych klas z oryginalnego DETR; cache w JSON pozwala wznowić."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DetrForObjectDetection

    store: dict[str, list] = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = [it for it in items if str(it.path) not in store]
    if todo:
        proc = make_processor(short)
        model = DetrForObjectDetection.from_pretrained(BASE_MODEL, attn_implementation="eager")
        model.to(device).eval()  # type: ignore[arg-type]
        dl: DataLoader = DataLoader(
            ImageSet(todo),  # type: ignore[arg-type]
            batch_size=batch,
            num_workers=workers,
            collate_fn=lambda b: b,
        )
        for n, chunk in enumerate(dl):
            imgs = [c[0] for c in chunk]
            with torch.no_grad():
                inp = proc(images=imgs, return_tensors="pt").to(device)
                out = model(**inp)
                res = proc.post_process_object_detection(
                    out, threshold=thr, target_sizes=[(im.height, im.width) for im in imgs]
                )
            for (im, it), r in zip(chunk, res, strict=True):
                scale = np.array([im.width, im.height, im.width, im.height], dtype=float)
                boxes = r["boxes"].cpu().numpy() / scale
                labels = r["labels"].cpu().numpy()
                store[str(it.path)] = np.column_stack([labels, boxes]).tolist() if len(labels) else []
            if n % 20 == 0:
                print(f"  pseudo-etykiety: {min((n + 1) * batch, len(todo))}/{len(todo)}", flush=True)
                cache.write_text(json.dumps(store))
        cache.write_text(json.dumps(store))
        del model
    for it in items:
        it.pseudo = np.array(store[str(it.path)], dtype=float).reshape(-1, 5)


def make_processor(short: int):
    from transformers import DetrImageProcessor

    return DetrImageProcessor.from_pretrained(
        BASE_MODEL, size={"shortest_edge": short, "longest_edge": int(short * 1333 / 800)}
    )


def to_coco(item: Item, width: int, height: int, image_id: int, iou_thr: float) -> dict:
    boxes = merge_boxes(item.gt, item.pseudo, iou_thr)
    anns = []
    for lab, x1, y1, x2, y2 in boxes:
        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        anns.append(
            {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "category_id": int(lab),
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    return {"image_id": image_id, "annotations": anns}


def expand_head(model, n_new: int) -> None:
    """Powiększa głowę: stare klasy i 'no-object' są kopiowane, nowe wchodzą przed 'no-object'."""
    import torch

    old = model.class_labels_classifier
    n_old = model.config.num_labels
    new = torch.nn.Linear(old.in_features, n_old + n_new + 1)
    with torch.no_grad():
        new.weight[:n_old] = old.weight[:n_old]
        new.bias[:n_old] = old.bias[:n_old]
        new.weight[n_old : n_old + n_new].normal_(0, 0.01)
        new.bias[n_old : n_old + n_new] = old.bias[:n_old].mean()
        new.weight[-1] = old.weight[-1]
        new.bias[-1] = old.bias[-1]
    model.class_labels_classifier = new
    model.config.num_labels = n_old + n_new


def distill_loss(s_logits, s_boxes, t_logits, t_boxes, n_old: int, box_weight: float = 1.0):
    """Destylacja z oryginalnego DETR (to samo zapytanie = ten sam slot, bo start z tych samych wag).

    Klasy: KL(nauczyciel || uczeń) w podprzestrzeni stare klasy + 'no-object'; logity nowych klas są
    pomijane, więc masa prawdopodobieństwa może swobodnie przechodzić na nowe klasy. Boksy: L1 ważone
    pewnością nauczyciela, żeby zapytania 'no-object' nie ściągały boksów w losowe miejsca.
    """
    import torch
    import torch.nn.functional as F

    s = torch.cat([s_logits[..., :n_old], s_logits[..., -1:]], dim=-1).float()
    t_prob = t_logits.float().softmax(-1)
    kl = F.kl_div(s.log_softmax(-1), t_prob, reduction="none").sum(-1)
    conf = t_prob[..., :-1].max(-1).values
    l1 = (s_boxes.float() - t_boxes.float()).abs().sum(-1)
    box = (conf * l1).sum() / conf.sum().clamp(min=1e-6)
    return kl.mean() + box_weight * box, kl.mean().detach(), box.detach()


def set_freeze(model, mode: str) -> None:
    prefixes = {
        "backbone": ["model.backbone"],
        "encoder": ["model.backbone", "model.encoder", "model.input_projection"],
    }
    for name, p in model.named_parameters():
        p.requires_grad = not any(name.startswith(pre) for pre in prefixes.get(mode, []))


def evaluate_map(model, proc, items: list[Item], device: str, iou_thr: float, n_old: int, batch: int) -> dict:
    """mAP względem GT zbioru + pseudo-GT starych klas. Retencja = mAP po klasach < n_old."""
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    model.eval()
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        pairs = [ImageSet([it])[0] for it in chunk]
        imgs = [p[0] for p in pairs]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            out = model(**inp)
            res = proc.post_process_object_detection(
                out, threshold=0.05, target_sizes=[(im.height, im.width) for im in imgs]
            )
        preds, tgts = [], []
        for im, it, r in zip(imgs, chunk, res, strict=True):
            scale = torch.tensor([im.width, im.height, im.width, im.height], dtype=torch.float32)
            gt = merge_boxes(it.gt, it.pseudo, iou_thr)
            preds.append(
                {"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(), "labels": r["labels"].cpu()}
            )
            tgts.append(
                {
                    "boxes": torch.tensor(gt[:, 1:], dtype=torch.float32) * scale,
                    "labels": torch.tensor(gt[:, 0], dtype=torch.long),
                }
            )
        metric.update(preds, tgts)
    res = metric.compute()
    classes = res["classes"].tolist()
    per_class = {int(c): float(v) for c, v in zip(classes, res["map_per_class"].tolist(), strict=True)}

    def mean(sel: list[float]) -> float | None:
        return float(np.mean(sel)) if sel else None

    old = [v for c, v in per_class.items() if c < n_old and v >= 0]
    new = [v for c, v in per_class.items() if c >= n_old and v >= 0]
    return {"map_old": mean(old), "map_new": mean(new), "map_all": float(res["map"]), "per_class": per_class}


def eval_log(metrics: dict, id2label: dict[int, str]) -> dict[str, float]:
    """Metryki ewaluacji w formie płaskiego słownika dla W&B (klucze `eval/...`)."""
    out = {f"eval/{k}": metrics[k] for k in ("map_old", "map_new", "map_all") if metrics[k] is not None}
    for cls, ap in metrics["per_class"].items():
        out[f"eval/ap/{id2label.get(cls, cls)}"] = ap
    return out


def init_wandb(args: argparse.Namespace, config: dict):
    """Uruchamia przebieg W&B (None, gdy wyłączone). ID zapisane w --out pozwala wznowić ten sam wykres."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        raise SystemExit("Brak pakietu wandb: uv sync --extra train (albo pip install wandb).") from None
    id_file = args.out / "wandb_id.txt"
    run_id = id_file.read_text().strip() if id_file.is_file() else uuid.uuid4().hex[:8]
    id_file.write_text(run_id)
    return wandb.init(
        project=args.wandb, name=args.wandb_name, id=run_id, resume="allow", config=config, dir=args.out
    )


def save_model_card(out: Path, names: list[str], args: argparse.Namespace) -> None:
    (out / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: facebook/detr-resnet-50\n---\n\n"
        "# DETR-ResNet-50 dotrenowany na polskich znakach i obiektach drogowych\n\n"
        f"Stare klasy COCO zachowane (pseudo-etykiety z oryginalnego modelu, próg {args.pseudo_thr}). "
        f"Nowe klasy: {', '.join(names) or 'brak'}.\n\n"
        "Zbiór: [Traffic Road Object Detection Polish 12k](https://www.kaggle.com/datasets/mikoajkoek/"
        "traffic-road-object-detection-polish-12k) (autor: mikoajkoek, Apache 2.0). "
        "Zmiany: konwersja formatu, dodane pseudo-etykiety klas COCO.\n\n"
        "Model bazowy: facebook/detr-resnet-50 (Apache 2.0).\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data",
        type=Path,
        help="katalog zbioru YOLO (z data.yaml); z --hf-dataset: dokąd pobrać (domyślnie data/<nazwa>)",
    )
    ap.add_argument(
        "--hf-dataset", metavar="REPO", help="pobierz zbiór z HF, np. marcin119a/polish-traffic-12k"
    )
    ap.add_argument("--out", type=Path, default=Path("runs/finetune"))
    ap.add_argument("--replay-dir", type=Path, help="dodatkowe obrazy bez etykiet (tylko pseudo-etykiety)")
    ap.add_argument(
        "--alias", action="append", default=[], metavar="KLASA=COCO", help="np. pedestrian=person"
    )
    ap.add_argument("--freeze", choices=["backbone", "encoder", "none"], default="backbone")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="wagi pretrenowane")
    ap.add_argument("--lr-head", type=float, default=1e-4, help="głowa klasyfikacyjna")
    ap.add_argument("--short-edge", type=int, default=640)
    ap.add_argument("--pseudo-thr", type=float, default=0.7)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument(
        "--distill", type=float, default=0.0, help="waga destylacji z oryginalnego DETR (0 = wyłączona)"
    )
    ap.add_argument("--distill-box", type=float, default=1.0, help="waga członu boksów w destylacji")
    ap.add_argument("--min-old-map", type=float, default=0.8, help="próg retencji starych klas (heurystyka)")
    ap.add_argument("--eval-every", type=int, default=0, help="co ile epok liczyć mAP (0 = tylko na końcu)")
    ap.add_argument("--keep-aug-val", action="store_true", help=f"nie wycinaj kopii '{AUG_MARK}' z walidacji")
    ap.add_argument("--limit", type=int, help="tylko N obrazów (test dymny)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="wypisz mapowanie klas i statystyki, nie trenuj")
    ap.add_argument("--push-to-hub", metavar="REPO", help="wypchnij model do prywatnego repo na HF")
    ap.add_argument("--wandb", metavar="PROJEKT", help="loguj do Weights & Biases w tym projekcie")
    ap.add_argument("--wandb-name", help="nazwa przebiegu W&B")
    ap.add_argument("--log-every", type=int, default=10, help="co ile kroków logować stratę do W&B")
    args = ap.parse_args()
    if not (args.data or args.hf_dataset):
        ap.error("podaj --data (katalog lokalny) albo --hf-dataset (repo na HF)")
    if args.hf_dataset:
        args.data = fetch_hf_dataset(
            args.hf_dataset, args.data or Path("data") / args.hf_dataset.split("/")[-1]
        )

    import torch
    from torch.utils.data import DataLoader
    from transformers import DetrConfig, DetrForObjectDetection

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    args.out.mkdir(parents=True, exist_ok=True)

    id2label = DetrConfig.from_pretrained(BASE_MODEL).id2label or {}
    coco = {int(i): str(n) for i, n in id2label.items()}
    names, train_imgs, _ = load_yolo(args.data, "train")
    _, val_imgs, _ = load_yolo(args.data, "val")
    n_val_all = len(val_imgs)
    if not args.keep_aug_val:
        val_imgs = drop_augmented(val_imgs)
    alias = dict(a.split("=", 1) for a in args.alias)
    mapping, new_names = build_label_map(names, coco, alias)

    print(f"Mapowanie klas zbioru -> model (stare id < {len(coco)}, nowe >= {len(coco)}):")
    for src, dst in mapping.items():
        tag = f"= COCO '{coco[dst]}'" if dst < len(coco) else "NOWA"
        print(f"  {src:3d} {names[src]:<24} -> {dst:3d} {tag}")
    print(f"Nowych klas: {len(new_names)}, obrazów train/val: {len(train_imgs)}/{len(val_imgs)}")
    if n_val_all != len(val_imgs):
        print(
            f"  (walidacja bez {n_val_all - len(val_imgs)} kopii augmentowanych; --keep-aug-val je zostawia)"
        )
    if args.dry_run:
        print("Jeśli NOWA klasa to w rzeczywistości klasa COCO, dodaj np. --alias pedestrian=person")
        return 0

    train = build_items(train_imgs, mapping, args.limit)
    val = build_items(val_imgs, mapping, args.limit and max(args.limit // 5, 8))
    if args.replay_dir:
        replay = sorted(p for p in args.replay_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        train += [Item(p, NO_ARRAY) for p in replay[: args.limit]]
    print("Pseudo-etykiety starych klas z oryginalnego modelu...")
    for tag, items in (("train", train), ("val", val)):
        add_pseudo_labels(
            items, args.out / f"pseudo_{tag}.json", device, args.pseudo_thr, 8, args.workers, args.short_edge
        )
    print(f"Skrzynki: GT {sum(len(i.gt) for i in train)}, pseudo {sum(len(i.pseudo) for i in train)} (train)")

    model = DetrForObjectDetection.from_pretrained(BASE_MODEL, attn_implementation="eager")
    n_old = len(coco)
    expand_head(model, len(new_names))
    model.config.id2label = {**coco, **{n_old + i: n for i, n in enumerate(new_names)}}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    set_freeze(model, args.freeze)
    teacher = None
    if args.distill > 0:
        teacher = DetrForObjectDetection.from_pretrained(BASE_MODEL, attn_implementation="eager")
        teacher.to(device).eval()  # type: ignore[arg-type]
        for p in teacher.parameters():
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = init_wandb(
        args,
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "n_train": len(train),
            "n_val": len(val),
            "new_classes": new_names,
            "trainable_params": trainable,
            "device": device,
        },
    )
    model.to(device)  # type: ignore[arg-type]

    head = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("class_labels_classifier")
    ]
    rest = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("class_labels_classifier")
    ]
    opt = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": head, "lr": args.lr_head}], weight_decay=1e-4
    )
    proc = make_processor(args.short_edge)

    def collate(batch):
        imgs = [b[0] for b in batch]
        anns = [to_coco(b[1], b[0].width, b[0].height, i, args.merge_iou) for i, b in enumerate(batch)]
        return proc(images=imgs, annotations=anns, format="coco_detection", return_tensors="pt")

    dl: DataLoader = DataLoader(
        ImageSet(train),  # type: ignore[arg-type]
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
    )
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total, min(200, total // 10 or 1)))
    cuda = device == "cuda"
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=cuda and amp_dtype == torch.float16)

    last, start_epoch = args.out / "last.pt", 0
    if last.is_file():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        print(f"Wznawiam od epoki {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = run_cls = run_box = 0.0
        for step, batch in enumerate(dl):
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cuda):
                px, pm = batch["pixel_values"].to(device), batch["pixel_mask"].to(device)
                out = model(pixel_values=px, pixel_mask=pm, labels=labels)
                loss_total = out.loss
                if teacher is not None:
                    with torch.no_grad():
                        tout = teacher(pixel_values=px, pixel_mask=pm)
                    d_loss, d_cls, d_box = distill_loss(
                        out.logits, out.pred_boxes, tout.logits, tout.pred_boxes, n_old, args.distill_box
                    )
                    loss_total = loss_total + args.distill * d_loss
                    run_cls += float(d_cls)
                    run_box += float(d_box)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss_total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss = out.loss.item()
            running += loss
            if run and step % args.log_every == 0:
                run.log(
                    {
                        "train/loss": loss,
                        "train/lr": sched.get_last_lr()[0],
                        "train/lr_head": sched.get_last_lr()[1],
                        "epoch": epoch + step / len(dl),
                    },
                    step=epoch * len(dl) + step,
                )
            if step % 50 == 0:
                extra = f" kl {run_cls / (step + 1):.4f} box {run_box / (step + 1):.4f}" if teacher else ""
                print(
                    f"epoka {epoch + 1}/{args.epochs} krok {step}/{len(dl)} "
                    f"loss {running / (step + 1):.3f}{extra}",
                    flush=True,
                )
        if run:
            run.log({"train/epoch_loss": running / len(dl)}, step=(epoch + 1) * len(dl) - 1)
        tmp = last.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
            },
            tmp,
        )
        tmp.replace(last)
        if args.eval_every and (epoch + 1) % args.eval_every == 0 and epoch + 1 < args.epochs:
            m = evaluate_map(model, proc, val, device, args.merge_iou, n_old, args.batch)
            print(f"  mAP stare {m['map_old']}, nowe {m['map_new']}", flush=True)
            if run:
                run.log(eval_log(m, model.config.id2label), step=(epoch + 1) * len(dl) - 1)

    metrics = evaluate_map(model, proc, val, device, args.merge_iou, n_old, args.batch)
    final = args.out / "final"
    model.save_pretrained(final)
    proc.save_pretrained(final)
    (final / "road_classes.json").write_text(json.dumps(new_names, ensure_ascii=False))
    (args.out / "eval.json").write_text(json.dumps(metrics, indent=2))
    save_model_card(final, new_names, args)
    print(
        f"mAP stare klasy (retencja): {metrics['map_old']}, nowe klasy: {metrics['map_new']}, "
        f"wszystkie: {metrics['map_all']:.3f}"
    )
    if run:
        run.log(eval_log(metrics, model.config.id2label), step=max(args.epochs * len(dl) - 1, 0))
        run.summary.update({k: v for k, v in eval_log(metrics, model.config.id2label).items()})
        run.finish()
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        proc.push_to_hub(args.push_to_hub, private=True)
    old = metrics["map_old"]
    if old is not None and old < args.min_old_map:
        print(
            f"UWAGA: retencja starych klas {old:.3f} < {args.min_old_map}: nie podmieniaj modelu.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
