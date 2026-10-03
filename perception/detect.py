import numpy as np

from perception.boxes import nms

MODEL_ID = "facebook/detr-resnet-50"
ROAD = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


def filter_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    road_ids: set[int],
    max_area: float = 0.3,
    max_cy: float = 0.82,
    nms_iou: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zostawia klasy drogowe, odrzuca gigantyczne boksy i maskę własnego auta, robi NMS ponad klasami."""
    height, width = shape
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (height * width)
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    keep = np.array([int(lab) in road_ids for lab in labels], dtype=bool)
    keep &= (area < max_area) & (cy < max_cy)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    kept = nms(boxes, scores, nms_iou)
    return boxes[kept], scores[kept], labels[kept]


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.torch = torch
        self.device = device or pick_device()
        self.processor = DetrImageProcessor.from_pretrained(model_id)
        self.model = DetrForObjectDetection.from_pretrained(model_id, attn_implementation="eager")
        self.model.to(self.device)  # type: ignore[arg-type]  # stuby torcha
        self.model.eval()
        self.id2label = {int(i): str(n) for i, n in (self.model.config.id2label or {}).items()}
        self.road_ids = {i for i, n in self.id2label.items() if n in ROAD}

    def __call__(self, img: np.ndarray, thr: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """img: RGB uint8 (H, W, 3). Zwraca (boxes xyxy, scores, labels)."""
        torch = self.torch
        with torch.no_grad():
            inp = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            res = self.processor.post_process_object_detection(
                out, threshold=thr, target_sizes=torch.tensor([img.shape[:2]])
            )[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        return filter_detections(boxes, scores, labels, img.shape[:2], self.road_ids)
