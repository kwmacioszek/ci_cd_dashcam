"""Głębia względna: Depth Anything V2 Small."""

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def near_score(depth: np.ndarray, box: np.ndarray) -> float | None:
    """Bliskość obiektu = mediana głębi w dolnej połowie skrzynki (stopy/zderzak, mniej tła).

    Skrzynka jest przycinana do obrazu. Zwraca None, gdy po przycięciu nic nie zostaje.
    """
    height, width = depth.shape
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), width - 1), min(int(box[3]), height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return float(np.median(depth[(y1 + y2) // 2 : y2, x1:x2]))


class DepthEstimator:
    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        from transformers import pipeline

        from perception.detect import pick_device

        self.pipe = pipeline(
            "depth-estimation", model=model_id, device=device or pick_device()
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: RGB uint8. Zwraca mapę (H, W) w 0..1, większe = bliżej (głębia WZGLĘDNA)."""
        import torch
        from PIL import Image

        d = self.pipe(Image.fromarray(img))["predicted_depth"]
        d = torch.nn.functional.interpolate(
            d.float().cpu().reshape(1, 1, *d.shape[-2:]),
            size=img.shape[:2],
            mode="bicubic",
        )[0, 0].numpy()
        return (d - d.min()) / (d.max() - d.min() + 1e-9)
