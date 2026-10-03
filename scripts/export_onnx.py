import argparse
import sys

import numpy as np
import torch

from perception.detect import MODEL_ID


class Wrapper(torch.nn.Module):
    """DETR zwraca ModelOutput; do ONNX potrzebujemy zwykłej krotki tensorów."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="detr.onnx")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from transformers import DetrForObjectDetection

    model = DetrForObjectDetection.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    wrapped = Wrapper(model).eval()
    dummy = torch.randn(1, 3, args.height, args.width)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            args.out,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(args.out))

    with torch.no_grad():
        ref_logits, ref_boxes = wrapped(dummy)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    logits, boxes = sess.run(None, {"pixel_values": dummy.numpy()})
    d_logits = float(np.abs(logits - ref_logits.numpy()).max())
    d_boxes = float(np.abs(boxes - ref_boxes.numpy()).max())
    print(f"max |onnx - torch|: logits={d_logits:.2e} boxes={d_boxes:.2e} (próg {args.atol})")
    if d_logits > args.atol or d_boxes > args.atol:
        print("BŁĄD: ONNX rozjeżdża się z PyTorchem", file=sys.stderr)
        return 1
    print(f"OK: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
