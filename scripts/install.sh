set -euo pipefail

TAG="${1:?użycie: install.sh <tag> [--from-dir DIR]}"
FROM_DIR=""
if [[ "${2:-}" == "--from-dir" ]]; then FROM_DIR="${3:?brak katalogu po --from-dir}"; fi

HOME_DIR="${PERCEPTION_HOME:-$HOME/perception}"
REL="$HOME_DIR/releases/$TAG"
mkdir -p "$HOME_DIR/releases"

if [[ -f "$REL/.ok" ]]; then
  echo "==> release $TAG już zainstalowany, przełączam (rollback/ponowna aktywacja)"
else
  STAGE="$(mktemp -d "$HOME_DIR/.stage.XXXXXX")"
  cleanup() {
    rm -rf "$STAGE"
    [[ -f "$REL/.ok" ]] || rm -rf "$REL" # niedokończona instalacja nigdy nie zostaje na dysku
  }
  trap cleanup EXIT

  if [[ -n "$FROM_DIR" ]]; then
    cp "$FROM_DIR"/*.whl "$FROM_DIR"/detr.onnx "$FROM_DIR"/metrics.json "$FROM_DIR"/SHA256SUMS "$STAGE"/
  else
    : "${PERCEPTION_REPO:?ustaw PERCEPTION_REPO=owner/repo}"
    gh release download "$TAG" -R "$PERCEPTION_REPO" -D "$STAGE"
  fi

  echo "==> weryfikacja sum kontrolnych"
  (cd "$STAGE" && { sha256sum -c SHA256SUMS 2>/dev/null || shasum -a 256 -c SHA256SUMS; })

  rm -rf "$REL"
  mkdir -p "$REL"
  mv "$STAGE"/*.whl "$STAGE"/detr.onnx "$STAGE"/metrics.json "$STAGE"/SHA256SUMS "$REL"/

  echo "==> instalacja do $REL/.venv"
  WHEEL="$(ls "$REL"/*.whl)"
  uv venv --quiet --python 3.11 "$REL/.venv"
  uv pip install --quiet --python "$REL/.venv/bin/python" "${WHEEL}[ml]" onnxruntime

  echo "==> test dymny"
  "$REL/.venv/bin/python" - "$REL/detr.onnx" "$TAG" <<'PY'
import sys
import onnxruntime as ort
import perception

onnx_path, tag = sys.argv[1], sys.argv[2]
assert f"v{perception.__version__}" == tag, f"wersja pakietu {perception.__version__} != tag {tag}"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
assert [i.name for i in sess.get_inputs()] == ["pixel_values"]
assert [o.name for o in sess.get_outputs()] == ["logits", "pred_boxes"]
print(f"OK: perception {perception.__version__}, model ONNX ładuje się")
PY
  touch "$REL/.ok"
fi

ln -sfn "$REL" "$HOME_DIR/current"
echo "==> aktywny release: $(readlink "$HOME_DIR/current")"
