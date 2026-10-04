"""Pobiera z Hugging Face tylko podział walidacyjny (valid) zbioru polskich obiektów drogowych.

  python scripts/fetch_val.py                                  # -> data/polish-traffic-12k
  python scripts/fetch_val.py --repo marcin119a/polish-traffic-12k --out data/val-only
  python scripts/fetch_val.py --split test                     # inny podział

Prywatne repo: HF_TOKEN w środowisku albo `hf auth login`.
"""

import argparse
import tarfile
from pathlib import Path

from envfile import load_env


def fetch_split(repo: str, out: Path, split: str) -> int:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=out,
        allow_patterns=["data.yaml", "README.md", "LICENSE", f"data/{split}-*.tar"],
    )
    shards = sorted((out / "data").glob(f"{split}-*.tar"))
    for tar in shards:
        with tarfile.open(tar) as tf:
            tf.extractall(out, filter="data")  # bez ścieżek wychodzących poza out
    return len(shards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--repo", default="marcin119a/polish-traffic-12k", help="zbiór na HF (domyślnie %(default)s)"
    )
    ap.add_argument("--out", type=Path, help="katalog docelowy (domyślnie data/<nazwa repo>)")
    ap.add_argument("--split", default="valid", help="podział do pobrania (domyślnie %(default)s)")
    args = ap.parse_args()

    load_env()  # HF_TOKEN z .env
    out = args.out or Path("data") / args.repo.split("/")[-1]
    print(f"Pobieram {args.split} z {args.repo} do {out}...", flush=True)
    n = fetch_split(args.repo, out, args.split)
    if n == 0:
        raise SystemExit(f"Brak shardów data/{args.split}-*.tar w {args.repo}.")
    imgs = sum(1 for p in (out / args.split).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Gotowe: {n} shard(ów), {imgs} obrazów w {out / args.split}")


if __name__ == "__main__":
    main()
