#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

URLS = {
    "hpo": "https://purl.obolibrary.org/obo/hp.obo",
    "do": "https://purl.obolibrary.org/obo/doid.obo",
    # MeSH availability varies by release. We try the OBO mirror first.
    "mesh": "https://purl.obolibrary.org/obo/mesh.obo",
}


def download(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for part in r.iter_content(chunk_size=chunk):
            if part:
                f.write(part)
                pbar.update(len(part))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hpo", "do"], choices=list(URLS.keys()))
    ap.add_argument("--out_dir", type=str, default="data/raw")
    ap.add_argument("--force", action="store_true", help="Redownload even if file exists.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for ds in args.datasets:
        url = URLS[ds]
        if ds in ["hpo", "do", "mesh"]:
            fname = url.split("/")[-1]
        else:
            fname = f"{ds}.dat"
        dest = out_dir / ds / fname
        if dest.exists() and not args.force:
            print(f"[skip] {ds}: {dest} exists")
            continue
        try:
            print(f"[download] {ds} <- {url}")
            download(url, dest)
        except Exception as e:
            print(f"[warning] Failed to download {ds} from {url}: {e}")
            print("If this persists (e.g., MeSH mirror moved), download the file manually and place it at:")
            print(f"  {dest}")
            raise


if __name__ == "__main__":
    main()
