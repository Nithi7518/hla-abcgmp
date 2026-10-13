"""Transform-robust near-duplicate grouping (Phase 3b).

Usage:
    python -m src.near_duplicates                # writes data/manifests/robust_groups.csv
    python -m src.near_duplicates --threshold 0.75

Why: the Phase 3 detector (exact bytes + perceptual hashes) misses augmented copies of the
same photograph -- flipped, rotated or cropped variants, common in the PlantVillage-derived
apple folders -- so they could land in both train and test. Here every image is described by
a spatial feature map (ImageNet ResNet18 layer3, 7x7 grid, per-channel mean removed) and
compared, within a class, against all 8 rotations/flips of the other image. Pairs at or above
the cosine threshold are linked; connected components (merged with the existing exact/perceptual
duplicate groups) become the splitting unit.

Output: data/manifests/robust_groups.csv (image_path, group_id) and
results/near_duplicate_report.json. Groups never mix labels' images across folders because the
comparison is done within a label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from src.assistant.model_io import resolve_device
from src.preprocessing import build_image_transforms

DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]


def dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


@torch.no_grad()
def spatial_features(paths: list[str], data_root: str = "data/raw", dev=None) -> np.ndarray:
    """(8, N, D) L2-normalised spatial descriptors, one slab per dihedral transform."""
    dev = dev or resolve_device()
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval().to(dev)
    body = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    tf = build_image_transforms({"size": 224, "normalization": "imagenet"})["eval"]
    out = [[] for _ in DIHEDRAL]
    for s in range(0, len(paths), 64):
        x = torch.stack([tf(Image.open(Path(data_root) / p).convert("RGB")) for p in paths[s:s + 64]]).to(dev)
        for t, (k, flip) in enumerate(DIHEDRAL):
            f = torch.nn.functional.adaptive_avg_pool2d(body(dihedral(x, k, flip)), 7)
            f = f - f.mean(dim=(2, 3), keepdim=True)
            out[t].append(torch.nn.functional.normalize(f.flatten(1), dim=1).cpu())
    return np.stack([torch.cat(o).numpy().astype(np.float32) for o in out])


def best_pair_similarity(F: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """For images `rows`: (n, n) matrix of max over the 8 transforms of query-vs-original cosine."""
    base = F[0][rows]
    sims = np.max([F[t][rows] @ base.T for t in range(len(F))], axis=0)
    np.fill_diagonal(sims, -1.0)
    return sims


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--features-cache", default=None, help="optional .npy cache of the (8,N,D) features")
    args = ap.parse_args()

    m = pd.concat([pd.read_csv(f"data/manifests/{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    paths = m.image_path.tolist()
    if args.features_cache and Path(args.features_cache).exists():
        F = np.load(args.features_cache)
    else:
        F = spatial_features(paths)
        if args.features_cache:
            np.save(args.features_cache, F)

    uf = UnionFind(len(m))
    pos = {p: i for i, p in enumerate(paths)}
    dup = pd.read_csv("data/manifests/duplicate_groups.csv")
    for _, g in dup[dup.image_path.isin(pos)].groupby("group_id"):
        ids = [pos[p] for p in g.image_path]
        for a in ids[1:]:
            uf.union(ids[0], a)

    n_edges = 0
    for label, g in m.groupby("label"):
        rows = g.index.to_numpy()
        sims = best_pair_similarity(F, rows)
        ii, jj = np.where(np.triu(sims >= args.threshold, 1))
        for a, b in zip(ii, jj):
            uf.union(int(rows[a]), int(rows[b]))
        n_edges += len(ii)

    comp = np.array([uf.find(i) for i in range(len(m))])
    sizes = pd.Series(comp).value_counts()
    out = pd.DataFrame({"image_path": paths, "group_id": [f"rg{c}" for c in comp]})
    out.to_csv("data/manifests/robust_groups.csv", index=False)
    report = {
        "threshold": args.threshold, "n_images": len(m), "similar_pairs_linked": int(n_edges),
        "n_groups": int(len(sizes)), "groups_with_2plus_images": int((sizes > 1).sum()),
        "images_in_2plus_groups": int(sizes[sizes > 1].sum()), "largest_group": int(sizes.max()),
        "group_size_quantiles": {str(q): float(sizes.quantile(q)) for q in (0.5, 0.9, 0.99)},
    }
    Path("results/near_duplicate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
