import numpy as np
import torch
from PIL import Image

from src.near_duplicates import DIHEDRAL, UnionFind, best_pair_similarity, dihedral, spatial_features


def test_dihedral_has_eight_distinct_transforms_and_preserves_content():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    outs = [dihedral(x, k, f) for k, f in DIHEDRAL]
    assert len(outs) == 8
    assert len({o[0].flatten().tolist().__str__() for o in outs}) == 8
    assert all(torch.equal(o.sum(), x.sum()) for o in outs)


def test_union_find_groups_transitively():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    assert uf.find(0) == uf.find(2) and uf.find(4) == uf.find(5)
    assert uf.find(0) != uf.find(3) and uf.find(0) != uf.find(4)


def test_flipped_copy_is_far_more_similar_than_a_different_image(tmp_path):
    rng = np.random.default_rng(0)
    base = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    base[20:60, 30:70] = [30, 140, 40]                      # a "leaf-like" blob
    other = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    other[50:90, 5:40] = [200, 60, 20]
    paths = []
    for name, arr in (("a.png", base), ("a_flip.png", base[:, ::-1]), ("b.png", other)):
        Image.fromarray(np.ascontiguousarray(arr)).save(tmp_path / name)
        paths.append(name)
    F = spatial_features(paths, data_root=str(tmp_path), dev=torch.device("cpu"))
    sims = best_pair_similarity(F, np.arange(3))
    assert sims[0, 1] > 0.9 > sims[0, 2]
    assert sims[0, 1] > sims[0, 2] + 0.2
