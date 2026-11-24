import numpy as np

from metalscope import experts as X
from tests.helpers import E, K, make_trace, random_ids


def test_corpus_dedupes_contexts_and_skips_padding():
    rng = np.random.default_rng(0)
    texts = ["a", " b", " c", " d", " e", " f"]
    ids = random_ids(rng, 2, 6)
    t1 = make_trace(texts, ids)
    t2 = make_trace(texts, ids)                                   # exact repeat: no new evidence
    ids3 = ids.copy()
    ids3[:, 4:] = random_ids(rng, 2, 2)
    t3 = make_trace(texts[:4] + [" x", " y"], ids3)               # shares a 4-token prefix
    ids_pad = ids.copy()
    ids_pad[:, :2] = -1                                           # right-aligned padding
    t4 = make_trace(["p", " q"] + texts[2:], ids_pad)
    c = X.build_corpus([t1, t2, t3, t4])
    assert c.n == 6 + 2 + 4
    assert c.total_positions == 6 + 6 + 6 + 4
    assert (c.ids >= 0).all()
    assert X.build_corpus([t1, t2], dedupe=False).n == 12


def test_coactivation_embedding_separates_groups():
    rng = np.random.default_rng(2)
    N = 300
    # Experts 0-7 fire together on half the tokens, 8-15 on the other half; 31 almost never.
    ids = np.array([rng.choice(8, K, replace=False) + (0 if t % 2 else 8) for t in range(N)])
    ids[0, 0] = 31
    for method in ("mds", "tsne"):
        xy, ev = X.coactivation_embedding(ids, E, min_uses=3, method=method)
        assert np.isnan(xy[31]).all() and np.isnan(xy[20]).all()
        if method == "mds":
            g1, g2 = xy[:8].mean(0), xy[8:16].mean(0)
            spread = max(xy[:8].std(0).max(), xy[8:16].std(0).max())
            assert np.linalg.norm(g1 - g2) > 2 * spread, method
        else:
            # t-SNE's layout differs across machines (float differences grow over its iterations: GitHub's runner
            # spreads the clusters wide where a local Mac packs them), so check the grouping, not the shape: most
            # experts have their nearest neighbour in their own group.
            pts = xy[:16]
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(16) * 1e9
            own = sum((d[i].argmin() < 8) == (i < 8) for i in range(16))
            assert own >= 14, (method, own)
