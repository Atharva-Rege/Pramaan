from types import SimpleNamespace

import numpy as np

from pramaan.evals import baselines


def test_embeddings_are_cached_and_invalidated_by_content(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "get_paths", lambda: SimpleNamespace(indexes=tmp_path))
    calls: list[int] = []

    def fake_embed(texts):
        calls.append(len(texts))
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)

    monkeypatch.setattr(baselines, "embed_texts", fake_embed)
    a = baselines.embed_chunks_cached(["alpha", "beta"])
    b = baselines.embed_chunks_cached(["alpha", "beta"])
    assert calls == [2]  # the second call came from disk
    assert np.array_equal(a, b)
    baselines.embed_chunks_cached(["alpha", "beta", "gamma"])
    assert calls == [2, 3]  # different chunk texts, so a new file
    baselines.embed_chunks_cached(["alpha", "BETA"])
    assert calls == [2, 3, 2]  # an edited chunk also invalidates
