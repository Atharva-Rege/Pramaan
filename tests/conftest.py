from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


@pytest.fixture(scope="session")
def synth_home(tmp_path_factory: pytest.TempPathFactory):
    from synth import build_synth_corpus

    from pramaan.core.paths import Paths, set_paths

    home = tmp_path_factory.mktemp("pramaan_home")
    info = build_synth_corpus(home, n=10)
    set_paths(Paths.for_root(home))
    yield home, info
    set_paths(None)


@pytest.fixture(scope="session")
def ingested(synth_home):
    from pramaan.data.ingest import ingest
    from pramaan.data.store import Store

    home, info = synth_home
    summary = ingest(workers=1, echo=lambda *_: None)
    store = Store()
    yield home, info, store, summary
    store.close()


@pytest.fixture(scope="session")
def built(ingested):
    from pramaan.gym.build import build_gym, cache_evidence

    home, info, store, summary = ingested
    cache_evidence(store, lambda *_: None)
    gym = build_gym(store, {"train": 40, "dev": 40, "test": 20}, lambda *_: None)
    return home, info, store, summary, gym
