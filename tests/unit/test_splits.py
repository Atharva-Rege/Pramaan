from datetime import date

import pytest

from pramaan.core.config import gym_config
from pramaan.core.errors import SplitAccessError
from pramaan.core.splits import assign_split, check_split_access, purpose


def test_assign_split_windows():
    p = gym_config("q2fy27_season").profile
    assert assign_split(date(2026, 10, 20), p) == "train"
    assert assign_split(date(2026, 10, 21), p) == "dev"
    assert assign_split(date(2026, 10, 31), p) == "dev"
    assert assign_split(date(2026, 11, 1), p) == "test"
    assert assign_split(None, p) is None


def test_test_split_is_guarded(tmp_path):
    from pramaan.core.paths import Paths, set_paths

    set_paths(Paths.for_root(tmp_path))
    try:
        with pytest.raises(SplitAccessError):
            check_split_access("test", "unit")
        with purpose("gepa"), pytest.raises(SplitAccessError):
            check_split_access("test", "unit")
        with purpose("final_eval"):
            check_split_access("test", "unit")
        check_split_access("dev", "unit")
        log = (tmp_path / "logs" / "split_access.jsonl").read_text().splitlines()
        assert len(log) == 3
    finally:
        set_paths(None)
