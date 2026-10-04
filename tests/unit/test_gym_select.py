from types import SimpleNamespace

import pytest

from pramaan.gym.generate import select_questions

MIX = {"T1": 0.5, "T2": 0.25, "T3": 0.25}


def _cand(split: str, tier: str, filing: str, i: int) -> SimpleNamespace:
    return SimpleNamespace(split=split, tier=tier, filing_id=filing, qid=f"{tier}-{filing}-{i}")


@pytest.mark.parametrize("seed", range(10))
def test_scarce_tiers_keep_their_quota_when_filing_cap_binds(seed):
    # 4 filings x cap 2 = 8 slots; the quotas ask for 4 / 2 / 2 of 8. Abundant T1 candidates
    # exist on every filing and must not crowd out the few T3 candidates.
    cands = [_cand("dev", "T1", f"f{f}", i) for f in range(4) for i in range(10)]
    cands += [_cand("dev", "T3", f"f{f}", 0) for f in range(2)]
    picks = select_questions(cands, {"dev": 8}, MIX, seed=seed, max_per_filing={"dev": 2})
    tiers = [c.tier for c in picks]
    assert tiers.count("T3") == 2
    assert len(picks) == 8


def test_cap_per_filing_is_respected():
    cands = [_cand("dev", "T1", "f0", i) for i in range(20)]
    picks = select_questions(cands, {"dev": 10}, MIX, seed=1, max_per_filing={"dev": 3})
    assert len(picks) == 3


@pytest.mark.parametrize("seed", range(10))
def test_mix_is_preserved_when_capacity_binds(seed):
    # 3 filings x cap 4 = 12 slots for a request of 24; every tier has plenty of candidates.
    # The mix must stay 50/25/25 instead of the first tier taking everything.
    cands = [_cand("dev", t, f"f{f}", i) for t in MIX for f in range(3) for i in range(8)]
    picks = select_questions(cands, {"dev": 24}, MIX, seed=seed, max_per_filing={"dev": 4})
    tiers = [c.tier for c in picks]
    assert len(picks) == 12
    assert tiers.count("T1") == 6 and tiers.count("T2") == 3 and tiers.count("T3") == 3
