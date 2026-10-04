import pytest

from pramaan.core.errors import BudgetExceeded
from pramaan.llm.backends import ScriptedBackend
from pramaan.llm.cache import LLMStore
from pramaan.llm.client import LLMClient, call_scope


def make_client(tmp_path, fn=lambda r: "hello"):
    client = LLMClient(store=LLMStore(tmp_path / "c.sqlite"), backends={"scripted": ScriptedBackend(fn)}, enforce_budget=True)
    return client


def test_cache_hit_is_free_and_deterministic(tmp_path):
    client = make_client(tmp_path)
    kw = {"model": "scripted/m", "system": "s", "messages": [{"role": "user", "content": "q"}], "max_tokens": 50}
    with call_scope() as sc:
        a = client.complete(**kw)
        b = client.complete(**kw)
    assert a.text == b.text == "hello"
    assert not a.cache_hit and b.cache_hit
    assert sc.calls == 2 and sc.cache_hits == 1
    assert client.store.total_spend() == pytest.approx(a.cost_usd)


def test_key_changes_with_messages(tmp_path):
    seen = []
    client = make_client(tmp_path, lambda r: seen.append(1) or "x")
    for q in ("a", "b", "a"):
        client.complete(model="scripted/m", messages=[{"role": "user", "content": q}], max_tokens=10)
    assert len(seen) == 2


def test_budget_cap_blocks(tmp_path):
    client = make_client(tmp_path)
    client.guard.total_cap = 0.0
    with pytest.raises(BudgetExceeded):
        client.complete(model="scripted/m", messages=[{"role": "user", "content": "x"}], max_tokens=10)


def test_sampling_params_skipped_for_claude5():
    from pramaan.core.config import models_config

    mc = models_config()
    assert not mc.allows_sampling("anthropic/claude-sonnet-5-5")
    assert mc.allows_sampling("claude-haiku-4-5")
    assert mc.price("anthropic/claude-sonnet-5-5").input == 2.0


def test_spend_is_booked_to_the_scope_run_and_capped_per_run(tmp_path):
    # Every call used to be booked to "adhoc", so the per-run cap acted as a lifetime cap.
    client = make_client(tmp_path)
    client.guard.run_cap = 8.0
    client.store.record_spend("adhoc", "scripted/m", "", 10.0, 0, 0, False)
    with call_scope("run-b"):
        client.complete(model="scripted/m", messages=[{"role": "user", "content": "x"}], max_tokens=10)
    rows = client.store._con.execute("SELECT run_id FROM spend WHERE cache_hit = 0 ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["adhoc", "run-b"]
    with pytest.raises(BudgetExceeded):
        client.complete(model="scripted/m", messages=[{"role": "user", "content": "y"}], max_tokens=10)
