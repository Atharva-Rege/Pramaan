import threading
import time
from types import SimpleNamespace

from pramaan.agent import tools as T


def _toolbox():
    tb = object.__new__(T.ToolBox)
    tb.internal_calls = 0
    tb.as_of = None
    tb.reader = "llm"
    tb.corpus = SimpleNamespace(page=lambda pid, as_of: {}, store=object())
    tb.modules = SimpleNamespace(fingerprint=lambda: "fp")
    tb.defense = SimpleNamespace(spotlight=False)
    return tb


def test_concurrent_reads_of_one_page_pay_once(monkeypatch):
    T.clear_table_cache()
    reads: list[str] = []

    def slow_read(self, page_id, page, key, note):
        with T._TABLE_LOCK:
            hit = T._TABLE_CACHE.get(key)
        if hit is not None:
            return hit
        reads.append(page_id)
        time.sleep(0.1)  # long enough for every other thread to arrive while the read is in flight
        with T._TABLE_LOCK:
            T._TABLE_CACHE[key] = "TABLE"
        return "TABLE"

    monkeypatch.setattr(T.ToolBox, "_read_table_locked", slow_read)
    tb = _toolbox()
    out: list[object] = []
    threads = [threading.Thread(target=lambda: out.append(tb.read_table("p1"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reads == ["p1"]
    assert out == ["TABLE"] * 8
    T.clear_table_cache()


def test_different_pages_are_not_serialised(monkeypatch):
    T.clear_table_cache()
    started: list[str] = []
    both_started = threading.Event()

    def read(self, page_id, page, key, note):
        started.append(page_id)
        if len(started) == 2:
            both_started.set()
        # each read waits for the other, which only works if the two reads run at the same time
        assert both_started.wait(timeout=2)
        return page_id

    monkeypatch.setattr(T.ToolBox, "_read_table_locked", read)
    tb = _toolbox()
    ths = [threading.Thread(target=lambda p=p: tb.read_table(p)) for p in ("p1", "p2")]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert sorted(started) == ["p1", "p2"]
    T.clear_table_cache()
