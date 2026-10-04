import json
from types import SimpleNamespace

from pramaan import render


def _write(root, name, config, tag):
    d = root / "E3"
    d.mkdir(exist_ok=True)
    (d / f"{name}_dev.json").write_text(json.dumps({"meta": {"config": config, "tag": tag}, "summary": {}}))


def test_tagged_run_does_not_replace_the_untagged_run(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "get_paths", lambda: SimpleNamespace(results=tmp_path))
    _write(tmp_path, "rules_only", "rules_only", "")
    _write(tmp_path, "rules_onlyft_reranker", "rules_only", "ft_reranker")
    rows = render._all("E3", "dev")
    assert set(rows) == {"rules_only", "rules_only [ft_reranker]"}


def test_files_without_a_summary_are_not_leaderboard_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "get_paths", lambda: SimpleNamespace(results=tmp_path))
    _write(tmp_path, "small_only", "small_only", "")
    (tmp_path / "E3" / "verifier_audit_dev.json").write_text(json.dumps({"meta": {"config": "verifier_audit"}, "audit": {}}))
    assert set(render._all("E3", "dev")) == {"small_only"}
