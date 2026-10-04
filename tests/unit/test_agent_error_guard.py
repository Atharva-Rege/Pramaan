from types import SimpleNamespace

from pramaan.evals.harness import agent_errors, error_summary


def _rec(error):
    return SimpleNamespace(ep=SimpleNamespace(error=error))


def test_clean_records_have_no_errors():
    recs = [_rec(""), _rec(None)]
    assert agent_errors(recs) == []
    assert error_summary(recs) == ""


def test_any_agent_error_is_reported():
    recs = [_rec(""), _rec("BudgetExceeded: per_run_usd_cap 8 reached for run x"), _rec("")]
    assert len(agent_errors(recs)) == 1
    assert "1 of 3 episodes" in error_summary(recs) and "BudgetExceeded" in error_summary(recs)
