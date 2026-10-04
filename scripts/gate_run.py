import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pramaan.agent.factory import build_agent, load_runtime
from pramaan.core.splits import purpose
from pramaan.evals.harness import load_questions, per_question_rows, run_episodes, summarize
from pramaan.improve.diagnose import diagnose
from pramaan.llm.cache import LLMStore
from pramaan.llm.client import LLMClient, set_client

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # local runs; in CI the key comes from a secret
p = argparse.ArgumentParser()
p.add_argument("--qids", required=True)
p.add_argument("--out", required=True)
p.add_argument("--cap", type=float, default=0.5)
p.add_argument("--cache", default=None, help="LLM cache file; a fresh path measures the cold cost")
a = p.parse_args()

client = LLMClient(store=LLMStore(Path(a.cache) if a.cache else None))
client.guard.run_cap = a.cap
set_client(client)
rt = load_runtime(client=client)
agent = build_agent(rt, "small_only")
qids = json.loads(Path(a.qids).read_text())
resolver = lambda n: (rt.corpus.resolve_company(n, limit=1) or [{}])[0].get("company_id")  # noqa: E731
with purpose("dev_eval"):
    qs = load_questions(rt.store, "dev", qids=qids)
    records = run_episodes(agent, qs, rt.store, "gate", workers=4, resolver=resolver, persist=False)
out = Path(a.out)
out.write_text(json.dumps({"summary": summarize(records), "per_question": per_question_rows(records)}, indent=1, default=str))
diagnostics = {r.q.qid: " ".join(diagnose(r.q, r.ep, r.g, rt.store, rt.corpus).messages[:2]) for r in records if not r.g.verified_correct}
out.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
errors = [r.q.qid for r in records if r.ep.error]
n_ok = sum(r.g.verified_correct for r in records)
print(f"gate run: {n_ok}/{len(records)} verified correct, cost ${sum(r.ep.cost_usd for r in records):.3f}, agent errors {len(errors)}")
if errors:
    raise SystemExit(f"{len(errors)} episodes raised an agent error (for example a missing API key): {records[0].ep.error if records else ''}")
