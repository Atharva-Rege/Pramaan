from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pramaan: a verifiable research agent for Indian company filings")
gym_app = typer.Typer(no_args_is_help=True, help="question gym")
improve_app = typer.Typer(no_args_is_help=True, help="reranker training data")
app.add_typer(gym_app, name="gym")
app.add_typer(improve_app, name="improve")
demo_app = typer.Typer(no_args_is_help=True, help="cached demo episodes")
app.add_typer(demo_app, name="demo")
console = Console()


def echo(msg: str) -> None:
    console.print(msg, highlight=False)


def _offline_env() -> None:
    for role in ("SMALL", "STRONG", "REFLECTION"):
        os.environ[f"PRAMAAN_MODEL_{role}"] = "offline/rules"


def _runtime(llm: str, index: str = "main", build: bool = False, read_only: bool = False):
    from pramaan.agent.factory import load_runtime
    from pramaan.llm.cache import LLMStore
    from pramaan.llm.client import LLMClient, set_client

    if llm == "offline":
        _offline_env()
    client = LLMClient(store=LLMStore(), enforce_budget=llm != "offline")
    set_client(client)
    return load_runtime(index, build_if_missing=build, client=client, read_only=read_only)


@app.command()
def collect(
    profile: str = typer.Option(None, help="gym profile (configs/gym.yaml)"),
    others: int = typer.Option(None, help="number of non Nifty 200 companies"),
    nifty_limit: int = typer.Option(None),
    skip_nifty: bool = typer.Option(False),
    reference_pdf: bool = typer.Option(False, help="also fetch previous quarter PDFs for Nifty 200"),
) -> None:
    from pramaan.data.collect_run import run_collection, setup_logging

    setup_logging("collect")
    echo(json.dumps(run_collection(profile, nifty=not skip_nifty, others=others, nifty_limit=nifty_limit, reference_pdf_nifty=reference_pdf, echo=echo), indent=1, default=str))


@app.command("restore-pdfs")
def restore_pdfs_cmd() -> None:
    from pramaan.data.collector import restore_pdfs

    echo(json.dumps(restore_pdfs(echo=echo), indent=1))


@app.command()
def ingest(
    workers: int = typer.Option(8),
    limit: int = typer.Option(None),
    force: bool = typer.Option(False),
    ocr: str = typer.Option("none", help="none | auto | tesseract | doctr"),
    profile: str = typer.Option(None),
    export_scanned: Path = typer.Option(None, help="write page images that need OCR to this folder and zip them"),
    import_ocr: Path = typer.Option(None, help="zip produced by notebook 02"),
) -> None:
    from pramaan.data.ingest import export_scanned_pages_cmd, import_ocr_cmd
    from pramaan.data.ingest import ingest as run

    if export_scanned:
        echo(json.dumps(export_scanned_pages_cmd(export_scanned), indent=1))
        return
    if import_ocr:
        echo(json.dumps(import_ocr_cmd(import_ocr), indent=1))
        return
    echo(json.dumps(run(workers, limit, force, ocr, profile, echo=echo), indent=1, default=str))


@app.command()
def index(name: str = "main", no_rows: bool = False) -> None:
    from pramaan.data.store import Store
    from pramaan.retrieval.index import RetrievalIndex

    store = Store()
    idx = RetrievalIndex.build(store, row_docs=not no_rows)
    root = idx.save(name)
    echo(f"index saved to {root}: {len(idx.pages.docs)} pages, {len(idx.rows.docs) if idx.rows else 0} rows")


@gym_app.command("build")
def gym_build(profile: str = typer.Option(None), train: int = typer.Option(None), dev: int = typer.Option(None), test: int = typer.Option(None), evidence: bool = True) -> None:
    from pramaan.core.config import gym_config
    from pramaan.data.store import Store
    from pramaan.gym.build import build_gym, cache_evidence

    if profile:
        os.environ["PRAMAAN_GYM_PROFILE"] = profile
    store = Store()
    if evidence:
        cache_evidence(store, echo)
    sizes = dict(gym_config().profile.sizes)
    for k, v in (("train", train), ("dev", dev), ("test", test)):
        if v is not None:
            sizes[k] = v
    echo(json.dumps(build_gym(store, sizes, echo), indent=1, default=str))


@gym_app.command("stats")
def gym_stats_cmd() -> None:
    from pramaan.data.store import Store
    from pramaan.gym.build import gym_stats

    echo(json.dumps(gym_stats(Store()), indent=1, default=str))


@gym_app.command("release")
def gym_release(out: Path = Path("release/gym")) -> None:
    from pramaan.gym.release import release_gym

    echo(json.dumps(release_gym(out), indent=1, default=str))


@app.command("eval")
def eval_cmd(
    config: str = typer.Option(..., "--config"),
    split: str = typer.Option("dev", "--split"),
    subset: int = typer.Option(None, "--subset"),
    workers: int = typer.Option(4),
    llm: str = typer.Option("api", help="api | offline"),
    stages: str = typer.Option(None, help="retrieval stage preset"),
    yes: bool = typer.Option(False, "--yes"),
    tag: str = typer.Option(""),
) -> None:
    from pramaan.core.errors import NeedsConfirmation
    from pramaan.evals.cost import estimate, require_confirmation
    from pramaan.evals.experiments import e3_compare

    rt = _runtime(llm)
    if llm == "api":
        est = estimate(config, rt.store, split, subset)
        echo(f"estimated cost: ${est['usd']:.2f} for {est['n_questions']} questions")
        try:
            require_confirmation(est["usd"], yes)
        except NeedsConfirmation as exc:
            echo(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
    baselines: dict[str, Any] = {}
    if config in ("B0_naive_rag", "B0_strong"):
        from pramaan.evals.baselines import NaiveRAG

        baselines[config] = {
            "B0_naive_rag": lambda: NaiveRAG(rt.corpus, rt.client, "small", "B0_naive_rag"),
            "B0_strong": lambda: NaiveRAG(rt.corpus, rt.client, "strong", "B0_strong"),
        }[config]()
    if stages:
        os.environ["PRAMAAN_STAGES"] = stages
    summ = e3_compare(rt, [config], split, subset, workers, baselines or None, llm, echo, None, None, tag)
    row = summ["table"][config]
    echo(json.dumps({k: row[k] for k in ("n", "accuracy", "verified_accuracy", "abstain_rate", "cost_per_question_usd", "latency_ms_p50", "latency_ms_p95")}, indent=1, default=str))


@app.command()
def exp(
    name: str = typer.Argument(..., help="E1 | E2 | E3 | E4 | E6 | E8"),
    split: str = typer.Option("dev"),
    subset: int = typer.Option(None),
    config: str = typer.Option("cascade"),
    configs: str = typer.Option(None, help="comma separated, for E3"),
    run_id: str = typer.Option(None, help="episode run id for E4 and E6"),
    run_id_after: str = typer.Option(None),
    finetuned: str = typer.Option(None, help="fine-tuned cross-encoder path for E2"),
    workers: int = typer.Option(4),
    llm: str = typer.Option("api"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    from pramaan.evals import experiments as X

    n = name.upper()
    if n == "E1":
        from pramaan.data.store import Store

        X.e1_gym_validity(Store(read_only=True), echo)
    elif n == "E8":
        from pramaan.data.store import Store

        X.e8_season(Store(read_only=True), None, echo)
    elif n == "E2":
        rt = _runtime("offline", read_only=True)
        X.e2_retrieval(rt, split, subset or 300, finetuned, echo)
    elif n == "E3":
        from pramaan.evals.baselines import NaiveRAG
        from pramaan.evals.cost import estimate, require_confirmation

        rt = _runtime(llm)
        names = (configs or "B0_naive_rag,small_only,strong_only,cascade").split(",")
        if llm == "api":
            total = sum(estimate(c, rt.store, split, subset)["usd"] for c in names)
            echo(f"estimated total cost: ${total:.2f}")
            require_confirmation(total, yes)
        base: dict[str, Any] = {}
        for c in names:
            if c == "B0_naive_rag":
                base[c] = NaiveRAG(rt.corpus, rt.client, "small", c)
            elif c == "B0_strong":
                base[c] = NaiveRAG(rt.corpus, rt.client, "strong", c)
        X.e3_compare(rt, names, split, subset, workers, base or None, llm, echo)
    elif n == "E4":
        rt = _runtime(llm)
        X.e4_calibration(rt, run_id or "", config, split, echo=echo)
    elif n == "E6":
        rt = _runtime(llm)
        X.e6_attribution(rt, config, run_id or "", run_id_after, split, echo=echo, llm_mode=llm)
    else:
        raise typer.BadParameter("unknown experiment")


@improve_app.command("export-reranker-data")
def improve_export() -> None:
    from pramaan.data.store import Store
    from pramaan.improve.reranker_data import export_reranker_data

    echo(json.dumps(export_reranker_data(Store()), indent=1))


@app.command()
def attrib(config: str = "cascade", run_id: str = typer.Option(...), run_id_after: str = typer.Option(None), split: str = "dev", llm: str = "api") -> None:
    from pramaan.evals.experiments import e6_attribution

    rt = _runtime(llm)
    echo(json.dumps(e6_attribution(rt, config, run_id, run_id_after, split, echo=echo, llm_mode=llm)["before"], indent=1))


@app.command()
def calibrate(run_id: str = typer.Option(...), config: str = "cascade", split: str = "dev") -> None:
    from pramaan.evals.experiments import e4_calibration

    rt = _runtime("offline")
    echo(json.dumps(e4_calibration(rt, run_id, config, split, echo=echo), indent=1, default=str)[:2500])


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, llm: str = "api", demo_only: bool = False) -> None:
    import uvicorn

    os.environ["PRAMAAN_LLM"] = llm
    if demo_only:
        os.environ["PRAMAAN_DEMO_ONLY"] = "1"
    uvicorn.run("pramaan.serving.api:create_app", host=host, port=port, factory=True)


@app.command()
def mcp(llm: str = "api", transport: str = "stdio") -> None:
    os.environ["PRAMAAN_LLM"] = llm
    from pramaan.mcp_server.server import main

    main(transport)


@app.command()
def stats() -> None:
    from pramaan.data.store import Store
    from pramaan.llm.cache import LLMStore

    store = Store(read_only=True)
    table = Table(title="Pramaan")
    table.add_column("metric")
    table.add_column("value")
    for t in ("companies", "filings", "pages", "xbrl_facts", "questions", "runs", "episodes", "memories"):
        table.add_row(t, str(store.count(t)))
    ls = LLMStore().stats()
    for k, v in ls.items():
        table.add_row(f"llm {k}", f"{v:.2f}" if isinstance(v, float) else str(v))
    rows = store.q("SELECT run_id, name, split, cost_usd, started_at FROM runs ORDER BY started_at DESC LIMIT 8")
    console.print(table)
    for r in rows:
        echo(f"  run {r['run_id']:44s} {r['name']:18s} {r['split']:5s} ${r['cost_usd']:.3f}")


@app.command()
def doctor(llm: bool = typer.Option(False, help="make tiny real LLM calls for every role")) -> None:
    from pramaan.doctor import run_doctor

    ok = run_doctor(llm, echo)
    raise typer.Exit(0 if ok else 1)


@demo_app.command("build")
def demo_build(config: str = "cascade", n: int = 20, llm: str = "api", from_run: str = typer.Option(None, help="build from a stored run, no API calls")) -> None:
    from pramaan.serving.demo import build_demo, build_demo_from_run

    if from_run:
        from pramaan.data.store import Store

        echo(json.dumps(build_demo_from_run(Store(read_only=True), from_run, n, echo), indent=1))
        return
    rt = _runtime(llm)
    echo(json.dumps(build_demo(rt, config, n, echo), indent=1))


@app.command()
def render() -> None:
    from pramaan.render import render_all

    echo(json.dumps(render_all(echo), indent=1, default=str))


@app.command()
def snapshot(name: str = "main", demo: bool = typer.Option(False, help="small bundle for a public demo host: results and cited page renders only")) -> None:
    from pramaan.snapshot import make_demo_snapshot, make_snapshot

    echo(str(make_demo_snapshot(echo) if demo else make_snapshot(name, echo)))


if __name__ == "__main__":
    app()
