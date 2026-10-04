from __future__ import annotations

import logging
from typing import Any

from pramaan.core.config import gym_config
from pramaan.core.paths import get_paths
from pramaan.data.collector import Collector, previous_quarter_end, prior_year_end


def setup_logging(name: str) -> None:
    paths = get_paths().ensure()
    handler = logging.FileHandler(paths.logs / f"{name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def run_collection(
    profile: str | None = None,
    nifty: bool = True,
    others: int | None = None,
    nifty_limit: int | None = None,
    reference_pdf_nifty: bool = True,
    reference_pdf_others: bool = False,
    echo: Any = print,
) -> dict[str, Any]:
    gcfg = gym_config(profile)
    target = gcfg.profile.target_period_end
    c = Collector()
    uni = c.universe(other_n=others, nifty_limit=nifty_limit)
    nifty_cos = [u for u in uni if u.in_nifty200]
    other_cos = [u for u in uni if not u.in_nifty200]
    summary: dict[str, Any] = {"target": target.isoformat()}

    def progress(sym: str, stats: dict[str, int]) -> None:
        echo(f"{sym:14s} {stats}")

    if nifty:
        echo(f"nifty200 phase: {len(nifty_cos)} companies")
        summary["nifty"] = c.run(
            target,
            prior_year_end(target),
            previous_quarter_end(target),
            nifty_cos,
            want_filers=10**6,
            reference_pdf=reference_pdf_nifty,
            progress=progress,
        )
    n_others = others if others is not None else int(c.cfg["universe"]["other_sample_size"])
    if n_others > 0:
        echo(f"others phase: want {n_others} filers from {len(other_cos)} candidates")
        summary["others"] = c.run(
            target,
            prior_year_end(target),
            previous_quarter_end(target),
            other_cos,
            want_filers=n_others,
            reference_pdf=reference_pdf_others,
            progress=progress,
        )
    return summary
