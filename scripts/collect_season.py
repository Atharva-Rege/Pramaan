import argparse
import json

from pramaan.data.collect_run import run_collection, setup_logging

p = argparse.ArgumentParser()
p.add_argument("--profile", default=None)
p.add_argument("--others", type=int, default=None)
p.add_argument("--nifty-limit", type=int, default=None)
p.add_argument("--skip-nifty", action="store_true")
p.add_argument("--reference-pdf", action="store_true")
a = p.parse_args()
setup_logging("collect")
print(json.dumps(run_collection(a.profile, nifty=not a.skip_nifty, others=a.others, nifty_limit=a.nifty_limit, reference_pdf_nifty=a.reference_pdf), indent=1))
