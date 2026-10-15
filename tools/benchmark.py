"""Record all outcomes of the same arrival traces in alternating fresh server trials."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import gzip
import hashlib
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_model import prepare, digest
from tiny_batch_service.experiment import fingerprint, host_metadata, make_trace, source_hash, summarize, trial
from tiny_batch_service.report import LIMITS, comparison, overview, write_report


async def run(args):
    fixture = ROOT / "examples/generated" / args.profile
    manifest = prepare(fixture, args.profile)
    import numpy as np
    with np.load(fixture / "fixture.npz", allow_pickle=False) as archive:
        retained = {name: {"dtype": str(archive[name].dtype), "shape": list(archive[name].shape),
                           "data": archive[name].tolist()} for name in archive.files}
    (args.output / "fixture.json").write_text(json.dumps(retained, separators=(",", ":"), allow_nan=False) + "\n")
    audit_limit = 16384 if args.profile == "heavy" else 4096
    report = {"schema_version": 4, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hash(), "host": host_metadata(), "model": manifest,
              "policy": {"capacity": 64, "batch_size": 16, "wait_ms": 2., "request_timeout_ms": 250,
                         "generator_max_inflight": 512, "arrival_p99_budget_ms": 25, "rss_interval_ms": 5, "audit_limit": audit_limit},
              "measurement_scope": LIMITS, "repeats": args.repeats, "cases": [], "execution_order": []}
    if args.profile == "heavy":
        selection = json.loads((ROOT / "examples/heavy-digits/flow-selection.json").read_text())
        calibration = ROOT / "results/heavy-macos-arm64/calibration.json.gz"
        if hashlib.sha256(calibration.read_bytes()).hexdigest() != selection["calibration_sha256"]:
            raise ValueError("Calibration provenance differs")
        report["flow_selection"] = selection
        report["selected_grid_applied"] = (args.rates == selection["rates"] and args.count == selection["count"]
                                            and args.repeats == selection["repeats"])
        if calibration.resolve() != (args.output / "calibration.json.gz").resolve():
            shutil.copyfile(calibration, args.output / "calibration.json.gz")
    for case_index, rate in enumerate(args.rates):
        trace = make_trace(args.count, rate)
        case = {"rate": rate, "trace": trace, "trace_sha256": fingerprint(trace), "runs": []}
        report["cases"].append(case)
        for repeat in range(args.repeats):
            for mode in (("single", "dynamic") if repeat % 2 == 0 else ("dynamic", "single")):
                settings = {"batch_size": 1 if mode == "single" else 16,
                            "wait_ms": 0. if mode == "single" else 2., "capacity": 64, "audit_limit": audit_limit}
                print(f"Measuring offered={rate}/s trial={repeat+1} mode={mode}", flush=True)
                result = await trial(fixture / "model.onnx", fixture / "fixture.npz", trace, settings)
                result.update(mode=mode, trial=repeat)
                result["summary"] = summarize(result, trace)
                if args.profile == "heavy":
                    filename = f"raw/case-{case_index}-trial-{repeat}-{mode}.json.gz"
                    payload = gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
                    (args.output / "raw").mkdir(exist_ok=True)
                    (args.output / filename).write_bytes(payload)
                    stub = {k: result[k] for k in ("mode", "trial", "settings", "environment", "trace_sha256", "summary", "workload")}
                    stub.update(evidence_file=filename, evidence_sha256=hashlib.sha256(payload).hexdigest())
                    case["runs"].append(stub)
                else:
                    case["runs"].append(result)
                report["execution_order"].append({"rate": rate, "trial": repeat, "mode": mode})
                # Preserve partial evidence even if a later trial fails; never fabricate missing runs.
                (args.output / "partial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        case["comparison"] = comparison(case, args.repeats)
    write_report(report, args.output)
    (args.output / "README.md").write_text(overview(report, os.path.relpath(ROOT / "docs/METHODOLOGY.md", args.output)))
    (args.output / "partial.json").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("tiny", "heavy"), default="tiny")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/experiment")
    parser.add_argument("--count", type=int)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--rates", type=float, nargs="+")
    args = parser.parse_args()
    selection = json.loads((ROOT / "examples/heavy-digits/flow-selection.json").read_text()) if args.profile == "heavy" else {"count": 480, "rates": [100., 1000., 3000.]}
    args.count = args.count if args.count is not None else selection["count"]
    args.rates = args.rates if args.rates is not None else selection["rates"]
    if not 1 <= args.count <= (12000 if args.profile == "heavy" else 3000) or args.repeats < 1 or any(not 0 < r <= 20000 for r in args.rates):
        parser.error("count must be 1..3000 (heavy: 12000), repeats positive, rates finite and 0..20000")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing results are not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))
