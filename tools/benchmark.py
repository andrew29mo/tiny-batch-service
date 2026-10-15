"""Record all outcomes of the same arrival traces in alternating fresh server trials."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_model import prepare, digest
from tiny_batch_service.experiment import fingerprint, host_metadata, make_trace, source_hash, summarize, trial
from tiny_batch_service.report import LIMITS, overview, write_report


async def run(args):
    fixture = ROOT / "examples/generated/digits"
    manifest = prepare(fixture)
    import numpy as np
    with np.load(fixture / "fixture.npz", allow_pickle=False) as archive:
        retained = {name: {"dtype": str(archive[name].dtype), "shape": list(archive[name].shape),
                           "data": archive[name].tolist()} for name in archive.files}
    (args.output / "fixture.json").write_text(json.dumps(retained, separators=(",", ":"), allow_nan=False) + "\n")
    report = {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hash(), "host": host_metadata(), "model": manifest,
              "policy": {"capacity": 64, "batch_size": 16, "wait_ms": 2., "request_timeout_ms": 250,
                         "generator_max_inflight": 512, "arrival_p99_budget_ms": 25, "rss_interval_ms": 5},
              "measurement_scope": LIMITS, "repeats": args.repeats, "cases": [], "execution_order": []}
    for rate in args.rates:
        trace = make_trace(args.count, rate)
        case = {"rate": rate, "trace": trace, "trace_sha256": fingerprint(trace), "runs": []}
        report["cases"].append(case)
        for repeat in range(args.repeats):
            for mode in (("single", "dynamic") if repeat % 2 == 0 else ("dynamic", "single")):
                settings = {"batch_size": 1 if mode == "single" else 16,
                            "wait_ms": 0. if mode == "single" else 2., "capacity": 64, "audit_limit": 4096}
                print(f"Measuring offered={rate}/s trial={repeat+1} mode={mode}", flush=True)
                result = await trial(fixture / "model.onnx", fixture / "fixture.npz", trace, settings)
                result.update(mode=mode, trial=repeat)
                result["summary"] = summarize(result, trace)
                case["runs"].append(result)
                report["execution_order"].append({"rate": rate, "trial": repeat, "mode": mode})
                # Preserve partial evidence even if a later trial fails; never fabricate missing runs.
                (args.output / "partial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_report(report, args.output)
    (args.output / "README.md").write_text(overview(report, os.path.relpath(ROOT / "docs/METHODOLOGY.md", args.output)))
    (args.output / "partial.json").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/experiment")
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--rates", type=float, nargs="+", default=[100., 1000., 3000.])
    args = parser.parse_args()
    if not 1 <= args.count <= 3000 or args.repeats < 1 or any(not 0 < r <= 10000 for r in args.rates):
        parser.error("count must be 1..3000, repeats positive, rates finite and 0..10000")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing results are not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))
