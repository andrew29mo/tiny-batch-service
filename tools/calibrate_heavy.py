"""Exploratory rate sweep only. Preserve every observation; do not use as final comparison."""
import asyncio
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from make_model import prepare, ROOT
from tiny_batch_service.experiment import trial, make_trace, summarize, source_hash, host_metadata


async def run():
    destination = ROOT / "results/local/heavy-calibration"
    destination.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / "examples/generated/heavy"
    model = prepare(fixture, "heavy")
    result = {"purpose": "Exploratory flow selection, not the repeated final experiment", "model": model,
              "generated_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hash(),
              "host": host_metadata(), "cases": []}
    for rate in (500., 1500., 3000., 6000., 10000., 15000., 20000.):
        trace = make_trace(1500, rate)
        case = {"rate": rate, "trace": trace, "runs": []}
        for mode in ("single", "dynamic"):
            settings = {"batch_size": 1 if mode == "single" else 16,
                        "wait_ms": 0. if mode == "single" else 2., "capacity": 64, "audit_limit": 4096}
            data = await trial(fixture / "model.onnx", fixture / "fixture.npz", trace, settings)
            data.update(mode=mode)
            data["summary"] = summarize(data, trace)
            case["runs"].append(data)
            print(rate, mode, json.dumps(data["summary"], separators=(",", ":")), flush=True)
        result["cases"].append(case)
        (destination / "calibration.json.gz").write_bytes(gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    asyncio.run(run())
