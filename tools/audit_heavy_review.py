"""Audit the original committed experiment without selecting passing pairs."""
import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train_heavy_model import fit_model
from tiny_batch_service.experiment import summarize, validate_response

BASE = "633eec023aaf1600807433df7d5a20e3eaac439e"
PREFIX = "results/heavy-macos-arm64/"


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def evidence_audit():
    raw_index = original(PREFIX + "report.json")
    report = json.loads(raw_index)
    fixture = json.loads(original(PREFIX + "fixture.json"))
    reference = np.asarray(fixture["reference"]["data"], dtype=np.float32)
    cases, hashes = [], {}
    for case in report["cases"]:
        pairs = {}
        for stub in case["runs"]:
            payload = original(PREFIX + stub["evidence_file"])
            digest = hashlib.sha256(payload).hexdigest()
            if digest != stub["evidence_sha256"]:
                raise ValueError("Original raw evidence hash differs")
            hashes[stub["evidence_file"]] = digest
            run = json.loads(gzip.decompress(payload))
            recomputed = summarize(run, case["trace"])
            if recomputed != stub["summary"] or recomputed != run["summary"]:
                raise ValueError("Original summary differs from raw records")
            for row in run["client"]:
                if row["status"] == "success":
                    validate_response(row, reference[row["index"]])
            pairs.setdefault(run["trial"], {})[run["mode"]] = recomputed
        records = []
        for trial, pair in sorted(pairs.items()):
            a, b = pair["single"], pair["dynamic"]
            records.append({"trial": trial, "arrival_check_passed": a["arrival_fidelity_passed"] and b["arrival_fidelity_passed"],
                            "descriptive_throughput_ratio": b["success_per_second"] / a["success_per_second"],
                            "single_rejection_rate": a["rejection_rate"], "dynamic_rejection_rate": b["rejection_rate"],
                            "single_scheduled_success_p95_ms": a["scheduled_success_latency_ms"]["p95"],
                            "dynamic_scheduled_success_p95_ms": b["scheduled_success_latency_ms"]["p95"]})
        passed = sum(p["arrival_check_passed"] for p in records)
        cases.append({"rate": case["rate"], "pairs": records, "passing_pairs": passed, "total_pairs": len(records),
                      "all_pair_ratio_median": median(p["descriptive_throughput_ratio"] for p in records),
                      "verdict": "inconclusive_arrival_fidelity" if passed != len(records) else "descriptive_only",
                      "selection_warning": "Do not estimate a batching benefit by conditioning on the post-run arrival-check outcome."})
    return {"original_commit": BASE, "original_report_sha256": hashlib.sha256(raw_index).hexdigest(),
            "original_source_sha256": report["source_sha256"], "raw_evidence_sha256": hashes, "cases": cases,
            "review_conclusion": "The former ~1.40x statement summarizes two passing pairs, not an unbiased effect estimate. Withdraw it as a confirmed batching benefit. All four pairs and warnings remain visible.",
            "historical_limit": "Original runs lack a loaded-worker model digest and a per-run fixture digest. Common path/export hashes and numerical agreement are evidence, but not byte-level attestation of each loaded session."}


def render(payload, conclusion):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Original heavy experiment review</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Original experiment review</h1><p>'
            + html.escape(conclusion) + '</p><p><a href="heavy-633eec0.json">JSON evidence audit</a></p><pre>' + html.escape(payload) + '</pre></html>')


def verify_saved(destination):
    payload = (destination / "heavy-633eec0.json").read_text()
    retained = json.loads(payload)
    current = evidence_audit()
    observed_training = retained.pop("training_reproduction")
    if retained != current:
        raise ValueError("Historical review differs from immutable committed raw evidence")
    names = {f"{prefix}{i}" for prefix in ("W", "b") for i in range(4)}
    if (observed_training["n_iter"] != 60 or set(observed_training["exact_arrays"]) != names
            or set(observed_training["max_abs_difference"]) != names
            or not all(observed_training["exact_arrays"].values())
            or any(observed_training["max_abs_difference"].values()) or observed_training["seconds"] <= 0):
        raise ValueError("Invalid retained same-environment training observation")
    if render(payload, current["review_conclusion"]) != (destination / "heavy-633eec0.html").read_text():
        raise ValueError("Historical review HTML differs from JSON")
    print("Original review reconciled with all immutable raw trials. Retained refit is a same-host observation, not rerun in this check.")


def write(destination, *, reproduce_training=False):
    result = evidence_audit()
    if reproduce_training:
        started = time.perf_counter()
        clf, _ = fit_model()
        keys = {**{f"W{i}": a.astype("<f4") for i, a in enumerate(clf.coefs_)},
                **{f"b{i}": a.astype("<f4") for i, a in enumerate(clf.intercepts_)}}
        with np.load(ROOT / "examples/heavy-digits/weights.npz", allow_pickle=False) as archive:
            equal = {k: bool(np.array_equal(a, archive[k])) for k, a in keys.items()}
            max_abs = {k: float(np.max(np.abs(a.astype(np.float64) - archive[k].astype(np.float64)))) for k, a in keys.items()}
        result["training_reproduction"] = {"n_iter": int(clf.n_iter_), "seconds": time.perf_counter() - started,
                                           "exact_arrays": equal, "max_abs_difference": max_abs,
                                           "scope": "A real same-environment refit of the formal training-only path; not a cross-platform bitwise guarantee."}
        if not all(equal.values()):
            raise ValueError("Same-environment training did not reproduce frozen weights exactly")
    destination.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (destination / "heavy-633eec0.json").write_text(payload)
    (destination / "heavy-633eec0.html").write_text(render(payload, result["review_conclusion"]))
    print("Original 192,000 request records rechecked; every pair retained.")
    if reproduce_training:
        print("Training refit:", result["training_reproduction"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reviews")
    parser.add_argument("--reproduce-training", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_saved(args.output)
    else:
        write(args.output, reproduce_training=args.reproduce_training)
