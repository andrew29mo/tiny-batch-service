"""Static offline report, rendered only from retained measurements."""

import html
import json
from pathlib import Path
from statistics import median
import numpy as np

from .experiment import distribution, fingerprint, summarize, source_hash, validate_response

LIMITS = [
    "Local CPU-only experiment with one HTTP process and one ORT worker. No authentication or multi-tenant isolation; loopback/trusted inputs only.",
    "Both modes replay identical scheduled arrivals and inputs. Actual dispatch/header/server arrival lag is retained; timing is not perfectly deterministic.",
    "Successful-request latency excludes rejected/timed-out work. All outcome counts and all-terminal latency are shown alongside it; throughput counts successes only.",
    "Scheduled latency runs from planned arrival through complete response decoding, including generator delay. Dispatch latency starts at actual client task dispatch. Reference checks are outside both timers but affect later scheduling; neither is raw model/kernel latency.",
    "Paired RSS adds near-contemporaneous HTTP and worker process observations. Shared pages may be counted twice; this is not unique physical RAM or model-only memory.",
    "RSS sampling requests 5 ms intervals and can miss short spikes. Actual sample gaps and paired-read duration are reported. Per-process OS lifetime high-water observations are retained separately, never added as a simultaneous peak.",
    "The generator/controller, fixtures and multiprocessing resource-tracker helper are excluded from HTTP-plus-worker RSS. Audit buffers, diagnostic stats serialization and HTTP/IPC overhead are included. This is not total deployment memory. Finite audit retention is checked for eviction.",
    "Warmups, imports and startup are excluded from traffic measurements. Processes are fresh per trial; caches, thermals and same-host load-generator contention are uncontrolled.",
    "No universal batching speedup is assumed. Tiny models may gain little or lose latency through batching. Results are descriptive on one host.",
]


def verify(report, reference):
    if report["schema_version"] != 2 or report["source_sha256"] != source_hash() or report["measurement_scope"] != LIMITS:
        raise ValueError("Stale report provenance or scope")
    if report["policy"] != {"capacity": 64, "batch_size": 16, "wait_ms": 2., "request_timeout_ms": 250,
                            "generator_max_inflight": 512, "arrival_p99_budget_ms": 25, "rss_interval_ms": 5}:
        raise ValueError("Comparison policy changed")
    expected_order = []
    for case in report["cases"]:
        trace = case["trace"]
        if (not trace or case["trace_sha256"] != fingerprint(trace)
                or any(t["id"] != f"r{i:06d}" or t["at_s"] != i / case["rate"] or not 0 <= t["index"] < 450
                       for i, t in enumerate(trace))):
            raise ValueError("Trace changed")
        expected = [(trial, mode) for trial in range(report["repeats"])
                    for mode in (("single", "dynamic") if trial % 2 == 0 else ("dynamic", "single"))]
        if [(r["trial"], r["mode"]) for r in case["runs"]] != expected:
            raise ValueError("Missing trials or unbalanced mode order")
        environment = None
        for run in case["runs"]:
            expected_settings = {"batch_size": 1 if run["mode"] == "single" else 16,
                                 "wait_ms": 0. if run["mode"] == "single" else 2., "capacity": 64, "audit_limit": 4096}
            if run["settings"] != expected_settings or run["trace_sha256"] != case["trace_sha256"]:
                raise ValueError("Unequal work or settings")
            environment = run["environment"] if environment is None else environment
            if (environment != run["environment"] or run["environment"]["worker"]["provider"] != "CPUExecutionProvider"
                    or run["environment"]["worker"]["intra_threads"] != 1 or run["environment"]["worker"]["inter_threads"] != 1):
                raise ValueError("Environment/provider mismatch")
            if run["summary"] != summarize(run, trace):
                raise ValueError("Statistics differ from raw evidence")
            for row in run["client"]:
                if row["status"] == "success":
                    validate_response(row, reference[row["index"]])
            expected_order.append({"rate": case["rate"], "trial": run["trial"], "mode": run["mode"]})
    if report["execution_order"] != expected_order:
        raise ValueError("Execution order differs")


def table_rows(report):
    rows = []
    for case in report["cases"]:
        for mode in ("single", "dynamic"):
            summaries = [r["summary"] for r in case["runs"] if r["mode"] == mode]
            def med(key):
                return median(s[key] for s in summaries)
            def latency(q, key="success_latency_ms"):
                vals = [s[key][q] for s in summaries if s[key][q] is not None]
                return f"{median(vals):.3f}" if vals else "—"
            counts = {k: sum(s["counts"].get(k, 0) for s in summaries) for k in ("success", "overloaded", "timeout")}
            rows.append([case["rate"], mode, f'{counts["success"]}/{len(case["trace"])*len(summaries)}',
                         counts["overloaded"], counts["timeout"], f'{med("success_per_second"):.1f}',
                         latency("p50"), latency("p95"), latency("p99"),
                         latency("p95", "scheduled_success_latency_ms"),
                         f'{median(s["header_send_lag_ms"]["p99"] for s in summaries):.3f}',
                         f'{median(s["server_arrival_lag_ms"]["p99"] for s in summaries):.3f}',
                         f'{max(s["peak_paired_rss_bytes"] for s in summaries)/1048576:.2f}',
                         f'{max(s["max_rss_gap_ms"] for s in summaries):.3f}',
                         "pass" if all(s["arrival_fidelity_passed"] for s in summaries) else "warning"])
    return rows


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((directory / "fixture.json").read_text())
    reference = np.asarray(fixture["reference"]["data"], dtype=fixture["reference"]["dtype"])
    verify(report, reference)
    raw = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    verify(json.loads(raw), reference)
    (directory / "report.json").write_text(raw)
    esc = lambda v: html.escape(str(v))
    headers = ["Offered req/s", "Mode", "Success / offered", "Rejected", "Timeout", "Success/s",
               "Dispatch success p50 ms", "p95 ms", "p99 ms", "Scheduled success p95 ms",
               "Header lag p99 ms", "Admission lag p99 ms", "Peak HTTP+worker RSS MiB", "Max RSS gap ms", "Arrival check"]
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in table_rows(report))
    scope = "".join(f"<li>{esc(line)}</li>" for line in LIMITS)
    details = {"source_sha256": report["source_sha256"], "model": report["model"], "policy": report["policy"],
               "trials": [{"rate": c["rate"], "mode": r["mode"], "trial": r["trial"], "summary": r["summary"],
                            "http_lifetime_high_water_rss_bytes": r["audit"]["http_high_water_rss_bytes"],
                            "worker_lifetime_high_water_rss_bytes": r["audit"]["worker_high_water_rss_bytes"]}
                           for c in report["cases"] for r in c["runs"]]}
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny Batch Service measurements</title><style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;background:#f5f8fa;color:#172b3a}}section{{background:white;border:1px solid #dce4e8;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}</style>
<h1>Tiny Batch Service</h1><p>{esc(report["generated_utc"])} · actual HTTP + CPU ONNX Runtime · {esc(report["host"])}</p>
<section><h2>Matched scheduled arrivals</h2><p>Median per-trial throughput and successful-request percentiles; maximum paired sampled RSS and sample gap. Counts sum every trial. Each mode receives the same trace and total admitted capacity. The fixed arrival check requires per-trial p99 header-send and admission lag ≤25 ms. Passing this coarse check does not prove identical actual arrivals or burstiness, especially at sub-millisecond spacing. Inspect lag and scheduled latency; warnings preclude a clean traffic-matched performance conclusion.</p>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{esc(h)}</th>' for h in headers)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>Successful-only percentiles can improve when more requests are rejected. Inspect outcome counts, all-terminal percentiles and raw trials before making a performance claim. No speedup headline is generated.</p></section>
<section><h2>Measurement scope</h2><ul>{scope}</ul></section>
<section><h2>Evidence</h2><p><a href="report.json">Complete raw JSON: every request, server terminal record, batch and RSS sample</a> · <a href="fixture.json">Retained input/reference fixture</a></p>
<details><summary>Per-trial statistics, lifetime high-water observations and provenance</summary><pre>{esc(json.dumps(details, indent=2))}</pre></details></section></html>'''
    (directory / "report.html").write_text(page)


def overview(report, methodology="../../docs/METHODOLOGY.md"):
    lines = ["# Recorded HTTP batching experiment", "", "These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.", "",
             "Success counts include all trials. Throughput and latency cells are medians of per-trial statistics; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.", "",
             "| Offered req/s | Mode | Success/offered | Rejected | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Header lag p99 ms | Admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Arrival check |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table_rows(report)]
    lines += ["", "Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.", "",
              "The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.", "",
              "The loaded model is a tiny frozen Digits MLP. This experiment does not establish throughput or latency for larger networks, accelerators or distributed deployments. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.", "",
              "- [Raw JSON](report.json)", "- [Offline HTML](report.html)", f"- [Methodology]({methodology})", ""]
    return "\n".join(lines)
