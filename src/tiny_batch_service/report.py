"""Static offline report, rendered only from retained measurements."""

import html
import gzip
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import numpy as np

from .experiment import distribution, fingerprint, make_trace, summarize, source_hash, validate_response

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


def resolve_run(stub, directory=None):
    if "evidence_file" not in stub:
        return stub
    if directory is None or not re.fullmatch(r"raw/[a-z0-9-]+\.json\.gz", stub["evidence_file"]):
        raise ValueError("Invalid raw evidence path")
    path = Path(directory) / stub["evidence_file"]
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Linked raw evidence")
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != stub["evidence_sha256"]:
            raise ValueError("Raw evidence hash differs")
        run = json.loads(gzip.decompress(data))
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError("Invalid raw evidence") from error
    expected_keys = {"mode", "trial", "settings", "environment", "trace_sha256", "summary", "workload"}
    if set(stub) != expected_keys | {"evidence_file", "evidence_sha256"} or any(stub[k] != run[k] for k in expected_keys):
        raise ValueError("Raw evidence differs from report index")
    return run


def verify(report, reference, directory=None):
    if report["schema_version"] != 4 or report["source_sha256"] != source_hash() or report["measurement_scope"] != LIMITS:
        raise ValueError("Stale report provenance or scope")
    profile = report["model"]["profile"]
    if profile not in ("tiny", "heavy"):
        raise ValueError("Unknown measured model")
    audit_limit = 16384 if profile == "heavy" else 4096
    if report["policy"] != {"capacity": 64, "batch_size": 16, "wait_ms": 2., "request_timeout_ms": 250,
                            "generator_max_inflight": 512, "arrival_p99_budget_ms": 25, "rss_interval_ms": 5,
                            "audit_limit": audit_limit}:
        raise ValueError("Comparison policy changed")
    expected_order = []
    for case in report["cases"]:
        trace = case["trace"]
        if (not trace or case["trace_sha256"] != fingerprint(trace)
                or trace != make_trace(len(trace), case["rate"])
                or any(t["id"] != f"r{i:06d}" or t["at_s"] != i / case["rate"] or not 0 <= t["index"] < 450
                       for i, t in enumerate(trace))):
            raise ValueError("Trace changed")
        expected = [(trial, mode) for trial in range(report["repeats"])
                    for mode in (("single", "dynamic") if trial % 2 == 0 else ("dynamic", "single"))]
        if [(r["trial"], r["mode"]) for r in case["runs"]] != expected:
            raise ValueError("Missing trials or unbalanced mode order")
        environment = None
        for stub in case["runs"]:
            run = resolve_run(stub, directory)
            expected_settings = {"batch_size": 1 if run["mode"] == "single" else 16,
                                 "wait_ms": 0. if run["mode"] == "single" else 2., "capacity": 64, "audit_limit": audit_limit}
            if run["settings"] != expected_settings or run["trace_sha256"] != case["trace_sha256"]:
                raise ValueError("Unequal work or settings")
            if (run["workload"] != {"loaded_model_sha256": report["model"]["model_sha256"],
                                    "fixture_sha256": report["model"]["fixture_sha256"]}
                    or run["environment"]["worker"]["loaded_model_sha256"] != report["model"]["model_sha256"]):
                raise ValueError("Loaded model or actual input fixture differs")
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
        if case["comparison"] != comparison(case, report["repeats"]):
            raise ValueError("Comparison verdict differs from all prespecified pairs")
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
                         counts["overloaded"], f'{100*counts["overloaded"]/(len(case["trace"])*len(summaries)):.2f}%', counts["timeout"], f'{med("success_per_second"):.1f}',
                         latency("p50"), latency("p95"), latency("p99"),
                         latency("p95", "scheduled_success_latency_ms"),
                         f'{max(s["header_send_lag_ms"]["p99"] for s in summaries):.3f}',
                         f'{max(s["server_arrival_lag_ms"]["p99"] for s in summaries):.3f}',
                         f'{max(s["peak_paired_rss_bytes"] for s in summaries)/1048576:.2f}',
                         f'{max(s["max_rss_gap_ms"] for s in summaries):.3f}',
                         f'{100*med("backend_busy_fraction"):.1f}%', f'{med("mean_batch_rows"):.2f}',
                         f'{sum(s["arrival_fidelity_passed"] for s in summaries)}/{len(summaries)} pass'])
    return rows


def comparison(case, repeats):
    pairs = []
    for trial in range(repeats):
        group = {r["mode"]: r["summary"] for r in case["runs"] if r["trial"] == trial}
        if set(group) != {"single", "dynamic"}:
            raise ValueError("Missing comparison pair")
        single, dynamic = group["single"], group["dynamic"]
        a, b = single["scheduled_success_latency_ms"]["p95"], dynamic["scheduled_success_latency_ms"]["p95"]
        pairs.append({"trial": trial, "arrival_check_passed": bool(single["arrival_fidelity_passed"] and dynamic["arrival_fidelity_passed"]),
                      "descriptive_throughput_ratio": dynamic["success_per_second"] / single["success_per_second"] if single["success_per_second"] else None,
                      "rejection_delta_pp": 100 * (dynamic["rejection_rate"] - single["rejection_rate"]),
                      "scheduled_success_p95_delta_ms": b - a if a is not None and b is not None else None})
    passed = sum(p["arrival_check_passed"] for p in pairs)
    return {"basis": "all_prespecified_pairs", "passing_arrival_pairs": passed, "total_pairs": repeats,
            "verdict": "inconclusive_arrival_fidelity" if passed != repeats else "descriptive_only_all_pairs_pass_coarse_check",
            "warning": "Arrival-check outcome is measured after execution and can depend on mode/performance. Conditioning on passing pairs can introduce selection bias. No pass-only effect estimate is endorsed. A coarse-check pass is not a causal or statistical-significance claim.",
            "pairs": pairs}


def paired_rows(report):
    rows = []
    for case in report["cases"]:
        ratios, rejects, tails, passing_pairs = [], [], [], 0
        for trial in range(report["repeats"]):
            pair = {r["mode"]: r["summary"] for r in case["runs"] if r["trial"] == trial}
            single, dynamic = pair["single"], pair["dynamic"]
            passing_pairs += single["arrival_fidelity_passed"] and dynamic["arrival_fidelity_passed"]
            if single["success_per_second"] > 0:
                ratios.append(dynamic["success_per_second"] / single["success_per_second"])
            rejects.append(100 * (dynamic["rejection_rate"] - single["rejection_rate"]))
            if single["scheduled_success_latency_ms"]["p95"] is not None and dynamic["scheduled_success_latency_ms"]["p95"] is not None:
                tails.append(dynamic["scheduled_success_latency_ms"]["p95"] - single["scheduled_success_latency_ms"]["p95"])
        rows.append([case["rate"], f"{median(ratios):.3f} [{min(ratios):.3f}, {max(ratios):.3f}]" if ratios else "—",
                     f"{median(rejects):+.2f}", f"{median(tails):+.3f}" if tails else "—",
                     f'{passing_pairs}/{report["repeats"]}', comparison(case, report["repeats"])["verdict"]])
    return rows


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((directory / "fixture.json").read_text())
    reference = np.asarray(fixture["reference"]["data"], dtype=fixture["reference"]["dtype"])
    verify(report, reference, directory)
    raw = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    verify(json.loads(raw), reference, directory)
    (directory / "report.json").write_text(raw)
    esc = lambda v: html.escape(str(v))
    headers = ["Offered req/s", "Mode", "Success / offered", "Rejected", "Rejected %", "Timeout", "Success/s",
               "Dispatch success p50 ms", "p95 ms", "p99 ms", "Scheduled success p95 ms",
               "Worst header lag p99 ms", "Worst admission lag p99 ms", "Peak HTTP+worker RSS MiB", "Max RSS gap ms",
               "Backend+IPC busy", "Mean batch rows", "Arrival check"]
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in table_rows(report))
    scope = "".join(f"<li>{esc(line)}</li>" for line in LIMITS)
    details = {"source_sha256": report["source_sha256"], "model": report["model"], "policy": report["policy"],
               "all_pair_comparisons": [{"rate": c["rate"], **c["comparison"]} for c in report["cases"]],
               "trials": [{"rate": c["rate"], "mode": r["mode"], "trial": r["trial"], "summary": r["summary"],
                            "workload": r["workload"],
                            "http_lifetime_high_water_rss_bytes": r["audit"]["http_high_water_rss_bytes"],
                            "worker_lifetime_high_water_rss_bytes": r["audit"]["worker_high_water_rss_bytes"]}
                           for c in report["cases"] for stub in c["runs"] for r in [resolve_run(stub, directory)]]}
    evidence_links = "".join(f'<li><a href="{esc(r["evidence_file"])}">{esc(r["evidence_file"])}</a> · SHA-256 {esc(r["evidence_sha256"])}</li>'
                             for c in report["cases"] for r in c["runs"] if "evidence_file" in r)
    comparisons = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in paired_rows(report))
    selection = (f'<section><h2>Flow selection and limits</h2><p>{esc(report["flow_selection"]["rationale"])}</p>'
                 f'<p>Selected formal grid applied to this run: {esc(report["selected_grid_applied"])}. '
                 f'<a href="calibration.json.gz">Every exploratory calibration trial (compressed JSON)</a>; '
                 f'SHA-256 {esc(report["flow_selection"]["calibration_sha256"])}</p></section>'
                 if "flow_selection" in report else "")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny Batch Service measurements</title><style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;background:#f5f8fa;color:#172b3a}}section{{background:white;border:1px solid #dce4e8;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}</style>
<h1>Tiny Batch Service</h1><p>{esc(report["generated_utc"])} · actual HTTP + CPU ONNX Runtime · {esc(report["host"])}</p>
<p>Model: {esc(report["model"]["name"])} · {esc(report["model"]["parameters"])} parameters · held-out reference accuracy {esc(report["model"]["correct"])}/{esc(report["model"]["rows"])}. Agreement with this reference is checked for every successful request. This architecture is a systems workload, not an efficiency recommendation.</p>
<section><h2>Matched scheduled arrivals</h2><p>Median per-trial throughput and successful-request percentiles; maximum paired sampled RSS, sample gap and per-trial p99 arrival lag. Counts sum every trial; rejection percentage uses total rejected / total offered. Each mode receives the same trace and total admitted capacity. The fixed arrival check requires per-trial p99 header-send and admission lag ≤25 ms. Passing this coarse check does not prove identical actual arrivals or burstiness, especially at sub-millisecond spacing. Inspect lag and scheduled latency; warnings preclude a clean traffic-matched performance conclusion.</p>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{esc(h)}</th>' for h in headers)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>Successful-only percentiles can improve when more requests are rejected. Inspect outcome counts, all-terminal percentiles and raw trials before making a performance claim. No speedup headline is generated.</p></section>
<section><h2>Paired comparison — all prespecified pairs</h2><p>Dynamic / single successful-throughput ratio: median [min, max] across every paired trial. Deltas are medians of paired differences, not differences of pooled rates. Positive rejection or p95 deltas mean dynamic is worse. This is descriptive, not a confidence interval. Any arrival-warning pair makes the case inconclusive for a matched-arrival benefit claim. Passing is a post-run outcome: restricting analysis to passing pairs can introduce selection bias. No pass-only effect estimate is endorsed. Even all pairs passing the coarse check does not establish causality or statistical significance. Reject/timeout rates must accompany every ratio; saturated successful-only latency has survivor bias.</p><table><tr><th>Offered req/s</th><th>All-pair descriptive ratio</th><th>Rejection delta pp</th><th>Scheduled success p95 delta ms</th><th>Passing arrival pairs</th><th>Assessment</th></tr>{comparisons}</table></section>
{selection}
<section><h2>Measurement scope</h2><ul>{scope}</ul></section>
<section><h2>Evidence</h2><p><a href="report.json">JSON report (embedded raw trials or hash-linked compressed JSON trial files below)</a> · <a href="fixture.json">Retained input/reference fixture</a></p><ul>{evidence_links}</ul>
<details><summary>Per-trial statistics, lifetime high-water observations and provenance</summary><pre>{esc(json.dumps(details, indent=2))}</pre></details></section></html>'''
    (directory / "report.html").write_text(page)


def overview(report, methodology="../../docs/METHODOLOGY.md"):
    lines = ["# Recorded HTTP batching experiment", "", "These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.", "",
             "Counts include all trials; rejection percentage is total rejected / total offered. Throughput and latency cells are medians of per-trial statistics; arrival-lag p99 is the worst trial; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.", "",
             "| Offered req/s | Mode | Success/offered | Rejected | Rejected % | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Worst header lag p99 ms | Worst admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Backend+IPC busy | Mean batch rows | Arrival check |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table_rows(report)]
    lines += ["", "All-pair descriptive throughput ratio is dynamic / single: median [min, max]. Deltas are medians of paired differences, not differences of pooled rates. Every prespecified pair is included. Any arrival-warning pair makes the case inconclusive for a matched-arrival benefit claim. Passing is a post-run outcome that may depend on mode/performance: selecting only passing pairs can introduce selection bias; no pass-only effect estimate is endorsed. All pairs passing the coarse check still does not establish causality or statistical significance. Successful-only latency has survivor bias; consider all rejection/timeout counts.", "",
              "| Offered req/s | All-pair descriptive ratio | Rejection delta pp | Scheduled success p95 delta ms | Passing arrival pairs | Assessment |", "|---:|---:|---:|---:|---:|---|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in paired_rows(report)]
    lines += ["", "Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.", "",
              "The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.", "",
              f'Model: {report["model"]["name"]}, {report["model"]["parameters"]:,} parameters, held-out reference accuracy {report["model"]["correct"]}/{report["model"]["rows"]}. This is a deliberately chosen systems workload, not an efficient-classifier recommendation or evidence for all networks/accelerators. Backend+IPC busy is dispatch-to-result wall time divided by the observation interval; it is not CPU utilization. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.', "",
              "- [Raw JSON](report.json)", "- [Offline HTML](report.html)", f"- [Methodology]({methodology})", ""]
    return "\n".join(lines)
