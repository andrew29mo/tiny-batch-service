"""Independently regenerate references, recompute every result and check publication hygiene."""

import argparse
import hashlib
import json
import gzip
import os
from pathlib import Path
import re
import subprocess
import shutil
import tempfile

import numpy as np

from make_model import prepare, digest
from tiny_batch_service.report import overview, verify, write_report
from tiny_batch_service.experiment import summarize, validate_response

ROOT = Path(__file__).resolve().parents[1]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check(directory, *, smoke=False):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    profile = report["model"]["profile"]
    if profile == "heavy":
        selection = json.loads((ROOT / "examples/heavy-digits/flow-selection.json").read_text())
        need(report["flow_selection"] == selection, "Flow selection changed")
        calibration = directory / "calibration.json.gz"
        need(digest(calibration) == selection["calibration_sha256"], "Calibration hash differs")
    if not smoke:
        if profile == "heavy":
            need(report["selected_grid_applied"] is True, "Not the selected repeated experiment")
        expected_rates = [100., 1000., 3000.] if profile == "tiny" else [500., 3500., 6000., 10000.]
        need(report["repeats"] == 4 and [c["rate"] for c in report["cases"]] == expected_rates, "Recorded case/trial policy differs")
        need(all(len(c["trace"]) == (480 if profile == "tiny" else 6000) for c in report["cases"]), "Recorded trace count differs")
    canonical = json.loads((directory / "fixture.json").read_text())
    need(set(canonical) == {"inputs", "reference", "labels", "sample_ids"}, "Incomplete retained fixture")
    arrays = {k: np.asarray(v["data"], dtype=v["dtype"]) for k, v in canonical.items()}
    for name, array in arrays.items():
        need(list(array.shape) == canonical[name]["shape"] and np.isfinite(array).all(), "Invalid retained fixture")
    with tempfile.TemporaryDirectory() as temp:
        fresh = Path(temp)
        manifest = prepare(fresh, profile)
        for key in manifest.keys() - {"fixture_sha256"}:
            need(report["model"][key] == manifest[key], "Model/dataset provenance differs")
        np.savez(fresh / "retained.npz", **arrays)
        need(digest(fresh / "retained.npz") == report["model"]["fixture_sha256"], "Measured fixture hash differs")
        with np.load(fresh / "fixture.npz", allow_pickle=False) as archive:
            for key in arrays:
                need(arrays[key].dtype == archive[key].dtype and arrays[key].shape == archive[key].shape, "Fixture dtype/shape differs")
                valid = (np.allclose(arrays[key], archive[key], rtol=1e-4, atol=1e-5) if key == "reference"
                         else np.array_equal(arrays[key], archive[key]))
                need(valid, "Independent fixture regeneration differs beyond fixed tolerance")
        verify(report, arrays["reference"], directory)
        if profile == "heavy":
            pilot = json.loads(gzip.decompress(calibration.read_bytes()))
            need(pilot["model"] == report["model"], "Calibration model differs")
            need([c["rate"] for c in pilot["cases"]] == [500., 1500., 3000., 6000., 10000., 15000., 20000.], "Missing calibration probes")
            for case in pilot["cases"]:
                for run in case["runs"]:
                    need(run["summary"] == summarize(run, case["trace"], require_memory_bracketing=False), "Legacy calibration summary differs")
                    for row in run["client"]:
                        if row["status"] == "success":
                            validate_response(row, arrays["reference"][row["index"]])
        (fresh / "fixture.json").write_text((directory / "fixture.json").read_text())
        if (directory / "raw").exists():
            shutil.copytree(directory / "raw", fresh / "raw")
        write_report(report, fresh)
        need((fresh / "report.json").read_bytes() == (directory / "report.json").read_bytes(), "JSON is not canonical")
        need((fresh / "report.html").read_bytes() == (directory / "report.html").read_bytes(), "HTML differs from evidence")
    need((directory / "README.md").read_text() == overview(report, os.path.relpath(ROOT / "docs/METHODOLOGY.md", directory)), "Overview differs")
    return report


def publication():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    private = re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\")
    secrets = re.compile(rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    banned = {"AGENTS.md", ".env", ".venv", "__pycache__", ".DS_Store", ".codex", ".agents"}
    for filename in filter(None, tracked):
        path = ROOT / filename
        need(not path.is_symlink() and not any(p in banned or p.startswith(".env.") for p in Path(filename).parts), "Private or linked tracked file: " + filename)
        frozen_weights = filename == "examples/heavy-digits/weights.npz"
        need(path.suffix not in (".onnx", ".npz", ".pyc", ".so", ".whl") or frozen_weights, "Generated tracked binary: " + filename)
        data = path.read_bytes()
        if frozen_weights:
            state = json.loads((ROOT / "examples/heavy-digits/model.json").read_text())
            need(hashlib.sha256(data).hexdigest() == state["weights_sha256"], "Frozen binary hash differs")
            continue  # Shape/dtype/finite contents are checked by independent model preparation.
        if path.suffix == ".gz":
            need(bool(re.fullmatch(r"results/heavy-macos-arm64/(?:raw/[a-z0-9-]+|calibration)\.json\.gz", filename)), "Unexpected compressed artifact")
            data = gzip.decompress(data)
        need(b"\0" not in data and not private.search(data) and not secrets.search(data), "Private content pattern: " + filename)
        data.decode("utf-8")
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", data.decode()):
                if "://" not in link and not link.startswith("#"):
                    need((path.parent / link.split("#")[0]).exists(), "Broken local link: " + filename)
    need(any(tracked), "Stage intended files before publication checks")
    print("Tracked files: no targeted private/generated-file patterns or broken documentation links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/macos-arm64")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    check(args.directory, smoke=args.smoke)
    publication()
    print("Verified independent references, every HTTP result/terminal ID, trace coverage, fair settings, all timing/RSS summaries and HTML.")
