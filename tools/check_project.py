"""Independently regenerate references, recompute every result and check publication hygiene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from make_model import prepare, digest
from tiny_batch_service.report import overview, verify, write_report

ROOT = Path(__file__).resolve().parents[1]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check(directory, *, smoke=False):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if not smoke:
        need(report["repeats"] == 4 and [c["rate"] for c in report["cases"]] == [100., 1000., 3000.], "Recorded case/trial policy differs")
        need(all(len(c["trace"]) == 480 for c in report["cases"]), "Recorded trace count differs")
    canonical = json.loads((directory / "fixture.json").read_text())
    need(set(canonical) == {"inputs", "reference", "labels", "sample_ids"}, "Incomplete retained fixture")
    arrays = {k: np.asarray(v["data"], dtype=v["dtype"]) for k, v in canonical.items()}
    for name, array in arrays.items():
        need(list(array.shape) == canonical[name]["shape"] and np.isfinite(array).all(), "Invalid retained fixture")
    with tempfile.TemporaryDirectory() as temp:
        fresh = Path(temp)
        manifest = prepare(fresh)
        for key in ("model_sha256", "state_sha256", "rows", "features", "classes", "correct", "reference"):
            need(report["model"][key] == manifest[key], "Model/dataset provenance differs")
        np.savez(fresh / "retained.npz", **arrays)
        need(digest(fresh / "retained.npz") == report["model"]["fixture_sha256"], "Measured fixture hash differs")
        with np.load(fresh / "fixture.npz", allow_pickle=False) as archive:
            for key in arrays:
                need(arrays[key].dtype == archive[key].dtype and arrays[key].shape == archive[key].shape, "Fixture dtype/shape differs")
                valid = (np.allclose(arrays[key], archive[key], rtol=1e-4, atol=1e-5) if key == "reference"
                         else np.array_equal(arrays[key], archive[key]))
                need(valid, "Independent fixture regeneration differs beyond fixed tolerance")
        verify(report, arrays["reference"])
        (fresh / "fixture.json").write_text((directory / "fixture.json").read_text())
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
        need(path.suffix not in (".onnx", ".npz", ".pyc", ".so", ".whl"), "Generated tracked binary: " + filename)
        data = path.read_bytes()
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
