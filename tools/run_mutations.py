"""Only behavioral assertion failures detect faults; hash/import failures do not count."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = "test_scheduler.SchedulerTests."
MUTATIONS = [
    ("reverse_result_mapping", "for entry, output in zip(batch, outputs):", "for entry, output in zip(batch, list(reversed(outputs))):", "test_tail_deadline_and_exact_response_mapping"),
    ("ignore_deadline", '        self._terminal(entry, "timeout", 504, reason="request_deadline")', '        return False', "test_queued_timeout_and_cancel_never_reach_backend"),
    ("free_running_capacity_early", '        if entry.phase == "queued":', '        if True:', "test_running_cancel_retains_capacity_until_compute_finishes"),
    ("ignore_running_admission", 'elif len(self.entries) >= self.capacity:', 'elif len(self.pending) >= self.capacity:', "test_capacity_includes_computing_and_queued_work"),
]


def run(mutation=None):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for folder in ("src", "tests"):
            shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        source = root / "src/tiny_batch_service/scheduler.py"
        if mutation is not None:
            name, before, after, target = mutation
            text = source.read_text()
            if text.count(before) != 1:
                raise ValueError("Mutation target is not unique: " + name)
            source.write_text(text.replace(before, after))
            tests = [TARGET + target]
        else:
            tests = [TARGET + m[3] for m in MUTATIONS]
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        result = subprocess.run([sys.executable, "-m", "unittest", *tests], cwd=root / "tests", env=env,
                                capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr


if __name__ == "__main__":
    code, output = run()
    if code != 0:
        raise SystemExit("Unchanged mutation baseline failed: " + output)
    for mutation in MUTATIONS:
        code, output = run(mutation)
        if code != 1 or "FAIL:" not in output or "ERROR:" in output:
            raise SystemExit("Fault not rejected by behavioral assertions: " + mutation[0] + "\n" + output)
        print("Detected:", mutation[0])
    print("All four source faults detected by behavioral assertions.")
