"""One fixed training protocol; held-out rows are never passed to fit or model selection."""
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(hidden_layer_sizes=(1024, 2048, 1024), activation="relu", solver="adam",
                batch_size=128, learning_rate_init=.001, alpha=.0001, max_iter=60,
                early_stopping=False, tol=0., n_iter_no_change=61, random_state=42)


def training_data():
    state = json.loads((ROOT / "examples/digits/model.json").read_text())
    p = state["provenance"]
    x, y = load_digits(return_X_y=True)
    train, test = np.asarray(p["train_ids"]), np.asarray(p["test_ids"])
    if (len(set(train)) != 1347 or len(set(test)) != 450 or set(train) & set(test)
            or set(train) | set(test) != set(range(len(x)))
            or hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest() != p["raw_pixels_uint8_sha256"]
            or hashlib.sha256(y.astype("<i8").tobytes()).hexdigest() != p["labels_int64_sha256"]):
        raise ValueError("Dataset/split differs")
    return (x[train] / 16).astype(np.float32), y[train], p


def fit_model():
    x_train, y_train, provenance = training_data()
    clf = MLPClassifier(**PROTOCOL)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # Fixed iteration budget, not a convergence claim.
        clf.fit(x_train, y_train)
    return clf, provenance


def train(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Refusing to overwrite frozen weights")
    started = time.perf_counter()
    clf, provenance = fit_model()
    destination.mkdir(parents=True)
    arrays = {**{f"W{i}": a.astype("<f4") for i, a in enumerate(clf.coefs_)},
              **{f"b{i}": a.astype("<f4") for i, a in enumerate(clf.intercepts_)}}
    np.savez_compressed(destination / "weights.npz", **arrays)
    metadata = {"name": "digits-wide-mlp", "dimensions": [64, 1024, 2048, 1024, 10],
                "parameters": sum(a.size for a in arrays.values()), "protocol": PROTOCOL,
                "n_iter": int(clf.n_iter_), "loss_curve": [float(v) for v in clf.loss_curve_],
                "training_seconds": time.perf_counter() - started,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                "weights_sha256": hashlib.sha256((destination / "weights.npz").read_bytes()).hexdigest(),
                "provenance": provenance, "classes": clf.classes_.tolist(),
                "selection": "One prespecified architecture and 60 epochs; no validation or held-out test based selection."}
    (destination / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("provenance", "loss_curve")}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/generated/retrained-heavy")
    train(parser.parse_args().output)
