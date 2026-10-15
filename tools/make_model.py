"""Export the attributed frozen Digits MLP; no fitting or test-based selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(directory, profile="tiny"):
    if profile == "heavy":
        from make_heavy_model import prepare as prepare_heavy
        return prepare_heavy(directory)
    if profile != "tiny":
        raise ValueError("Unknown model profile")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "examples/digits/model.json"
    state = json.loads(state_path.read_text())
    x, y = load_digits(return_X_y=True)
    p = state["provenance"]
    if (hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest() != p["raw_pixels_uint8_sha256"]
            or hashlib.sha256(y.astype("<i8").tobytes()).hexdigest() != p["labels_int64_sha256"]
            or set(p["train_ids"]) & set(p["test_ids"])):
        raise ValueError("Dataset or frozen split differs")
    selected = np.array(p["test_ids"])
    values = (x[selected] / 16).astype(np.float32)
    w1, w2 = [np.asarray(a, dtype=np.float32) for a in state["coefs_"]]
    b1, b2 = [np.asarray(a, dtype=np.float32) for a in state["intercepts_"]]
    nodes = [h.make_node("Gemm", ["X", "W1", "b1"], ["A"]), h.make_node("Relu", ["A"], ["H"]),
             h.make_node("Gemm", ["H", "W2", "b2"], ["logits"]),
             h.make_node("Softmax", ["logits"], ["probabilities"], axis=1)]
    graph = h.make_graph(nodes, "frozen-digits", [h.make_tensor_value_info("X", T.FLOAT, ["N", 64])],
                         [h.make_tensor_value_info("probabilities", T.FLOAT, ["N", 10])],
                         initializer=[nh.from_array(a, n) for n, a in (("W1", w1), ("b1", b1), ("W2", w2), ("b2", b2))])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    clf = MLPClassifier()
    for key in ("coefs_", "intercepts_"):
        setattr(clf, key, [np.asarray(a, dtype=np.float64) for a in state[key]])
    clf.classes_ = np.asarray(state["classes_"], dtype=np.int64)
    for key in ("n_features_in_", "n_layers_", "n_outputs_", "out_activation_"):
        setattr(clf, key, state[key])
    with threadpool_limits(limits=1):
        reference = clf.predict_proba(x[selected] / 16).astype(np.float32)
        hidden = np.maximum(values @ w1 + b1, 0)
        logits = hidden @ w2 + b2
        exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
        numpy_probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.allclose(numpy_probabilities, reference, rtol=1e-4, atol=1e-5):
        raise ValueError("Independent float32 and sklearn references differ")
    np.savez(directory / "fixture.npz", inputs=values, reference=reference, labels=y[selected], sample_ids=selected)
    manifest = {"profile": "tiny", "name": "digits-tiny-mlp", "dimensions": [64, 32, 10], "parameters": 2410,
                "model_bytes": (directory / "model.onnx").stat().st_size,
                "model_sha256": digest(directory / "model.onnx"), "fixture_sha256": digest(directory / "fixture.npz"),
                "state_sha256": digest(state_path), "rows": len(selected), "features": 64, "classes": 10,
                "correct": int(np.sum(reference.argmax(1) == y[selected])),
                "reference": "frozen sklearn float64 predict_proba cast to float32; independent NumPy float32 export check"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(ROOT / "examples/generated/digits"), indent=2))
