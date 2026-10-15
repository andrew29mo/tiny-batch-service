"""Export the frozen wider trained network and independently regenerate held-out references."""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from make_model import ROOT, digest
from train_heavy_model import PROTOCOL, training_data


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "examples/heavy-digits"
    state = json.loads((source / "model.json").read_text())
    _, _, provenance = training_data()
    if (state["provenance"] != provenance or state["protocol"] != json.loads(json.dumps(PROTOCOL))
            or state["n_iter"] != 60 or state["dimensions"] != [64, 1024, 2048, 1024, 10]
            or state["classes"] != list(range(10)) or digest(source / "weights.npz") != state["weights_sha256"]):
        raise ValueError("Frozen training provenance/weights differs")
    with np.load(source / "weights.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"{prefix}{i}" for prefix in ("W", "b") for i in range(4)}:
            raise ValueError("Unexpected weight arrays")
        weights, biases = [[archive[f"{prefix}{i}"] for i in range(4)] for prefix in ("W", "b")]
    dims = state["dimensions"]
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if (weight.dtype != np.float32 or bias.dtype != np.float32 or weight.shape != (dims[i], dims[i+1])
                or bias.shape != (dims[i+1],) or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
            raise ValueError("Invalid frozen parameters")
    if sum(a.size for a in weights + biases) != state["parameters"]:
        raise ValueError("Parameter count differs")
    nodes, initializers, previous = [], [], "X"
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        nodes.append(h.make_node("Gemm", [previous, f"W{i}", f"b{i}"], [f"A{i}"]))
        previous = f"H{i}" if i < 3 else "probabilities"
        nodes.append(h.make_node("Relu" if i < 3 else "Softmax", [f"A{i}"], [previous], **({} if i < 3 else {"axis": 1})))
        initializers += [nh.from_array(weight, f"W{i}"), nh.from_array(bias, f"b{i}")]
    graph = h.make_graph(nodes, "trained-wide-digits", [h.make_tensor_value_info("X", T.FLOAT, ["N", 64])],
                         [h.make_tensor_value_info("probabilities", T.FLOAT, ["N", 10])], initializer=initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    x, y = load_digits(return_X_y=True)
    selected = np.asarray(provenance["test_ids"])
    inputs = (x[selected] / 16).astype(np.float32)
    clf = MLPClassifier()
    clf.coefs_, clf.intercepts_ = [[a.astype(np.float64) for a in group] for group in (weights, biases)]
    clf.classes_, clf.n_features_in_, clf.n_layers_, clf.n_outputs_, clf.out_activation_ = np.arange(10), 64, 5, 10, "softmax"
    with threadpool_limits(limits=1):
        reference = clf.predict_proba(x[selected] / 16).astype(np.float32)
        values = inputs
        for weight, bias in zip(weights, biases):
            values = values @ weight + bias
            if weight is not weights[-1]:
                values = np.maximum(values, 0)
        exp = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
        independent = exp / exp.sum(axis=1, keepdims=True)
    if not np.allclose(reference, independent, rtol=1e-4, atol=1e-5):
        raise ValueError("Independent float32/sklearn references differ")
    np.savez(directory / "fixture.npz", inputs=inputs, reference=reference, labels=y[selected], sample_ids=selected)
    manifest = {"profile": "heavy", "name": state["name"], "dimensions": dims, "parameters": state["parameters"],
                "model_sha256": digest(directory / "model.onnx"), "model_bytes": (directory / "model.onnx").stat().st_size,
                "fixture_sha256": digest(directory / "fixture.npz"), "state_sha256": digest(source / "model.json"),
                "weights_sha256": digest(source / "weights.npz"), "rows": len(selected), "features": 64, "classes": 10,
                "correct": int(np.sum(reference.argmax(1) == y[selected])),
                "reference": "frozen float32 weights promoted to sklearn float64 predict_proba, cast to float32; independent NumPy float32 check"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(ROOT / "examples/generated/heavy"), indent=2))
