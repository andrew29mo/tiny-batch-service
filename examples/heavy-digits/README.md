# Frozen wider Digits MLP

A deliberately wider, genuinely trained matrix workload for batching experiments. It is not an efficient architecture recommendation for Digits.

- Architecture: 64 → 1,024 → 2,048 → 1,024 → 10; ReLU hidden layers and Softmax output.
- Parameters: 4,274,186 float32 values; exported ONNX size 17,097,159 bytes.
- Training: one fixed seed-42 Adam run, 60 epochs, batch 128, learning rate 0.001, L2 alpha 0.0001; no early stopping or held-out model selection.
- Split: the same 1,347 training IDs / 450 test IDs as the tiny model; fixed input division by 16. IDs and public-dataset hashes are in [model.json](model.json).
- Frozen reference accuracy: 444/450 (98.67%); not a verified writer-independent split or the original UCI evaluation protocol.
- [weights.npz](weights.npz) intentionally retains eight numeric arrays (four matrices and four bias vectors), not generated caches or executable serialization. Loading always uses `allow_pickle=False`; its SHA-256 is recorded in the metadata.
- [flow-selection.json](flow-selection.json) fixes the repeated experiment's rates/count/repeats and exploratory calibration hash. Flow tuning never changes weights or uses held-out labels for model selection.

Normal export/evaluation uses these exact frozen arrays, not retraining. The ONNX initializers are tested byte for byte against them; every successful HTTP result is compared with independent sklearn/NumPy references.

Optional retraining into a new ignored directory:

```sh
python tools/train_heavy_model.py --output examples/generated/retrained-heavy
```

The new model is trained locally from the attributed public dataset; it is not a downloaded third-party pretrained model. See [data attribution and license status](../../THIRD_PARTY_NOTICES.md). No license has yet been selected for original project contributions.
