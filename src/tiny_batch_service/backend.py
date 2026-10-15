"""A single spawned CPU worker. Only the dispatcher can submit IPC work."""

import asyncio
import multiprocessing
import os
import platform
import resource
import sys
import time


def worker(connection, model):
    try:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[key] = "1"
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        if session.get_providers()[0] != "CPUExecutionProvider":
            raise ValueError("CPU registration failed")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "X" or inputs[0].type != "tensor(float)"
                or inputs[0].shape != ["N", 64] or len(outputs) != 1 or outputs[0].name != "probabilities"):
            raise ValueError("Model must implement the fixed MVP tensor contract")
        # Identical warm-up shapes in both modes; outside measured HTTP trials.
        for rows in (1, 32):
            session.run(None, {"X": np.zeros((rows, 64), dtype=np.float32)})
        def highwater():
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value if sys.platform == "darwin" else value * 1024)
        connection.send({"ready": True, "pid": os.getpid(), "python": platform.python_version(),
                         "numpy": np.__version__, "onnxruntime": ort.__version__,
                         "provider": "CPUExecutionProvider", "intra_threads": 1, "inter_threads": 1,
                         "optimization": sys.flags.optimize, "high_water_rss_bytes": highwater()})
        while True:
            request = connection.recv()
            if request is None:
                return
            array = np.asarray(request, dtype=np.float32)
            started = time.perf_counter_ns()
            output, = session.run(None, {"X": array})
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if output.shape != (len(array), 10) or not np.isfinite(output).all():
                raise ValueError("Invalid output")
            connection.send({"outputs": output.tolist(), "inference_ms": elapsed,
                             "high_water_rss_bytes": highwater()})
    except Exception:
        try:
            connection.send({"error": "worker_execution_failed"})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ProcessBackend:
    def __init__(self, model, timeout=10.):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, str(model)), daemon=True)
        self.timeout = timeout
        self.process.start()
        child.close()
        self.metadata = None
        self.highwater = 0

    def _receive(self):
        if not self.connection.poll(self.timeout):
            self.process.terminate()
            self.process.join(2)
            raise RuntimeError("Worker response timed out")
        response = self.connection.recv()
        if response.get("error"):
            raise RuntimeError("Worker execution failed")
        return response

    async def ready(self):
        self.metadata = await asyncio.to_thread(self._receive)
        if self.metadata.get("ready") is not True:
            raise RuntimeError("Worker did not initialize")
        self.highwater = self.metadata["high_water_rss_bytes"]
        return self.metadata

    async def infer(self, values):
        def exchange():
            self.connection.send(values)
            return self._receive()
        response = await asyncio.to_thread(exchange)
        self.highwater = max(self.highwater, response["high_water_rss_bytes"])
        return response

    async def close(self):
        def stop():
            if self.process.is_alive():
                try:
                    self.connection.send(None)
                except (OSError, EOFError):
                    pass
                self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            self.connection.close()
        await asyncio.to_thread(stop)
