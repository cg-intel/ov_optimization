import openvino as ov
import numpy as np
import time
import openvino.properties.hint as hints
from bmk_utils.benchmark_utils import throughput_benchmark

@throughput_benchmark(warmup=10, repeat=100)
def run_throughput_infer():
    model_path = "models/resnet50.onnx"
    device = "GPU.0"
    input_shape = (1, 3, 640, 640)
    num_requests = 32
    precision = "FP16"

    core = ov.Core()
    model = core.read_model(model_path)
    config = {
        hints.inference_precision: precision,
        hints.performance_mode: hints.PerformanceMode.THROUGHPUT
    }
    compiled_model = core.compile_model(model, device, config)
    input_name = compiled_model.input(0).get_any_name()

    infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
    input_data = [np.random.rand(*input_shape).astype(np.float32) for _ in range(num_requests)]

    return infer_requests, input_data, input_name

if __name__ == "__main__":
    run_throughput_infer()