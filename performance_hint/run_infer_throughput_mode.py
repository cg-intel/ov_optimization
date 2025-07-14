import openvino as ov
import numpy as np
import openvino.properties.hint as hint
from bmk_utils.benchmark_utils import throughput_benchmark

@throughput_benchmark()
def run_throughput_infer():
    model_path = "../models/resnet50.onnx"
    device = "GPU.0"
    input_shape = (1, 3, 640, 640)
    precision = "FP16"

    core = ov.Core()
    config = {
        hint.inference_precision: precision,
        hint.performance_mode: hint.PerformanceMode.THROUGHPUT
    }
    compiled_model = core.compile_model(model_path, device, config)

    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print(f"Using {num_requests} inference requests")

    input_name = compiled_model.input(0).get_any_name()
    infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]

    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    input_data = [dummy_input] * num_requests

    return infer_requests, input_data, input_name

if __name__ == "__main__":
    run_throughput_infer()