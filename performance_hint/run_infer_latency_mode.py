from bmk_utils.benchmark_utils import latency_benchmark
import openvino as ov
import numpy as np
import openvino.properties.hint as hint

@latency_benchmark()
def run_inference_onnx():
    model_path = "../models/resnet50.onnx"
    device = "GPU.0"
    precision = "FP16"
    input_shape = (1, 3, 640, 640)

    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    core = ov.Core()

    config = {
        hint.inference_precision: precision,
        hint.performance_mode: hint.PerformanceMode.LATENCY
    }

    compiled_model = core.compile_model(model_path, device, config)
    input_name = compiled_model.input(0).get_any_name()
    infer_request = compiled_model.create_infer_request()

    def infer():
        infer_request.infer({input_name: dummy_input})

    return infer

if __name__ == "__main__":
    run_inference_onnx()