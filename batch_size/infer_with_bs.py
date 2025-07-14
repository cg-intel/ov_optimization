import openvino as ov
import numpy as np
import openvino.properties.hint as hint
from bmk_utils.benchmark_utils import throughput_benchmark

model_path = "../models/resnet50.onnx"
device = "GPU"
image_shape = (3, 640, 640)
num_requests = 16

def batch_size_runner(batch_size):
    @throughput_benchmark(batch_size=batch_size)
    def run():
        core = ov.Core()
        config = {
            hint.performance_mode: hint.PerformanceMode.THROUGHPUT
        }

        model = core.read_model(model_path)
        model.reshape({model.input(0).get_any_name(): [batch_size, *image_shape]})
        compiled_model = core.compile_model(model, device, config)
        input_name = compiled_model.input(0).get_any_name()

        infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        input_data = [
            np.random.rand(batch_size, *image_shape).astype(np.float32)
            for _ in range(num_requests)
        ]
        return infer_requests, input_data, input_name
    return run

if __name__ == "__main__":
    for batch_size in [1, 2, 4, 8, 16]:
        print(f"\nBatch Size = {batch_size}")
        runner = batch_size_runner(batch_size)
        runner()

