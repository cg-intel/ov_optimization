import openvino as ov
import numpy as np
import openvino.properties.hint as hint
from bmk_utils.benchmark_utils import throughput_benchmark

model_path = "../models/resnet50.onnx"
device = "GPU"
input_shape = (1, 3, 640, 640)
num_requests = 16

def manual_nstream_runner(nstream):
    @throughput_benchmark()
    def run():
        core = ov.Core()
        config = {
            "NUM_STREAMS": str(nstream),
            hint.performance_mode: hint.PerformanceMode.THROUGHPUT
        }
        compiled_model = core.compile_model(model_path, device, config)
        input_name = compiled_model.input(0).get_any_name()

        infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        input_data = [np.random.rand(*input_shape).astype(np.float32)] * num_requests
        return infer_requests, input_data, input_name
    return run

def auto_nstream_runner2():
    @throughput_benchmark()
    def run():
        core = ov.Core()
        config = {
            hint.performance_mode: hint.PerformanceMode.THROUGHPUT
        }
        compiled_model = core.compile_model(model_path, device, config)
        nstream = compiled_model.get_property("NUM_STREAMS")
        print(f"\nAUTO N Streams =  {nstream}")
        input_name = compiled_model.input(0).get_any_name()

        infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        input_data = [np.random.rand(*input_shape).astype(np.float32)] * num_requests
        return infer_requests, input_data, input_name
    return run

if __name__ == "__main__":
    for nstream in [1, 2, 4, 8, 16, 32]:
        print(f"\nN Streams =  {nstream}")
        runner = manual_nstream_runner(nstream)
        runner()

    runner = auto_nstream_runner2()
    runner()
