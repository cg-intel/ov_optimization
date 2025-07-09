from bmk_utils.benchmark_utils import latency_benchmark
import openvino as ov
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

@latency_benchmark(warmup=100, repeat=500)
def run_inference_torch():
    device = "GPU.0"
    precision = "FP16"
    input_shape = (1, 3, 640, 640)

    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_input)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()

    ov_model = ov.convert_model(model, example_input=dummy_tensor)
    core = ov.Core()

    config = {
        "INFERENCE_PRECISION_HINT": precision
    }

    compiled_model = core.compile_model(ov_model, device, config)
    input_name = compiled_model.input(0).get_any_name()
    infer_request = compiled_model.create_infer_request()

    def infer():
        infer_request.infer({input_name: dummy_input})

    return infer

if __name__ == "__main__":
    run_inference_torch()
