from bmk_utils.benchmark_utils import latency_benchmark
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

@latency_benchmark()
def run_inference_xpu():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")
    input_shape = (1, 3, 640, 640)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)

    def infer():
        with torch.no_grad():
            _ = model(dummy_input)

    return infer

if __name__ == "__main__":
    run_inference_xpu()
