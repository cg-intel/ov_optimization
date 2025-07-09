import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()

dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    (dummy_input,),
    "resnet50.onnx",
    input_names=["images"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True
)
