import openvino as ov
import time
import shutil
import os
import torch
import torchvision.models as models
import openvino.properties as props

onnx_path = "models/wide_resnet101_2.onnx"
device_name = "GPU"
cache_dir = "cache"

def export_torch_model_to_onnx():
    if not os.path.exists(onnx_path):
        print("Exporting wide_resnet101_2 to ONNX...")
        model = models.wide_resnet101_2(weights="DEFAULT").eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, (dummy_input,), onnx_path,
            input_names=["images"], output_names=["output"],
            opset_version=12, do_constant_folding=True
        )
        print(f"ONNX model saved to {onnx_path}")
    else:
        print("ONNX model already exists.")


def compile_with_cache(enable_cache: bool):
    core = ov.Core()

    if enable_cache:
        print("Enabling model cache...")
        core.set_property({props.cache_dir: cache_dir})
    else:
        print("Disabling model cache...")

    start = time.time()
    _ = core.compile_model(onnx_path, device_name)
    end = time.time()

    elapsed = (end - start) * 1000
    print(f"Compile time ({'with' if enable_cache else 'without'} cache): {elapsed:.2f} ms")
    return elapsed


if __name__ == "__main__":
    export_torch_model_to_onnx()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print("\nFirst run (no cache):")
    compile_with_cache(enable_cache=False)

    print("\nSecond run (with cache - first time, create cache):")
    compile_with_cache(enable_cache=True)

    print("\nThird run (with cache - cache already exists):")
    compile_with_cache(enable_cache=True)