from compilation.run_infer_onnx import run_inference_onnx
from compilation.run_infer_ir import run_inference_ir
from compilation.run_infer_torch import run_inference_torch
from compilation.run_infer_xpu import run_inference_xpu

def compare_all():
    print("\n=== ONNX ===")
    run_inference_onnx()

    print("\n=== IR ===")
    run_inference_ir()

    print("\n=== Torch ===")
    run_inference_torch()

    print("\n=== Torch XPU ===")
    run_inference_xpu()

if __name__ == "__main__":
    compare_all()
