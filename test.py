import torch

if torch.xpu.is_available():
    print("XPU is available.")
    print(f"Device count: {torch.xpu.device_count()}")
    print(f"Device name: {torch.xpu.get_device_name(0)}")
else:
    print("XPU is NOT available.")