import openvino as ov
import time
import shutil
import os
import openvino.properties as props

model_path = "../models/resnet50.onnx"
device_name = "GPU"
cache_dir = "cache"


def compile_with_cache(enable_cache: bool):
    core = ov.Core()

    if enable_cache:
        print("Enabling model cache...")
        core.set_property({props.cache_dir: cache_dir})
    else:
        print("Disabling model cache...")

    start = time.time()
    _ = core.compile_model(model=model_path, device_name=device_name)
    end = time.time()

    elapsed = (end - start) * 1000
    print(f"Compile time ({'with' if enable_cache else 'without'} cache): {elapsed:.2f} ms")
    return elapsed


if __name__ == "__main__":

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print("\nFirst run (no cache):")
    compile_with_cache(enable_cache=False)

    print("\nSecond run (with cache - first time, create cache):")
    compile_with_cache(enable_cache=True)

    print("\nThird run (with cache - cache already exists):")
    compile_with_cache(enable_cache=True)
