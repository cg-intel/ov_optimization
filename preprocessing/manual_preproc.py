from bmk_utils.benchmark_utils import latency_benchmark
import openvino as ov
import numpy as np
import cv2
import time

@latency_benchmark(warmup=100, repeat=500)
def run_inference_manual_preproc():
    model_path = "../models/resnet50.onnx"
    device = "GPU.0"
    precision = "FP16"
    raw_shape = (1, 480, 640, 3)
    target_shape = (1, 3, 640, 640)

    raw_input = np.random.randint(0, 256, size=raw_shape, dtype=np.uint8)

    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": precision})
    input_name = compiled_model.input(0).get_any_name()
    infer_request = compiled_model.create_infer_request()

    def preprocess(image):
        image = image[0]
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = (image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        return image

    timings = {"pre": [], "infer": []}

    def infer():
        start = time.time()
        processed = preprocess(raw_input)
        mid = time.time()
        infer_request.infer({input_name: processed})
        end = time.time()
        
        timings["pre"].append((mid - start) * 1000)
        timings["infer"].append((end - mid) * 1000)

    return infer

if __name__ == "__main__":
    run_inference_manual_preproc()
