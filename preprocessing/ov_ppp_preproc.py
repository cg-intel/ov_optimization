from bmk_utils.benchmark_utils import latency_benchmark
import openvino as ov
import numpy as np
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino import Layout, Type
import openvino.properties.hint as hint

@latency_benchmark()
def run_inference_ov_preproc():
    model_path = "../models/resnet50.onnx"
    device = "GPU.0"
    precision = "FP16"
    input_shape = (1, 480, 640, 3)

    raw_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

    core = ov.Core()
    model = core.read_model(model_path)

    ppp = PrePostProcessor(model)
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.BGR) \
        .set_shape(list(input_shape))

    ppp.input().model().set_layout(Layout('NCHW'))

    ppp.input().preprocess() \
        .convert_element_type(Type.f32) \
        .convert_color(ColorFormat.RGB) \
        .resize(ResizeAlgorithm.RESIZE_LINEAR) \
        .mean([123.675, 116.28, 103.53]) \
        .scale([58.395, 57.12, 57.375])

    model_with_preproc = ppp.build()

    ov.serialize(model_with_preproc, "../models/resnet50_with_preproc.xml")

    config = {hint.inference_precision: precision}
    compiled_model = core.compile_model(model_with_preproc, device, config)
    input_name = compiled_model.input(0).get_any_name()
    infer_request = compiled_model.create_infer_request()

    def infer():
        infer_request.infer({input_name: raw_input})

    return infer

if __name__ == "__main__":
    run_inference_ov_preproc()
