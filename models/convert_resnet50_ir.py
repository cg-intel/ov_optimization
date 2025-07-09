import openvino as ov

onnx_model = "resnet50.onnx"
ir_xml = "resnet50.xml"

ov_model = ov.convert_model(onnx_model)
ov.save_model(ov_model, ir_xml, compress_to_fp16=True)