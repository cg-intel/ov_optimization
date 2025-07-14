import openvino as ov

onnx_model = "resnet50.onnx"
ir_xml_fp16 = "resnet50_fp16.xml"
ir_xml_fp32 = "resnet50_fp32.xml"

ov_model = ov.convert_model(onnx_model)
ov.save_model(ov_model, ir_xml_fp16, compress_to_fp16=True)

ov.save_model(ov_model, ir_xml_fp32, compress_to_fp16=False)