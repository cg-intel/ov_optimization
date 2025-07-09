import openvino as ov

core = ov.Core()

for device in core.available_devices:
    print(device)
    properties = core.get_property(device, "FULL_DEVICE_NAME")
    print(properties)
    opt_precision = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
    print(opt_precision)