import argparse

def export_engine(file, half, save_name, workspace=4):
    import tensorrt as trt

    print(trt.__version__)

    onnx = file

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(0)]

    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f' output "{out.name}" with shape{out.shape} {out.dtype}')

    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(save_name, 'wb') as t:
        t.write(engine.serialize())
    return save_name, None

def run(
        weights='yolov5n.onnx',
        save_name='yolov5n.trt'
):
    half = False

    f, _= export_engine(weights, half, save_name )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights you chose to convert')
    parser.add_argument('--save-name', type=str, help='file name for the converted weights')   
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)