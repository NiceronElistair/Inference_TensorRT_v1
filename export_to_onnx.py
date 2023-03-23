import argparse
 
import torch
import torch.nn as nn
from models.experimental import attempt_load
import json

def export_onnx(model, im, save_name, verbose, dynamic, opset):
    print(im.shape)

    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}

    torch.onnx.export(model,
                       im, 
                       save_name, 
                       input_names=['images'], 
                       output_names= ['output'], 
                       verbose=verbose,
                       opset_version= opset,
                       export_params= True, 
                       dynamic_axes=dynamic or None
                       )
    
    model.eval()
    
    # model_onnx = onnx.load(save_name)
    # onnx.checker.check_model(model_onnx) 
    # onnx.save(model_onnx, save_name)

    return save_name

def export_torchscript(model, im, file):
    # YOLOv5 TorchScript model export
    print('start torchscript export')
    f = file.spplit('.')[0] + '.torchscript'

    ts = torch.jit.trace(model, im, strict=False)
    d = {'shape': im.shape, 'stride': int(max(model.stride)), 'names': model.names}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()

    ts.save(str(f), _extra_files=extra_files)
    return f, None

def run(
        weights='yolov5n.pt',
        save_name='yolov5n.onnx',
        verbose=True, 
        dynamic = True,
        opset = 9
        ):

    #load model
    device = 'cuda'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    batch_size = 1
    
    #check
    imgsz = (640, 640)
    imgsz *= 2 if len(imgsz) == 1 else 1

    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetectio

    model.eval()
    
    for _ in range(2):
        y = model(im)  # dry runs


    f = export_onnx(model, im, save_name, verbose, dynamic, opset)
    #check_onnx(f)

def check_onnx(model):
    assert model.split('.')[-1] == 'onnx'
    
    model= onnx.load(model)
    print(model)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights you chose to convert')
    parser.add_argument('--save-name', type=str, help='file name for the converted weights')
    parser.add_argument('--verbose', action='store_true', help='verbose log')
    parser.add_argument('--dynamic', action='store_true', help='dynamic mode')
    parser.add_argument('--opset', type=int, help='opset version of the export function. La version par défaut de la jetson est la 9')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

