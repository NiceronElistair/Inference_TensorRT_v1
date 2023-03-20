import argparse
 
import torch
import torch.nn as nn
from models.experimental import attempt_load

def export_onnx(model, im, save_name):
    f = 'yo.onnx'
    torch.onnx.export(model,
                       im, 
                       save_name, 
                       input_names=['images'], 
                       output_names= ['output0'], 
                       verbose=False, 
                       dynamic_axes= False,
                       opset_version=17, 
                       do_constant_folding=False
                       )
    return f

def run(
        weights='yolov5n.pt',
        save_name='yolov5n.onnx'
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


    f = export_onnx(model, im, save_name)

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

