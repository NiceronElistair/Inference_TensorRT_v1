import torch
import torch.nn as nn
from models.experimental import attempt_load

def export_onnx(model, im):
    f = 'yo.onnx'
    output_names = ['output0']
    torch.onnx.export(model, im, f, input_names=['images'], output_names=output_names)
    return f

def run():

    #load model
    device = 'cuda'
    weights = 'yolov5n.pt'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    batch_size = 1
    
    #check
    imgsz = (640, 640)
    imgsz *= 2 if len(imgsz) == 1 else 1

    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    print(im.shape, type(im))

    model.eval()
    for _ in range(2):
        y = model(im)  # dry runs


    f = export_onnx(model, im)



def main():
    run()

if __name__ == "__main__":
    main()

