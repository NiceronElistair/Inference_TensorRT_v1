import torch
import tensorrt as trt
from collections import namedtuple, OrderedDict
import cv2
import numpy as np
import time
from utils import letterbox, scale_boxes, draw_bounding_boxes, non_max_suppression

weights = 'yolov5s.engine'
source = 0
webcam = True
out_file = './runs/detect/video.mp4'

# initialize the engine

device = torch.device('cuda')
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
context = model.create_execution_context()
bindings = OrderedDict()
output_names = []
fp16 = False
dynamic = False
for i in range(model.num_bindings):
    name = model.get_binding_name(i)
    dtype = trt.nptype(model.get_binding_dtype(i))
    if model.binding_is_input(i):
        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
            dynamic = True
            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
        if dtype == np.float16:
            fp16 = True
    else:  # output
        output_names.append(name)
    shape = tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
print('bindings addr', binding_addrs.items())
print('bindings images: ', binding_addrs['images'])
batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size


# Load the data 

img_size = 640 
stride = 32 
auto = False

cap = cv2.VideoCapture(source)
assert cap.isOpened(), f'Failed to open {source}'
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(out_file, four_cc, 20, (w, h)) 

def infer_one_frame(im):

    dynamic = False
    # je comprends rien de ce qui se passe à l'intérieur de ça par contre

    if dynamic and im.shape != bindings['images'].shape:
        i = model.get_binding_index('images')
        context.set_binding_shape(i, im.shape)  # reshape if dynamic
        bindings['images'] = bindings['images']._replace(shape=im.shape)
        for name in output_names:
            i = model.get_binding_index(name)
            bindings[name].data.resize_(tuple(context.get_binding_shape(i)))
    s = bindings['images'].shape
    assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    y = [bindings[x].data for x in sorted(output_names)]

    return y 


while True: 
    t1 = time.time()
    #load image
    _, im0 = cap.read()
    #pre processing of the image size
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to('cuda')
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    #perform image inference 
    y = infer_one_frame(im)

    # post processing of the output tensor
    # NMS 
    y = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=100)
    det = y[0]

    # rescaling of the bounding boxes depending on input im
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    

    im0 = draw_bounding_boxes(det, im0)

    out.write(im0)

    t2 = time.time()
    fps = 1/np.round(t2 - t1, 3)

    bottomLeftCornerOfText = (450, 50)
    cv2.putText(im0, f'FPS: {fps}', bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2)

    cv2.imshow('Frame', im0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    


