# Algorithm that takes as input the video stream of the detected camaera on the computer
# and as an outout create a window that stream the webcam with a bounding box surrounding 
# objects that are detected on the images, with there label and the confidence score 
# associated

import torch
import tensorrt as trt
from collections import namedtuple, OrderedDict
import cv2
import numpy as np
import time
from uutils import letterbox, scale_boxes, draw_bounding_boxes, non_max_suppression, infer_one_frame, select_camera

# Path to the weights of the neural network file with tensorRT format
weights = 'yolov5n.trt'
# webcan flag
webcam = True
# path to save the output
out_file = './video.mp4'

# In order to load the tensorRT file, we need to create what is called an engine

# select the device used for runnung the algo, either CPU or GPU, cuda stands for GPU utilisation
device = torch.device('cuda')
# Initialisation of the dictionnary that contains the input (images from video), and output tensor of the deep learning network
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)                            #return log at execution

# Read and deserialize the tensorRT file 
with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())

context = model.create_execution_context() 
bindings = OrderedDict()      
output_names = []
fp16 = False                # numerical format of the weigths
dynamic = False             # don't know what is that

# go through all the inputs 
for i in range(model.num_bindings):                         
    name = model.get_binding_name(i)   # get name of the binding, either image of output 
    dtype = trt.nptype(model.get_binding_dtype(i))  # get dtype of binfin
    if not model.binding_is_input(i):               
        output_names.append(name)
    shape = tuple(context.get_binding_shape(i))     # get binding shape 
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)            # get binding array data convefrt it to tensor and send to GPU device
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))      # fill the dictionnary create before



# Load the data 
# Variables responsible for input reshape
img_size = 640 
cap = select_camera()  # create the camera object
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # get width frame  
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # get height frame
four_cc = cv2.VideoWriter_fourcc(*"MJPG")      
out = cv2.VideoWriter(out_file, four_cc, 20, (w, h)) # create a recordered to save the camera stream


#inference time
while True:                 # process frame one by one
    t1 = time.time()
    #load image 
    _, im0 = cap.read()        # read frame
    
    # resize the frame to fit the input size necessary for the neural network
    im = letterbox(im0, img_size)[0]  # size the input image
    im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to('cuda')    # to tensor, to cuda
    im = im.float()  # to fp32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # inference on one frame
    y = infer_one_frame(im, model, bindings, context, output_names)  # return a tensor that contain coordinates of bounding box, label probability and confience score

    y = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45) # apply non max suppression minimize redundancy of some binding box
    det = y[0] # list of list to list

    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # bounding box coord are not adapted to the im0 frame, need to rescale the coord
    
    im0 = draw_bounding_boxes(det, im0) # draw boundinx box dans print label name and confidence score

    out.write(im0)  # write frame for save

    t2 = time.time()
    fps = 1/np.round(t2 - t1, 3) # compute frame rate

    bottomLeftCornerOfText = (450, 50)
    cv2.putText(im0, f'FPS: {fps}', bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2) # print frame rate on frame

    cv2.imshow('Frame', im0)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # destroy window
        break

cap.release()
out.release()
cv2.destroyAllWindows()

    


