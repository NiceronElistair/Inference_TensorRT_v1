import torch
import cv2
import time 
import torchvision
import numpy as np
from collections import OrderedDict

NAME_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
                9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
                16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
                24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
                31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
                37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
                52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
                67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def select_camera():
    """
    If a camera logitec is pluged on the computer choose the camera pluged
    If no camera is pluged, chose the webcam of the laptop
    """
    source = 0
    cap = cv2.VideoCapture(source)
    return cap

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    """
    Need to resize the input frame shape so that is can fit the entry 
    of the neurol network entry
    THe goal is to resize without any deformation. Thus, Ratio is calculated between the desired shape and input shape, 
    then there is a first resize depending on the calculated. Then we create a black border that surround the image
    to have the right size of our matrix 
    """
    shape = im.shape[:2]  # input shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # dif width and heigth

    dw /= 2  # divide padding into 2 sides to have the image center in relation to the black border
    dh /= 2

    if shape[::-1] != new_unpad:  # have a first resize 
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))        
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border to get the right shape
    return im, ratio, (dw, dh)

def infer_one_frame(im, model, bindings, context, output_names):
    """
    take as input a deserialized model, an input image with the right size, 
    bindings that containes informations about the input shape, dtype, address
    and also a contexte that I don't understand yet. 

    As output the function return à tensor for each detected object on the image
    that ccontaine :

    [x,y,w,h,C,p(c_1), ....... p(c_N)] where,

    (x,y) are the coord of the center of the bounding box relatively to the SxS grid
    (w,h) is the size of the bounding box relatively to the image 
    C confidence score
    p(c_n) probability of the object relative to the class n
    """
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())          # d.ptr stands for the address of the actual data tensor

    dynamic = False # don't know what is this

    
    s = bindings['images'].shape
    assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))  #line that actually run the inference ans store the result in output_names 
    y = [bindings[x].data for x in sorted(output_names)]  #get the result value

    return y 

def non_max_suppression(prediction,conf_thres=0.25, iou_thres=0.45):
    """
    Execute the non max suppression algorithm:
    infer_one_frame returns for one object many bounding boxes. There is a redundancy problem that 
    non max suppression algo solves. The goal is to eliminate the redundant boxes in order to have 
    one box per actual object. 
    It takes as input the tensor of size (batch_size, nb_boxes_detected, 85)
    85 because there are x, y, w, h C and 80 probabilities

    then we return a tensor of size (batch_size, nb_boxes_detected_without_redundancy, 6)
    6 because x1, y2, x1, y2, C and one class proba
    """
    
    if isinstance(prediction, (list, tuple)): 
        prediction = prediction[0]  

    device = prediction.device

    bs = prediction.shape[0]  # number of images infered, for our usecase it is juste one image at the time
    nc = prediction.shape[2] - 5  # number of classes that our model can predicte
    xc = prediction[..., 4] > conf_thres  # return a tensor with only boolean, false if the box has a confidence score below conf_thres

    
    max_wh = 7680  # Je commprends pas encore très bien pourquoi ce truc existe, ça me semble etre du bricolage des dev... 

    output = [torch.zeros((0, 6), device=device)] * bs # initialisation of the output tensor 

    for xi, x in enumerate(prediction):  # in our case there is only one iteration because our batch size = 1
        
        x = x[xc[xi]]                                   # #  Apply the boolean tensor to eliminate all the boxes from x that are under conf_thres                                 
        x[:, 5:] *= x[:, 4:5]                           # conf = obj_conf * cls_proba
        box = xywh2xyxy(x[:, :4])
        print(x.shape)
        if x.shape[0] != 0:                       # convert x,y,w,h system into x1, y1, x2, y2 system
            conf, j = x[:, 5:].max(1, keepdim=True)         # among the the 80 classes proba, return the max with it's corresponding confidence score 
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]        # reduce the size of the tensor from 85 to 6 by removing the class proba of the 79 classes 
            x = x[x[:, 4].argsort(descending=True)[:]]      # sort by confidence and remove excess boxes from x

            # Batched NMS
            c = x[:, 5:6] * max_wh  # Applying this strange mnipulation that I don't understand I think it is an offset 
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # takes box coord, and score, calculate the box_iou with all the boxes from the same class, and then return index of the best boxes j'vais surement coder cette fonction moi même parce qye telecharger torchvision juste poyr ça c'est chiant
            output[xi] = x[i]

    return output

def box_iou(box1, box2, eps=1e-7):   # a redefinir, j'en ai plus besoin sauf si je réécris la fonction de torchvision
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def draw_bounding_boxes(det, im0):
    """
    Draw the binding box and add the texte of the label 
    """

    xyxy = det[:, :4]       # get the coord of the bounding boxes
    conf = det[:, 4]        # get confidence scores
    c = det[:, 5]           # get labels as int
    list_label = [[NAME_CLASSES[int(c[i])], "%.2f" % float(conf[i])] for i in range(len(c))]        # return list of all the label detected but as string

    for i in range(len(det)):       # For each detected object 
        box = xyxy[i]               # get coord bounding box 
        label = str(list_label[i][0] + ' ' + list_label[i][1])              #get readable label 
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))     
        color=(255, 0, 0)         #color for bouding box
        txt_color=(255, 255, 255)   # color for texte
        cv2.rectangle(im0, p1, p2, color, thickness=3, lineType=cv2.LINE_AA) #draw bounding box
        tf = 1
        w, h = cv2.getTextSize(label, 0, fontScale=3 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled bounding box to make it prettier
        cv2.putText(im0,                                                            # print label on the image 
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),   # label and coord of the print
                    0,
                    3 / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    return im0
