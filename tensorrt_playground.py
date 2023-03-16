from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

import tensorrt as trt

f = open("resnet_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
