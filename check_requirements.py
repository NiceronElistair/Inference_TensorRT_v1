import torch
import tensorrt as trt

print(f'tensorrt is available with {trt.__version__}')

device = torch.cuda.is_available()
if device:
    print('cuda is available')
else:
    print('cuda is not available')