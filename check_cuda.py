import torch

device = torch.cuda.is_available()
if device:
    print('cuda is available')
else:
    print('cuda is not available')