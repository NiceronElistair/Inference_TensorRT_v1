BATCH_SIZE = 64

import numpy as np
PRECISION = np.float32

import torchvision.models as models
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

import torch

dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)

import torch.onnx
torch.onnx.export(resnext50_32x4d, dummy_input, "resnet50_onnx_model.onnx", verbose=False)