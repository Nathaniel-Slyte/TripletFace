"""inference.py

Script to use our previous CNN
"""

import numpy as np
import torch
import cv2

from tripletface.core.model import Encoder
from torchvision import transforms
from PIL import Image

"""model initialization

This part define the model and load his weights.
"""
print("Loading model & weight...\n")
model = Encoder(64).cuda() # embeding_size = 64
weight = torch.load("model.pt")['model']
model.load_state_dict(weight)
print("Model & weight loaded\n")

"""trans

This part descibes all the transformations applied to the images for training
and testing.
"""
trans   = transforms.Compose( [
        transforms.Resize( size = 299 ),
        transforms.CenterCrop( size = 299 ),
        transforms.ToTensor( ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] )
