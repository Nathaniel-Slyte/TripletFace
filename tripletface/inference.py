"""inference.py

Script to use our previous CNN
"""

import numpy as np
import torch
import cv2

from tripletface.core.model import Encoder
from torchvision import transforms
from PIL import Image

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
cap.set(5, 15)


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
trans         = transforms.Compose( [
        transforms.Resize( size = 299 ),
        transforms.CenterCrop( size = 299 ),
        transforms.ToTensor( ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] )


while True :
    ret, frame = cap.read()
    cv2.imread('frame')

    frame = trans(frame)
    input = torch.randn((1, 3, 299, 299)).float()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
