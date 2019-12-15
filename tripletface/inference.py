"""inference.py

Script to use our previous CNN
"""

import numpy as np
import argparse
import torch

from tripletface.core.model import Encoder
from torchvision import transforms
from PIL import Image

"""argparse

This part describes all the options the module can be executed with.
"""
parser        = argparse.ArgumentParser( )
parser.add_argument( '-s', '--dataset_path',  type = str,   required = True )
parser.add_argument( '-i', '--input_size',    type = int,   default  = 224 )
args          = parser.parse_args( )

dataset_path  = args.dataset_path
input_size    = args.input_size

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


"""compute

This part push the image throught the model and output the embedding 
"""
img  = Image.open( dataset_path ).convert( 'RGB' )
X    = trans( img )
X    = torch.unsqueeze(X,0)

embeddings      = model( X.cuda() ).detach( ).cpu( ).numpy( )

print(embeddings)
