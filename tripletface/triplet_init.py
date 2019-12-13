"""triplet_init.py

The file generate a centroid and a treshold from all the persons in the dataset
with a few images of them
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import sys
import os

from tripletface.core.dataset import ImageFolder
from tripletface.core.model import Encoder
from torch.utils.data import DataLoader
from triplettorch import TripletDataset
from torchvision import transforms
from sklearn.manifold import TSNE


"""argparse

This part describes all the options the module can be executed with.
"""
parser        = argparse.ArgumentParser( )
parser.add_argument( '-s', '--dataset_path',  type = str,   required = True )
parser.add_argument( '-m', '--save_path',     type = str,   required = True )
parser.add_argument( '-i', '--input_size',    type = int,   default  = 224 )
parser.add_argument( '-z', '--latent_size',   type = int,   default  = 64 )
parser.add_argument( '-b', '--batch_size',    type = int,   default  = 32 )
parser.add_argument( '-e', '--epochs',        type = int,   default  = 10 )
parser.add_argument( '-l', '--learning_rate', type = float, default  = 1e-3 )
parser.add_argument( '-w', '--n_workers',     type = int,   default  = 4 )
parser.add_argument( '-r', '--n_samples',     type = int,   default  = 6 )
args          = parser.parse_args( )

dataset_path  = args.dataset_path
save_path    = args.save_path

input_size    = args.input_size
latent_size   = args.latent_size

batch_size    = args.batch_size
epochs        = args.epochs
learning_rate = args.learning_rate
n_workers     = args.n_workers
n_samples     = args.n_samples
noise         = 1e-2

"""trans

This part descibes all the transformations applied to the images for training
and testing.
"""
trans         = {
    'transforms':transforms.Compose( [
        transforms.Resize( size = input_size ),
        transforms.CenterCrop( size = input_size ),
        transforms.ToTensor( ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] )
}

"""folder

This part descibes all the folder dataset
"""
folder        = {
    'train': ImageFolder( os.path.join( dataset_path, 'train' ), trans[ 'transforms' ] )
}


"""samples

This part select 32 images randomly for each persons in the dataset
"""
nb_peoples    = len(set(folder['train'].labels))
samples       = {}
batch_sample  = 64

for i in range(nb_peoples):
    samples[i] = []
    while len(samples[i]) < batch_sample :
        pos = random.randint(0, len( folder[ 'train' ]) - 1)
        samples[i].append(pos) if folder[ 'train' ].labels[pos] == i else None

"""model initialization

This part define the model and load his weights.
"""

print("Loading model & weight...\n")
model = Encoder(64).cuda() # embeding_size = 64
weight = torch.load("model.pt")['model']
model.load_state_dict(weight)
print("Model & weight loaded\n")
fig = plt.figure( figsize = ( 8, 8 ) )
ax  = fig.add_subplot( 111 )

for mad in range(nb_peoples):
    """dataset

    This part describes all the triplet dataset to pass inside the model.
    We still used TripletDataset for output format
    """
    dataset       = {
        'train': TripletDataset(
            np.array( folder[ 'train' ].labels ),
            lambda i: folder[ 'train' ][ samples[mad][i] ][ 1 ],
            batch_sample,
            1
        )
    }

    """loader

    This part descibes all the dataset loaders.
    """
    loader        = {
        'train': DataLoader( dataset[ 'train' ],
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = n_workers,
            pin_memory  = True
        )
    }


    """ forward

    This part give the samples to the dataset in order to generate a tensor
    """
    test_embeddings = [ ]
    test_labels     = [ ]

    for b, batch in enumerate( loader[ 'train' ] ):
        labels, data    = batch
        labels          = torch.cat( [ label for label in labels ], axis = 0 )
        data            = torch.cat( [ datum for datum in   data ], axis = 0 )

        embeddings      = model( data.cuda() ).detach( ).cpu( ).numpy( )

        test_embeddings.append( embeddings )
        test_labels.append( labels )


    test_embeddings = np.concatenate( test_embeddings, axis = 0 )
    test_labels     = np.concatenate(     test_labels, axis = 0 )

    if latent_size > 2:
        test_embeddings = TSNE( n_components = 2 ).fit_transform( test_embeddings )

    ax.clear( )
    ax.scatter(
        test_embeddings[ :, 0 ],
        test_embeddings[ :, 1 ],
        c = test_labels
    )


    """Calculus & save

    This part calculate the treshold ans centroid and save them with a figure of
    the tensor
    """
    centroid  = []
    tresholds = 0
    for i in range(latent_size):
        crazy_night = 0
        for j in range(batch_size):
            crazy_night += embeddings[j,i]
            tresholds += embeddings[j,i]
        centroid.append(crazy_night/batch_size)
    tresholds = abs(tresholds * 0.005) # Chuuuuut, c'est Steven qui m'as dit que ça serait un bon chiffre

    fd = os.open("peoples_data/peoples_data.txt", os.O_APPEND|os.O_RDWR) # Steven c'est la peluche, elle me parle, la plante verte aussi
    line = str.encode(f'People {mad}: \nTresholds = {tresholds}\nCentroid = {centroid}\n\n')
    os.write(fd, line)
    os.close(fd)

    fig.canvas.draw( ) # y a Zackarie qui ronfle, y dort alors que je bosse, le bataaaaaaaard !
    fig.savefig( os.path.join( save_path, f'peoples_vizualisation_{mad}.png' ) )

    print(f'people {mad} done\n')
