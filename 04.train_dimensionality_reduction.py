#!/usr/bin/env python
#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Train a dimensionality reduction model for deep features
# Date: 08/04/2025
# Author: Emma Amblard & Thelma Panaïotis
#--------------------------------------------------------------------------#

import tensorflow_tricks  # settings for tensorflow to behave nicely

from os.path import dirname, join
import argparse

import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

import dataset   # custom data generator


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

parser = argparse.ArgumentParser(description='Script to train a model')

parser.add_argument('--dataset_name', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model folder containing the feature extractor')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--n_dim', type=int, default=50, help='Number of dimensions to keep after dimensionality reduction')
parser.add_argument('--mode', type=str, default='default', help='Which type of feature extractor to use: resize or default')
parser.add_argument('--bottom_crop', type=int, default=0, help='Number of pixel to crop at bottom of images to remove scale bar')


args = parser.parse_args()

dataset_name = args.dataset_name
data_path = args.data_path
model_name = args.model_name
model_path = join(args.model_path, 'models', model_name)
bottom_crop = args.bottom_crop

print('Set options')

batch_size = args.batch_size  # size of images batches in GPU memory
workers = 10                  # number of parallel threads to prepare batches
n_dims = args.n_dim           # number of dimensions to keep after dimensionality reduction

print('Load feature extractor')

fe_name = 'feature_extractor' if args.mode == 'default' else 'feature_extractor_{}'.format(args.mode)
my_fe = tf.keras.models.load_model(join(model_path, fe_name), compile=False)

# get model input shape
input_shape = my_fe.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)


print('Load data and extract features for the training set')

# read DataFrame with image ids, paths and labels
# NB: those would be in the database in EcoTaxa
train_csv_path = join(data_path, 'train_labels.csv')
df = pd.read_csv(train_csv_path)

# prepare data batches
batches = dataset.EcoTaxaGenerator(
    images_paths=np.asarray([join(data_path, e) for e in df['img_path'].values]),
    input_shape=input_shape,
    labels=None,
    classes=None,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])


# extract features by going through the batches
features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)

print('Fit dimensionality reduction')

# define the PCA
pca = PCA(n_components=n_dims)
# fit it to the training data
pca.fit(features)

# save it for later application
with open(join(model_path, 'dim_reducer.pickle'), 'wb') as pca_file:
    pickle.dump(pca, pca_file)
