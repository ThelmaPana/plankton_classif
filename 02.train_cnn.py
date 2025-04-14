#!/usr/bin/env python
#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Train a CNN classifier
# Date: 27/03/2025
# Author: Emma Amblard & Thelma Panaïotis
#--------------------------------------------------------------------------#


import tensorflow_tricks  # settings for tensorflow to behave nicely

from os import makedirs
from os.path import dirname, join, isdir
import argparse

import pandas as pd
import numpy as np

from importlib import reload
import dataset            # custom data generator
import cnn                # custom functions for CNN generation
import biol_metrics       # custom functions model evaluation
dataset = reload(dataset)
cnn = reload(cnn)
biol_metrics = reload(biol_metrics)


# options to display all rows and columns for large DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

NON_BIOL_CLASSES_DICT = {
    'ifcb': ["other_living", "detritus", "other_living_elongated", "bad", "other_interaction", "spore"],
    'flowcam': ["dark", "detritus", "light", "lightsphere", "fiber", "lightrods", "contrasted_blob",
                "artefact", "darksphere", "darkrods", "ball_bearing_like", "other_living", "crumple sphere",
                "badfocus", "bubble", "dinophyceae_shape", "transparent_u"],
    'isiis': ["detritus", "streak", "other_living", "vertical line"],
    'zoocam': ["detritus", "fiber_detritus", "bubble", "light_detritus", "other_living", "artefact",
               "other_plastic", "medium_detritus", "gelatinous", "feces", "fiber_plastic"],
    'zooscan': ["detritus", "artefact", "fiber", "badfocus", "bubble", "other_egg", "seaweed", "Insecta", "other_living"],
    'uvp6': ["detritus", "fiber", "artefact", "reflection", "other<living", "dead<house", "darksphere", "t004", "t001"]
}

parser = argparse.ArgumentParser(description='Script to train a CNN model')

parser.add_argument('--dataset_name', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument("--folder_out", action="store", type=str, default=DEFAULT_MODEL_PATH, help="The path to the folder where to save the models and evaluation results")
parser.add_argument('--name_out', type=str, default='', help='The name of the file for saving results')
parser.add_argument('--cnn_backbone', type=str, default='mobilenet_v2_140_224', help='CNN architecture to use')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--fc_layer_size', type=str, default="600", help='size of the last(s) fc layer(s)') 
parser.add_argument('--weight_sensitivity', type=float, default=0.0, help='A scaling exponent that controls the impact of class weighting (0 for no weights, 1 for inverse weighting)') 
parser.add_argument('--non_biol_classes', type=str, default='', help='Labels of non biological classes (separated by commas) for evaluation')
parser.add_argument('--bottom_crop', type=int, default=0, help='Number of pixel to crop at bottom of images to remove scale bar')


args = parser.parse_args()

dataset_name = args.dataset_name
data_path = args.data_path
name_out = args.name_out
folder_out = args.folder_out
cnn_backbone = args.cnn_backbone
weight_sensitivity = args.weight_sensitivity


if args.non_biol_classes == '':
    non_biol_classes = NON_BIOL_CLASSES_DICT[dataset_name] if dataset_name in NON_BIOL_CLASSES_DICT else []
else:
    non_biol_classes = [label for label in args.non_biol_classes.split(',')]



print('Set options')

# directory to save training checkpoints
ckpt_dir = join(folder_out, 'models', name_out, 'checkpoints/')
#ckpt_dir = join(folder_out, 'checkpoints/')
makedirs(ckpt_dir, exist_ok=True)

# options for data generator (see dataset.EcoTaxaGenerator)
batch_size = args.batch_size
augment = True
upscale = True
bottom_crop = args.bottom_crop

# CNN structure (see cnn.Create and cnn.Compile)
model_handle_map = {
    "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
    "efficientnet_v2_S": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
    "efficientnet_v2_XL": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
}
fe_url = model_handle_map[cnn_backbone]
input_shape = (224, 224, 3)
fe_trainable = True
fc_layers_sizes = [int(x) for x in args.fc_layer_size.split(',')]
fc_layers_dropout = 0.4
classif_layer_dropout = 0.2

# CNN training (see cnn.Train)
lr_method = 'decay'
initial_lr = 0.0005
decay_rate = 0.97
epochs = args.epochs
workers = 10

print('Prepare datasets')

# read DataFrame with image ids, paths and labels
train_csv_path = join(data_path, 'train_labels.csv')
val_csv_path = join(data_path, 'valid_labels.csv')
test_csv_path = join(data_path, 'test_labels.csv')
df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)
df_test = pd.read_csv(test_csv_path)

# count nb of examples per class in the training set
class_counts = df_train.groupby('label').size()
class_counts

# list classes
classes = class_counts.index.to_list()

# generate categories weights
# i.e. a dict with format { class number : class weight }
if weight_sensitivity==0:
    class_weights = None
else:
    max_count = np.max(class_counts)
    class_weights = {}
    for idx,count in enumerate(class_counts.items()):
        class_weights.update({idx : (max_count / count[1])**weight_sensitivity})

# define numnber of  classes to train on
nb_of_classes = len(classes)

# define data generators
train_batches = dataset.EcoTaxaGenerator(
    images_paths=np.asarray([join(data_path, e) for e in df_train['img_path'].values]),
    input_shape=input_shape,
    labels=df_train['label'].values, 
    classes=classes,
    batch_size=batch_size, augment=augment, shuffle=True,
    crop=[0,0,bottom_crop,0])

val_batches = dataset.EcoTaxaGenerator(
    images_paths=np.asarray([join(data_path, e) for e in df_val['img_path'].values]),
    input_shape=input_shape,
    labels=df_val['label'].values, classes=classes,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])
# NB: do not shuffle or augment data for validation, it is useless
    
test_batches = dataset.EcoTaxaGenerator(
    images_paths=np.asarray([join(data_path, e) for e in df_test['img_path'].values]),
    input_shape=input_shape,
    labels=None, classes=None,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])

print('Prepare model')

# try loading the model from a previous training checkpoint
my_cnn, initial_epoch = cnn.Load(ckpt_dir)

# if nothing is loaded this means the model was never trained
# in this case, define it
if (my_cnn is not None) :
    print('  restart from model trained until epoch ' + str(initial_epoch))
else :
    print('  define model')
    # define CNN
    my_cnn = cnn.Create(
        # feature extractor
        fe_url=fe_url,
        input_shape=input_shape,
        fe_trainable=fe_trainable,
        # fully connected layer(s)
        fc_layers_sizes=fc_layers_sizes,
        fc_layers_dropout=fc_layers_dropout,
        # classification layer
        classif_layer_size=nb_of_classes,
        classif_layer_dropout=classif_layer_dropout
    )

    print('  compile model')
    # compile CNN
    my_cnn = cnn.Compile(
        my_cnn,
        initial_lr=initial_lr,
        lr_method=lr_method,
        decay_steps=len(train_batches),
        decay_rate=decay_rate,
    )

print('Train model') ## ----

# train CNN
history = cnn.Train(
    model=my_cnn,
    train_batches=train_batches,
    valid_batches=val_batches,
    epochs=epochs,
    initial_epoch=initial_epoch,
    log_frequency=1,
    class_weight=class_weights,
    output_dir=ckpt_dir,
    workers=workers
)

print('Evaluate model')

# load model for best epoch
best_epoch = None  # use None to get latest epoch
my_cnn, epoch = cnn.Load(ckpt_dir, epoch=best_epoch)
print(' at epoch {:d}'.format(epoch))

# predict classes for test dataset
pred, prob = cnn.Predict(
    model=my_cnn,
    batches=test_batches,
    classes=classes,
    workers=workers
)

# save prediction
pred_path = join(folder_out, 'predictions')
eval_path = join(folder_out, 'evaluations')
makedirs(pred_path, exist_ok=True)
makedirs(eval_path, exist_ok=True)

df_test = df_test[['objid', 'img_path', 'label']].copy() # keep only a few columns
df_test['predicted_label'] = pred
for i, label in enumerate(classes):
    df_test[label] = prob[:,i]
df_test.to_csv(join(pred_path, '{}.csv'.format(name_out)))

# compute a few scores
cr = biol_metrics.classification_report(
    y_true=df_test.label, 
    y_pred=df_test.predicted_label, 
    y_prob=prob,
    non_biol_classes = non_biol_classes
)
print(cr)
cr.to_csv(join(eval_path, '{}.csv'.format(name_out)))

# save model
my_cnn.save(join(folder_out, 'models', name_out), include_optimizer=False)
# NB: do not include the optimizer state: (i) we don't need to retrain this final
#     model, (ii) it does not work with the native TF format anyhow.

