#!/usr/bin/env python
#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Prepare datasets from Seanoe format
# Date: 27/03/2025
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

import pandas as pd
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prepare datasets from Seanoe format")
parser.add_argument("data_dir", type=str, help="Directory to raw dataset")
parser.add_argument("output_dir", type=str, help="Directory to save prepared datasets")
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

# Make sure that output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read native features
print("Reading native features")
df_all = pd.read_csv(f"{data_dir}/features_native.csv.gz")

# Read taxonomic labels
labels = pd.read_csv(f"{data_dir}/taxa.csv.gz")[['objid', 'set', 'taxon_level1', 'img_path']]
labels.rename(columns={'taxon_level1': 'label'}, inplace=True)

# Join labels with features
df_all = df_all.merge(labels, on='objid', how='left')

# Reorder columns
col_to_order = ['set', 'objid', 'img_path', 'label']
cols = col_to_order + [col for col in df_all.columns if col not in col_to_order]
df_all = df_all[cols]

# Split into training, validation, and test sets
df_train = df_all[df_all['set'] == 'train'].drop(columns=['set'])
df_valid = df_all[df_all['set'] == 'valid'].drop(columns=['set'])
df_test = df_all[df_all['set'] == 'test'].drop(columns=['set'])

# Write to CSV
print("Writing data splits")
df_train.to_csv(f"{output_dir}/train_features.csv", index=False)
df_valid.to_csv(f"{output_dir}/valid_features.csv", index=False)
df_test.to_csv(f"{output_dir}/test_features.csv", index=False)
