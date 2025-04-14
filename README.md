# plankton_classif

Benchmark for plankton images classifications methods for images from multiple plankton imaging devices (IFCB, FlowCam, ISIIS, ZooCam, Zooscan, UVP6).

This code was used to run the comparison between multiple Convolutional Neural Network architectures and a Random Forest classifier in the paper **Benchmark of plankton images classification: emphasizing features extraction over classifier complexity**.

## Data

### Instruments

The comparison is to be done on data from multiple plankton imaging devices:

-   IFCB (Imaging FlowCytobot)
-   FlowCAM
-   ISIIS (In Situ Ichthyoplankton Imaging System)
-   ZooCam
-   ZooScan
-   UVP6 (Underwater Vision Profiler)

### Data source

All datasets, except the IFCB one, can be found on SeaNoe:

-   FlowCAM: <https://www.seanoe.org/data/00908/101961/>

-   ISIIS: <https://www.seanoe.org/data/00908/101950/>

-   UVP6: <https://www.seanoe.org/data/00908/101948/>

-   ZooCam: <https://www.seanoe.org/data/00907/101928/>

-   ZooScan: <https://www.seanoe.org/data/00446/55741/>

The IFCB dataset can be found at <https://darchive.mblwhoilibrary.org/handle/1912/7341>

### Data preparation

Store your input data in `data/<instrument_name>` (by default, the `data` folder is in `../io/data`). Data preparation is handled by `00.prepare_datasets.py` for the SeaNoe dataset: it reads `features_native.csv.gz` as well as `taxa.csv.gz` to prepare 3 new csv files ready for model training: `train_labels.csv`, `valid_labels.csv` and `test_labels.csv` with one row per object. These csv files contain the following columns:

-   `img_path`: path to image
-   `label`: object classification
-   `features_1` to `features_n`: object features for random forest fit (choices for names of these columns are up to you)

## Classification models

This code can be used to train two types of classification models: Convolutional Neural Networkd (CNN) and Random Forests (RF).

In both cases, training is done in two phases:

-   the model is optimized by training on the training set (defined by the `train_labels.csv` file in our case) and evaluating on the validation set (`valid_labels.csv`)
-   the optimized model is evaluated on the test set (`test_labels.csv`) never used before

### Random Forest

#### Description

A random forest takes a vector of features as input and predicts a class from these values.

#### Training

Random Forests can be trained using `01.grid_search_rf.py`. Parameters are optimized with a gridsearch including:

-   number of trees
-   number of features to use to compute each split (default for classification is sqrt(n_features))
-   minimum number of samples required to be at a leaf node (default for classification is 5)

For each set of parameters, a model is trained on training data and evaluated on test data. For the purpose of the paper, only the number of trees was searched for, similarly is fine-tuning the number of epochs when training a CNN. Parameters to try should be stored in a `json` file, the path of which is to be given as an argument.

#### Output

After training, an output directory is created (by default `./grid_search_classifier_results`) and results are stored in this folder for each RF trained with grid search.

### Convolutional Neural Network

#### Description

A convolutional neural network takes an image as input and predicts a class for this image.

The CNN backbone can be :

-   a MobileNetV2 feature extractor (<https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>)
-   an EfficientNet V2 S feature extractor (<https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2>)
-   an EfficientNet V2 XL feature extractor (<https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2>)

A classification head with the number of classes to predict is added on top of the backbone. Intermediate fully connected layers with customizable dropout rate can be inserted between both.

Input images are expected to have colour values in the range [0,1] and a size of 224 x 224 pixels. If need be, images are automatically resized by the `EcoTaxaGenerator`.

#### Training

The CNN can be trained with `02.train_cnn.py`. For each step (i.e. epoch) in the training of the CNN model, the model is trained on training data and evaluated on validation data. Last saved weights are then used to test the model on the test data.

#### Output

After training, an output directory is created (by default `../io/models`) and results are stored in a subfolder with the model name.

The CNN can also be used as a deep features extractor with the following scripts:

-   `03.convert_to_feature_extractor.py` converts the model to a feature extractor
-   `04.train_dimensionality_reduction.py` trains a PCA to apply to the output of the feature extractor
-   `05.extract_deep_features.py` uses the feature extractor to extract deep features and optionally applies PCA.

All results of theses steps are saved in the same subfolder. The deep features (with or without PCA reduction) can then be used to train a RF  classifier.

## Results

After all trainings are performed, classification performance are assessed:

-   `06.write_metrics.py` computes all classification metrics and writes them into `./perf/prediction_metrics.csv`
-   `07.write_classification_report.py` generates all classification reports as csv files into `./perf`
-   `08.generate_plots.R` reads the content of `./perf/prediction_metrics.csv` and generates figures for the paper, saved in `./figures`
