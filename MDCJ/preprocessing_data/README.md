# Preprocessing data
This folder contains several scripts that aid in processing image data, before the CNN scripts are used to train the 
models.

## Table of Contents
* [Auto exclusion](#auto-exclusion)
* [Colour augmentation](#colour-augmentation)
* [Multiprocess colour augmentation](#multiprocess-colour-augmentation)
* [Kfold crossvalidation](#kfold-cross-validation)
* [Marchenko normalization](#Marchenko-normalization)
* [Mask filtering](#mask-filtering)
* [Oversampling](#oversampling)
* [Parameter loading](#parameter-loading)
* [Patch extraction](#patch-extraction)
* [Default parameters csv](#default-parameters)
* [binary classification xlsx](#binary-classification-xlsx)

## Auto exclusion
This folder contains the various debug versions that were used to create the 
[magnification extraction](https://github.com/mdcjansen/EMC_Bladder/blob/main/MDCJ/mag_extraction.py) and 
[multiprocess extraction](https://github.com/mdcjansen/EMC_Bladder/blob/main/MDCJ/multi_mag_extraction.py)scripts.

## Colour augmentation
The first version of augmenting input images. These images are augmented in batches of ten images at any given time. The
brightness, contrast, saturation, and hue are changed by a set value, which can be changed by the user in line 39. This 
script was developed to allow for direct adaptation into the CNN codes, once images could successfully be colour 
augmented. The current version is no longer present in the CNN scripts. However, this script will function on its own. 
All input images present within the input folder, will be colour augmented and saved as a separate image within the 
output folder.

## Multiprocess colour augmentation
This multiprocessing version of colour augmentation is incorporated into the CNN scripts. Several changes have been made
in comparison to the [colour augmentation](#colour-augmentation) script.

Firstly, a range of values now has to be provided instead of a fixed value for augmentation. Providing a broader range 
of augmented images. Secondly, specific image formats can be provided for processing. By default, only '.jpg' files are 
processed. Lastly, current time will be given at each print. Allowing for simplistic time keeping, when each step has 
been performed.

## Kfold cross validation
Based on user input, multiple kfold cross validation folds and a test set are produced. By default, a test set is 
created that contains 30% of the input data. Additionally, four kfold cross validated folds are produced.

The script requires and input folder, containing the images to create folds for, an output folder that will contain the 
folds as well as the test set, and the [binary label file](#binary-classification-xlsx). This file is used to maintain 
the same class distribution in the folds, as is present in the input folder.

## Marchenko normalization
Marchenko normalization is applied on the input images. The normalization is based off a reference image, which should 
not be part of the dataset that will be used to train the CNNs. To perform the normalization, an input folder, output 
folder, and reference image must be provided by the user. The input images are not modified during normalization, 
instead normalized images are saved as new '.jpg' images.

## Mask filtering
This script filters the input images, based on their accompanying mask size. Images that have a masked size percentage 
lower than the input value, are moved to the output folder. By default, images with less than 75% mask coverage will be 
filtered out. The script requires an input, output, and a value between 0.0 and 1.0 for the mask size threshold 
(default= 0.75).

## Oversampling
This script will perform oversampling on a set of images to remove class imbalance from two classes. Oversampling is 
performed by randomly selecting input images and created copies that are horizontally, vertically, or hybrid flipped. 
Where images with a hybrid flip are both horizontally and vertically flipped. This script requires an input folder which 
contains the images to be oversampled. As well as an output folder and the 
[binary classification file](#binary-classification-xlsx).

## Parameter loading
This debug script was used to test the functionality of loading all input parameters for the CNNs through a 
[.csv file](#default-parameters). This script has been adapted into the CNN scripts to handle loading the parameters. 
Currently, this script will load the parameters into the correct data format that is required by the CNN scrips. 
all parameters can be changed, used, or printed as needed by the user.

## Patch extraction
This groovy script is used to extract the images used by the CNN from a larger image. The user has to provide at which 
magnification the extraction is performed, as well as the output folder, image format, and image dimensions. Only images
that have a accompanying label image with a partially or completely coloured background (non-white) will be extracted 
and saved in the output folder.

## Default parameters
Here, an example version of the parameter .csv file is shown that is required as input for the CNNs. A description of 
each variable is also present within the .csv file

| Variable            | Value                                    | Description                                                                                                                                                                                              |
|---------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| root_dir            | D:\path\to\root_dir                      | Path to local directory, containing a training and validation folder for training CNNs                                                                                                                   |
| xlsx_path           | D:\path\to\binary.xlsx                   | Path to local [xlsx file](#binary-classification-xlsx), containing two columns. The first column must have all unique study IDs, the second columns the binary classes for majority (1) and minority (0) |
| train_dirname       | Training                                 | The name of the folder in root_dir, containing the images for training the model                                                                                                                         |
| val_dirname         | Validation                               | The name of the folder in root_dir, containing the images for validating the model                                                                                                                       |
| wandb_name          | wandb_project_name                       | Name of the project that will be stored on wandb                                                                                                                                                         |
| wandb_save          | D:\path\to\local_wandb_save_folder       | Path to local directory where models and CNN data will be stored that is logged on wandb                                                                                                                 |
| model_param_csv     | D:\path\to\CNN_model_hyperparameters.csv | Name of the csv file where the model will be stored along with the hyperparameters to create said model                                                                                                  |
| dataload_workers    | 3                                        | Number of multiprocessing workers to be used for the dataloaders (3 workers was determined to be optimal for an Intel(R) I9-13900K                                                                       |
| accumulation_steps  | 4                                        | Accumulation step size to be taken during training and validation                                                                                                                                        |
| num_epochs          | 50                                       | Number of epochs a model should be trained for                                                                                                                                                           |
| num_trials          | 100                                      | Number of trials the code should run for, where one trial equals one model                                                                                                                               |
| es_counter          | 0                                        | Start value of the early stop counter                                                                                                                                                                    |
| es_limit            | 15                                       | Value at which early stop is triggered and a trial will be terminated                                                                                                                                    |
| tl_loss_rate        | 1e-4;1e-3;1e-2                           | Loss rate values to be chosen at random by the CNN during training                                                                                                                                       |
| tl_batch_norm       | True;False                               | Allow for batch normalization for the entire trial. Statements chosen at random at the start of a trial                                                                                                  |
| tl_dropout_rate     | 0;0.1;0.2;0.5                            | Dropout rate values to be chosen at random for each trial                                                                                                                                                |
| tl_batch_size       | 256                                      | Images are processed in batches during training and validation. batch_size determines the number of images to be included into a batch                                                                   |
| tl_weight_decay_min | 1.00E-05                                 | Minimum weight decay to be used during training                                                                                                                                                          |
| tl_weight_decay_max | 1.00E-01                                 | Maximum weight decay to be used during training. Must be higher than tl_weight_decay_min. Weight decay will be chosen at random within the given range                                                   |
| tl_gamma_min        | 0.1                                      | Minimum gamma to be used during training                                                                                                                                                                 |
| tl_gamma_max        | 0.9                                      | Maximum gamma to be used during training. Must be higher than tl_gamma_min. Gamma value will be chosen at random within the given range at the start of a trial                                          |
| tl_gamma_step       | 0.1                                      | Gamma step size                                                                                                                                                                                          |

## Binary classification xlsx
This Excel file contains two columns which detail which study id has which binary label. The IDs and labels were chosen 
at random.

| study_id | binary_label |
|----------|--------------|
| 001      | 1            |
| 078      | 0            |
| 376      | 1            |

