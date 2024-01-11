# MDCJ

This folder contains various scripts produced during the project. These 
include various scripts to run a multitude of CNNs. Additionally, a few utility
scripts can be found, which were produced to aid in the processing of data
before any of the CNNs scripts were used to train the models.

## Table of contents

* [requirements](#requirements)
* [preprocessing data](#preprocessing-data)
* [AlexNet](#AlexNet)
* [Debugging InceptionV3](#Debugging-InceptionV3)
* [Debugging running DenseNet121](#Debugging-running-DenseNet121)
* [DenseNet121](#DenseNet121)
* [InceptionV3](#InceptionV3)
* [InceptionV3 epoch loading](#InceptionV3-epoch-loading)
* [ResNet50](#ResNet50)
* [ShuffleNet](#ShuffleNet)
* [auto resort](#auto-resort)
* [file sorting](#file-sorting)
* [magnification extraction](#magnification-extraction)
* [multiprocess extraction](#multiprocess-extraction)
* [CNN input parameters](#CNN-input-parameter-csv-file)
* [CNN output](#CNN-output)

## Requirements
All models were run and trained on an in-house anaconda environment.
This environment mirrors the freely [pytorch2 environment](https://pytorch.org/get-started/pytorch-2.0/#getting-started). Additionally,
[wandb.ai](https://wandb.ai/site) is required for logging the training and validation data produced 
by the CNNs

## Preprocessing data
This folder contains multiple scripts that were used in preprocessing the
data before running the CNNs. These include patch extraction scripts,
Macenko normalisation, colour augmentation, k-fold cross validation, and
oversampling.

## AlexNet
This script will train AlexNet models on the input dataset. By modifying the
[parameter csv file](#CNN-input-parameter-csv-file), the hyperparameters can be tuned in addition to
determining the number of trials and epochs the CNN should run for.
Results will automatically be logged to wandb.ai

AlexNet can be run in the pytorch2 environment by running the command 
below:
```
python AlexNet.py
```

## Debugging InceptionV3
This version of InceptionV3 was used to debug the initial CNN script in
order to train the models and allow for a dynamic design, which has
allowed for a faster adaptation of other CNNs used.

This version can still contain bugs, and it is not recommended to use this
script to train InceptionV3 models.

## Debugging running DenseNet121
This version of DenseNet121 is designed to run the models created by the
[DenseNet121](#DenseNet121) script. Where the last few layers of the DenseNet model are
manually modified to allow the created models to be newly loaded 
and create predictions on novel datasets.

This script is currently still in development. 

## DenseNet121
This script will train DenseNet121 models on the input dataset. By modifying the
[parameter csv file](#CNN-input-parameter-csv-file), the hyperparameters can be tuned in addition to
determining the number of trials and epochs the CNN should run for.
Results will automatically be logged to wandb.ai

DenseNet121 can be run in the pytorch2 environment by running the 
command below:
```
python DenseNet121.py
```

## InceptionV3
This script will train InceptionV3 models on the input dataset. This script
was developed before the other CNN scripts were produced. New
changes to the models will be tested on InceptionV3, before they will are
adapted to the other CNN scripts.

By modifying the [parameter csv file](#CNN-input-parameter-csv-file), the hyperparameters can be tuned in
addition to determining the number of trials and epochs the CNN
should run for. Results will automatically be logged to wandb.ai

InceptionV3 can be run in the pytorch2 environment by running the 
command below:
```
python InceptionV3.py
```

## InceptionV3 epoch loading
This script will train InceptionV3 models on the input dataset. This script
was developed to load all training data used for a trial before training,
instead of loading and unloading the data used to train one epoch. This
method of loading data saved up to 30 seconds when training without
colour augmentation, and up to 200 seconds when training on data with
colour augmentation. Results are based on training InceptionV3 with 
20.000 images using an RTX 4090. 

Development of this script was discontinued in favour of the old design, 
due to time constrained and persistent issues in properly loading, ordering,
of input data. In addition to logging of metrics to wandb and simplifying
the CNN script for future bug fixing and improvements.

## ResNet50
This script will train ResNet50 models on the input dataset. By modifying the
[parameter csv file](#CNN-input-parameter-csv-file), the hyperparameters can be tuned in addition to
determining the number of trials and epochs the CNN should run for.
Results will automatically be logged to wandb.ai

ResNet50 can be run in the pytorch2 environment by running the 
command below:
```
python ResNet50.py
```

## ShuffleNet
This script will train ShuffleNet models on the input dataset. By modifying the
[parameter csv file](#CNN-input-parameter-csv-file), the hyperparameters can be tuned in addition to
determining the number of trials and epochs the CNN should run for.
Results will automatically be logged to wandb.ai

ShuffleNet can be run in the pytorch2 environment by running the 
command below:
```
python ShuffleNet.py
```

## Auto resort
This script automatically filters files out of folders into a single destination folder, 
based on previously sorted files with identical names.
Paths and suffixes should be changed by the user based on their own
folder structure and file names

Example:

Data folder:
```
	.
	├── ...
	├── main_dir               
	│	├── Unique_id_1.img				
	│	├── Unique_id_2.img
	│	├── Folder_01
	│	│	├── Subfolder_01				
	│	│	│	├── Unique_id_3.img
	│	│	│	└── ...
	│	│	├── Unique_id_4.img
	│	│	├── Unique_id_5.img			
	│	│	└── ...
	│	├── Folder_02					
	│	│	├── Subfolder_01				
	│	│	│	├── Unique_id_6.img
	│	│	│	├── Unique_id_7.img
	│	│	│	├── Unique_id_8.img
	│	│	│	└── ...
	│	│	└── ...
	│	└── ...
	└── ...
```

Previously sorted files:
```
Unique_id_1.img
Unique_id_3.img
Unique_id_7.img
Unique_id_8.img
```

Output folder result:
```
Unique_id_1.img
Unique_id_3.img
Unique_id_7.img
Unique_id_8.img
```

## File sorting
Sorts files with unique identifiers into their own folder. The script looks for 
the unique identifier right after the first underscore.

Example:

Data folder:
```
	.
	├── ...
	├── File_id_1_suffix.img
	├── File_id_1_suffix.img
	├── File_id_1_suffix.img
	├── File_id_2_suffix.img
	├── File_id_2_suffix.img
	├── File_id_3_suffix.img
	├── File_id_3_suffix.img
	├── File_id_3_suffix.img
	├── File_id_3_suffix.img
	├── File_id_4_suffix.img
	├── File_id_6_suffix.img
	├── File_id_6_suffix.img
	├── File_id_6_suffix.img             
	└── ...
```

Result:
```
	.
	├── ...
	├── id_1               
	│	├── File_id_1_suffix.img
	│	├── File_id_1_suffix.img
	│	├── File_id_1_suffix.img
	│	└── ...
	├── id_2               
	│	├── File_id_2_suffix.img
	│	├── File_id_2_suffix.img
	│	└── ...
	├── id_3
	│	├── File_id_3_suffix.img
	│	├── File_id_3_suffix.img
	│	├── File_id_3_suffix.img
	│	├── File_id_3_suffix.img
	│	└── ...
	├── id_4
	│	├── File_id_4_suffix.img
	│	└── ...
	├── id_6               
	│	├── File_id_6_suffix.img
	│	├── File_id_6_suffix.img
	│	├── File_id_6_suffix.img
	│	└── ...
	└── ...
```

## Magnification extraction
This script was developed to automatically extract a set of images of a
lower magnification(l-mag), by using the corner coordinates found in the name of a
higher magnification(h-mag) image.

The script takes three inputs, where the inner folder structure of the folders
do not matter and will be replicated by the script
- One input folder containing lower magnification images to be extracted
- One output folder, which can be created by the script
- One folder containing images of higher magnification

It is assumed that the folder containing higher magnification images only 
contains images with suffixes that match those of the images in the input
folder. Additionally, the script assumes that the X and Y coordinates
presented in the file name are from the top left corner.

The script will first grab the X and Y coordinates from a h-mag image.
Next, the coordinates of the other three corners are calculated, using the
presented height and width in pixels. After all coordinates, have been
determined, the script will walk through all l-mag images and
one by one calculate the corner coordinates and check if all four h-mag coordinates are 
present within the l-mag coordinates. If all coordinates are present, the
l-mag image is moved to the output folder. If one or more coordinates do
not fit, the image will stay within the input folder.


## Multiprocess extraction
This version of magnification extraction utilises multiprocessing to
significantly improve extraction time compared to [Magnification extraction](#magnification-extraction).
Additionally, a second folder containing h-mag images must now be
defined. This folder functions the same as the first h-mag folder, with the
change being that only two of the four coordinates need to be present in
the l-mag image for it to be extracted.


## CNN input parameter csv file
Here, an example version of the parameter .csv file is shown that is
required as input for the CNNs. A description of each variable is also
present within the .csv file

| Variable            | Value                                    | Description                                                                                                                                                               |
|---------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| root_dir            | D:\path\to\root_dir                      | Path to local directory, containing a training and validation folder for training CNNs                                                                                    |
| xlsx_path           | D:\path\to\binary.xlsx                   | Path to local xlsx file, containing two columns. The first column must have all unique study IDs, the second columns the binary classes for majority (1) and minority (0) |
| train_dirname       | Training                                 | The name of the folder in root_dir, containing the images for training the model                                                                                          |
| val_dirname         | Validation                               | The name of the folder in root_dir, containing the images for validating the model                                                                                        |
| wandb_name          | wandb_project_name                       | Name of the project that will be stored on wandb                                                                                                                          |
| wandb_save          | D:\path\to\local_wandb_save_folder       | Path to local directory where models and CNN data will be stored that is logged on wandb                                                                                  |
| model_param_csv     | D:\path\to\CNN_model_hyperparameters.csv | Name of the csv file where the model will be stored along with the hyperparameters to create said model                                                                   |
| dataload_workers    | 3                                        | Number of multiprocessing workers to be used for the dataloaders (3 workers was determined to be optimal for an Intel(R) I9-13900K                                        |
| accumulation_steps  | 4                                        | Accumulation step size to be taken during training and validation                                                                                                         |
| num_epochs          | 50                                       | Number of epochs a model should be trained for                                                                                                                            |
| num_trials          | 100                                      | Number of trials the code should run for, where one trial equals one model                                                                                                |
| es_counter          | 0                                        | Start value of the early stop counter                                                                                                                                     |
| es_limit            | 15                                       | Value at which early stop is trigged and a trial will be terminated                                                                                                       |
| tl_loss_rate        | 1e-4;1e-3;1e-2                           | Loss rate values to be chosen at random by the CNN during training                                                                                                        |
| tl_batch_norm       | True;False                               | Allow for batch normalization for the entire trial. Statements chosen at random at the start of a trial                                                                   |
| tl_dropout_rate     | 0;0.1;0.2;0.5                            | Dropout rate values to be chosen at random for each trial                                                                                                                 |
| tl_batch_size       | 256                                      | Images are processed in batches during training and validation. batch_size determines the number of images to be included into a batch                                    |
| tl_weight_decay_min | 1.00E-05                                 | Minimum weight decay to be used during training                                                                                                                           |
| tl_weight_decay_max | 1.00E-01                                 | Maximum weight decay to be used during training. Must be higher than tl_weight_decay_min. Weight decay will be chosen at random within the given range                    |
| tl_gamma_min        | 0.1                                      | Minimum gamma to be used during training                                                                                                                                  |
| tl_gamma_max        | 0.9                                      | Maximum gamma to be used during training. Must be higher than tl_gamma_min. Gamma value will be chosen at random within the given range at the start of a trial           |
| tl_gamma_step       | 0.1                                      | Gamma step size                                                                                                                                                           |

## CNN output

