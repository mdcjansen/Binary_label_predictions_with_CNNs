#!/usr/bin/env python3

import csv
import numpy as np
import logging
import optuna
import os
import pandas as pd
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import torchvision.transforms as transforms
import wandb

from collections import defaultdict
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve, \
    auc, confusion_matrix
# from time import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import sigmoid
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import densenet121
# from torchvision.datasets import DatasetFolder

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.0"
__date__ = "23/11/2023"

# file paths
param_path = r"path\to\param\file.csv"
model_path = r"path\to\model.pt"


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, xlsx_dir, transform=None):
        # print("Initializing CustomImageDataset...")  # Debug statement
        self.root_dir = root_dir
        self.transform = transform
        self.df_labels = pd.read_excel(xlsx_dir)

        # Precompute the list of all image paths
        self.all_images = [os.path.join(subdir, file)
                           for subdir, dirs, files in os.walk(self.root_dir)
                           for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Debug: Print the first few image paths
        # print(f"First few image paths: {self.all_images[:5]}")  # Debug statement

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # print(f"__getitem__ called for index: {idx}")  # Debug statement
        img_name = self.all_images[idx]

        # Extract study_id from image name
        segments = img_name.split('_')

        if len(segments) <= 1:
            raise ValueError(f"Unexpected image name format for {img_name}")

        study_id = segments[1]

        # Debug: Print the extracted study_id
        # print(f"Extracted study_id: {study_id}")  # Debug statement

        # Get label from Excel
        label_entries = self.df_labels[self.df_labels['study_id'] == int(study_id)]['label'].values

        # Debug: Print the label entries from the DataFrame
        # print(f"Label entries for study_id {study_id}: {label_entries}")  # Debug statement

        if len(label_entries) == 0:
            print(f"No label found for study_id: {study_id} in image {img_name}.")  # Debug statement
            # logging.warning(f"No label found for study_id: {study_id} in image {img_name}.")
            label = None
        else:
            label = label_entries[0]

        # Load image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, study_id


class CustomHead(nn.Module):
    def __init__(self, input_size, output_size, batch_norm, dropout_rate):
        super(CustomHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

        # Conditionally create the BatchNorm1d layer
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)

        x = self.fc(x)

        # Apply dropout after linear layer (since there's no activation function here)
        x = self.dropout(x)

        return x


def load_parameters(param_path):
    input_param = {}
    with open(param_path, 'r') as param_file:
        details = csv.DictReader(param_file)
        for d in details:
            input_param.setdefault(d['variable'], []).append(d['value'])
        param_file.close()

    input_variables = {
        'root_dir': ''.join(input_param['root_dir']),
        'xlsx_dir': ''.join(input_param['xlsx_path']),
        'test_dir': ''.join(input_param['test_dirname']),
        'wdb_name': ''.join(input_param['wandb_name']),
        'wdb_save': ''.join(input_param['wandb_save']),
        'log_lvl': ''.join(input_param['log_level']),
        'dl_work': int(''.join(input_param['dataload_workers'])),
        'a_steps': int(''.join(input_param['accumulation_steps'])),
        'num_e': int(''.join(input_param['num_epochs'])),
        'num_t': int(''.join(input_param['num_trials'])),
        'es_count': int(''.join(input_param['es_counter'])),
        'es_limit': int(''.join(input_param['es_limit'])),
        'tl_lr': list(map(float, ''.join(input_param['tl_loss_rate']).split(';'))),
        'tl_bn': ''.join(input_param['tl_batch_norm']).split(';'),
        'tl_dr': list(map(float, ''.join(input_param['tl_dropout_rate']).split(';'))),
        'tl_bs': list(map(int, ''.join(input_param['tl_batch_size']).split())),
        'tl_wd_min': float(''.join(input_param['tl_weight_decay_min'])),
        'tl_wd_max': float(''.join(input_param['tl_weight_decay_max'])),
        'tl_ga_min': float(''.join(input_param['tl_gamma_min'])),
        'tl_ga_max': float(''.join(input_param['tl_gamma_max'])),
        'tl_ga_tp': float(''.join(input_param['tl_gamma_step']))
    }

    return input_variables


def get_class_counts(dataset):
    print("Counting classes for dataset...")  # Debug statement
    # logging.info("Counting classes for dataset")
    class_counts = defaultdict(int)
    for idx, (img, label, _) in enumerate(dataset):  # Notice the added underscore
        try:
            class_counts[label] += 1
            if idx == len(dataset) - 1:
                print(f"Last index processed: {idx}")  # Debugging line
                # logging.info(f"Last index processed: {idx}")
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")  # Debug statement
            # logging.error(f"Error processing image at index {idx}: {e}")
    return class_counts


def calculate_metrics(y_true, y_pred, y_proba):
    # Assuming y_true and y_pred are already numpy arrays or Python lists. If they are tensors, convert them only once.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }

    # Calculate precision, recall, and F1 score, handling any exceptions
    for metric, func in [('f1', f1_score), ('precision', precision_score), ('recall', recall_score)]:
        try:
            metrics[metric] = func(y_true, y_pred)
        except:
            pass  # Handle the exception or pass in this context means to ignore the error and proceed

    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    # Calculate the ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    metrics['roc_auc'] = auc(fpr, tpr)

    # Attempt to calculate the confusion matrix
    try:
        metrics['cm'] = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")
        # logging.error(f"Error computing confusion matrix: {e}")
        metrics['cm'] = None

    return metrics  # Return the metrics dictionary directly


def log_metrics(metrics, split, prefix, loss):
    # Log metrics using a dictionary with W&B
    wandb.log({
        f"{prefix}_{split}_loss": loss,
        f"{prefix}_{split}_acc": metrics['acc'],
        f"{prefix}_{split}_f1": metrics['f1'],
        f"{prefix}_{split}_balacc": metrics['bal_acc'],
        f"{prefix}_{split}_recall": metrics['recall'],
        f"{prefix}_{split}_precision": metrics['precision'],
        f"{prefix}_{split}_cnfmatrix": metrics['cm'],
        f"{prefix}_{split}_auc": metrics['roc_auc']
    })

    # logging.info("wandb metric logging:\n"
    # f"{prefix}_{split}_loss: {loss}\n"
    # f"{prefix}_{split}_acc: {metrics['acc']}\n"
    # f"{prefix}_{split}_f1: {metrics['f1']}\n"
    # f"{prefix}_{split}_balacc: {metrics['bal_acc']}\n"
    # f"{prefix}_{split}_recall: {metrics['recall']}\n"
    # f"{prefix}_{split}_precision: {metrics['precision']}\n"
    # f"{prefix}_{split}_cnfmatrix: {metrics['cm']}\n"
    # f"{prefix}_{split}_auc: {metrics['roc_auc']}\n"
    # )


## DEFINE MODEL PATH
def load_model(model_path, batch_norm, dropout_rate):
    model = densenet121(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False

    state_dict = torch.load(model_path, map_location=device)

    # Modify the main classifier
    num_ftrs = model.fc.in_features
    model.fc = CustomHead(num_ftrs, 1, batch_norm, dropout_rate)

    # Modify the auxiliary classifier
    aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_ftrs, 1)

    # new_state_dict = {}
    #    if key.startswith("classifier.fc.weight"):
    #        new_key = key.replace("classifier.fc.weight", "classifier.weight")
    #        new_state_dict[new_key] = value
    #    elif key.startswith("classifier.fc.bias"):
    #        new_key = key.replace("classifier.fc.bias", "classifier.bias")
    #        new_state_dict[new_key] = value
    #    else:
    #        new_state_dict[key] = value

    # for key, value in state_dict.items():
    #    print(key)

    # model.load_state_dict(new_state_dict)

    model.eval()
    model = model.to(device)

    return model, "DenseNet121"


def predict(model, data_loader, device):
    print("Starting prediction...")  # Debug statement
    # logging.info("Starting prediction")

    batch_predictions = []
    batch_labels = []
    study_summary = {}
    y_proba = []

    with torch.no_grad():
        for i, (inputs, labels, study_ids) in enumerate(data_loader):
            # print(f"Predicting batch {i+1}/{len(data_loader)}...")  # Debug statement
            # logging.info(f"Predicting batch {i+1}/{len(data_loader)}")
            inputs = inputs.to(device)
            outputs = model(inputs)

            output = outputs.view(-1)
            probs = torch.sigmoid(output).detach().cpu().numpy()
            y_proba.extend(probs)

            batch_predictions.extend((probs > 0.5).astype(int).tolist())
            batch_labels.extend(labels.tolist())

            # Process predictions by study_id
            for prediction, probability, label, study_id in zip(batch_predictions, probs, batch_labels, study_ids):
                if study_id not in study_summary:
                    study_summary[study_id] = {'predictions': [], 'probs': [], 'labels': []}
                study_summary[study_id]['predictions'].append(prediction)
                study_summary[study_id]['probs'].append(probability)
                study_summary[study_id]['labels'].append(label)

            # Clear memory if needed
            del inputs, outputs, labels, study_ids
            torch.cuda.empty_cache()

        # Calculate patient-level predictions
        patient_predictions = []
        patient_probabilities = []
        patient_labels = []
        for study_id, summaries in study_summary.items():
            mode_prediction, _ = stats.mode(summaries['predictions'])
            mean_probability = np.mean(summaries['probs'])
            mode_label, _ = stats.mode(summaries['labels'])

            patient_predictions.append(mode_prediction[0])
            patient_probabilities.append(mean_probability)
            patient_labels.append(mode_label[0])

        print("Prediction complete.")  # Debug statement
        # logging.info("Prediction complete")
        return batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probabilities


def objective(trial):
    print(f"Starting trial {trial.number}...")  # Debug statement

    # Initialize Weights & Biases run
    run = wandb.init(project=parameters['wdb_name'], config={})

    # Save the script being run to Weights & Biases
    wandb.save(parameters['wdb_save'])

    config = run.config
    trial_lr = trial.suggest_categorical('lr', parameters['tl_lr'])
    trial_batch_norm = trial.suggest_categorical('batch_norm', parameters['tl_bn'])
    trial_dropout_rate = trial.suggest_categorical('dropout_rate', parameters['tl_dr'])

    # Configuration setup
    run.config.update({
        "lr": trial_lr,
        "batch_size": trial.suggest_categorical('batch_size', parameters['tl_bs']),
        "num_epochs": parameters['num_e'],
        "weight_decay": trial.suggest_float('weight_decay', parameters['tl_wd_min'], parameters['tl_wd_max'], log=True),
        "step_size": 5,
        "gamma": trial.suggest_float('gamma', parameters['tl_ga_min'], parameters['tl_ga_max'],
                                     step=parameters['tl_ga_tp']),
        "batch_norm": trial_batch_norm,
        "dropout_rate": trial_dropout_rate
    }, allow_val_change=True)

    # logging.info("Config setup:\n"
    #             f"lr: {trail_lr}\n"
    #             f"batch_size: {trail.suggest_categorial('batch_size', [256])}\n"
    #             f"num_epochs: {parameters['num_e']}\n"
    #             f"weight_decay: {trail.suggest_float('weight_decay', 1e-5, 1e-1, log=True)}\n"
    #             f"step_size: 5\n"
    #             f"gamma: {trail.suggest_float('gamma', 0.1, 0.9, step=0.1)}\n"
    #             f"batch_norm: {trail_batch_norm}\n"
    #             f"dropout_rate: {trail_dropout_rate}\n"
    #             )

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Create DataLoaders
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                  num_workers=parameters['dl_work'], pin_memory=True)

    ## LOAD EXISTING MODEL
    model, model_name = load_model(model_path, config["batch_norm"], config["dropout_rate"])

    # Move model to the device (CPU/GPU)
    model.to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = GradScaler()

    # Training and validation loop

    print("Start prediction")  # Debug statement
    # logging.debug(f"Epoch {epoch + 1} - Start prediction")
    batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probs = predict(model,
                                                                                                           test_data_loader,
                                                                                                           device)
    print("Prediction done")  # Debug statement
    # logging.debug(f"Epoch {epoch + 1} - Prediction done")

    patient_metrics = calculate_metrics(patient_labels, patient_predictions, patient_probs)
    log_metrics(patient_metrics, 'test', 'ptnt', None)
    print(
        f"Patient-Level Metrics: Acc: {patient_metrics['acc']:.4f}, F1: {patient_metrics['f1']:.4f}, Bal Acc: {patient_metrics['bal_acc']:.4f}")  # Debug statement
    # logging.debug(f"Epoch {epoch + 1} - Patient-Level Metrics: Acc: {patient_metrics['acc']:.4f}, F1: {patient_metrics['f1']:.4f}, Bal Acc: {patient_metrics['bal_acc']:.4f}")

    run.finish()
    print("Prediction finished")  # Debug statement
    # logging.debug(f"Trial {trial.number} complete.")

    return


def main():
    # Set the multiprocessing start method to 'spawn' for Windows compatibility
    mp.set_start_method('spawn', force=True)

    # Initialize Weights & Biases (if you're using it)
    wandb.init(project=parameters['wdb_name'])

    # Create the Optuna study and start the optimization process
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=parameters['num_t'])


if __name__ == '__main__':
    # Setup logger
    # logfile = r"{rd}\{dt}_logfile.log".format(rd=parameters['root_dir'], dt=datetime.now().strftime("%y%m%d_%H-%M))
    # if os.path.isfile(logfile):
    #    os.remove(logfile)
    # logging.basicConfig(level=parameters['log_lvl'],
    #                    format="%(asctime)s - %(levelname)-8s - %(message)s",
    #                    handlers=[
    #                        logging.FileHandler(logfile),
    #                        logging.StreamHandler()
    #                        ]
    #                    )

    # logging.info("Starting predictions for:\t{wn}".format(wn=parameters['wdb_name']))

    parameters = load_parameters(param_path)
    root_dir = parameters['root_dir']
    xlsx_dir = parameters['xlsx_dir']
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CustomImageDataset(os.path.join(parameters['root_dir'], parameters['test_dir']),
                                      parameters['xlsx_dir'], transform=transform)
    test_class_counts = get_class_counts(test_dataset)

    print("\Test set class counts:")
    for label, count in test_class_counts.items():
        print(f"Class {label}: {count} images")
    #    logging.info(f"Validation set class counts {label}: {count} images")

    # Checking for GPU availability
    print("Checking for GPU availability...")
    # logging.info("Checking for GPU availability")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # logging.info(f"Using device: {device}")

    if not os.path.exists('Best models'):
        os.makedirs('Best models')

    # logging.info("Starting main")
    print("Starting main...")
    main()
    # log = profmain.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    # logging.info(f"Profiler info main:\n{log}")
    print("Main complete.")
