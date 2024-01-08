#!/usr/bin/env python3

import csv
import numpy as np
import logging
import optuna
import os
import pandas as pd
import random
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
#from time import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import sigmoid
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import inception_v3
#from torchvision.datasets import DatasetFolder

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.15"
__date__ = "05/12/2023"

# Parameter file path
param_path = r"D:\path\to\parameter\file.csv"


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, xlsx_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df_labels = pd.read_excel(xlsx_dir)
        self.all_images = [os.path.join(subdir, file)
                           for subdir, dirs, files in os.walk(self.root_dir)
                           for file in files if file.lower().endswith(('.png',
                                                                       '.jpg',
                                                                       '.jpeg'
                                                                       ))]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        segments = img_name.split('_')

        if len(segments) <= 1:
            raise ValueError(f"Unexpected image name format for {img_name}")

        study_id = segments[1]
        label_entries = self.df_labels[self.df_labels['study_id'] == int(study_id)]['label'].values

        if len(label_entries) == 0:
            print(f"No label found for study_id: {study_id} in image {img_name}.")
            label = None
        else:
            label = label_entries[0]

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, study_id


class CustomHead(nn.Module):
    def __init__(self, input_size, output_size, batch_norm, dropout_rate):
        super(CustomHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)

        x = self.fc(x)
        x = self.dropout(x)

        return x


def generate_filename(prefix, run_name, config, criterion, epoch):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        os.path.join("Best models",
                                     f"{prefix}_{run_name}_epoch_{epoch + 1}_{criterion}.pt"
                                     ))


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
        'train_dir': ''.join(input_param['train_dirname']),
        'val_dir': ''.join(input_param['val_dirname']),
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


def custom_transform(image):
    rd_colour_change = random.random() < 0.3

    if rd_colour_change:
        brightness = random.uniform(0.0, 0.25)
        contrast = random.uniform(0.0, 0.2)
        saturation = random.uniform(0.0, 0.4)
        hue = random.uniform(0.0, 0.5)
        image = transforms.functional.adjust_brightness(image, brightness)
        image = transforms.functional.adjust_contrast(image, contrast)
        image = transforms.functional.adjust_saturation(image, saturation)
        image = transforms.functional.adjust_hue(image, hue)

    return transforms.functional.to_tensor(image)


def load_img_label(dataset):
    images = []
    labels = []
    ids = []

    for idx, (img, label, study_id) in enumerate(dataset):
        images.append(img)
        labels.append(label)
        ids.append(study_id)

    return images, labels, ids


def get_class_counts(dataset):
    print("[{time}][INFO]:\tCounting classes for dataset"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    class_counts = defaultdict(int)

    for idx, (img, label, _) in enumerate(dataset):
        try:
            class_counts[label] += 1
            if idx == len(dataset) - 1:
                print(f"Last index processed: {idx}")
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")

    return class_counts


def calculate_metrics(y_true, y_pred, y_proba):
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

    for metric, func in [('f1', f1_score),
                         ('precision', precision_score),
                         ('recall', recall_score
                          )]:
        try:
            metrics[metric] = func(y_true, y_pred)
        except:
            pass

    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    metrics['roc_auc'] = auc(fpr, tpr)

    try:
        metrics['cm'] = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print("[{time}][ERROR]:\tError computing confusion matrix: {e}"
              .format(time=datetime.now().strftime("%H:%M:%S"),
                      e=e
                      )
              )
        metrics['cm'] = None

    return metrics


def log_metrics(metrics, split, prefix, loss):
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


def create_model(batch_norm, dropout_rate):
    model = inception_v3(pretrained=True, aux_logits=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = CustomHead(num_ftrs, 1, batch_norm, dropout_rate)
    aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_ftrs, 1)

    for param in model.fc.parameters():
        param.requires_grad = True

    for param in model.AuxLogits.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    return model, "InceptionV3"


def train(model, train_data_loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_logits = []
    optimizer.zero_grad()

    for i in range(len(train_images)):
        images = torch.tensor(train_images[i],
                              dtype=torch.float32
                              ).to(device)
        labels = torch.tensor(train_images[i],
                              dtype=torch.long
                              ).to(device)
        with autocast():
            outputs, aux_outputs = model(images)
            loss1 = F.binary_cross_entropy_with_logits(outputs.squeeze(),
                                                       labels.float()
                                                       )
            loss2 = F.binary_cross_entropy_with_logits(aux_outputs.squeeze(),
                                                       labels.float()
                                                       )
            loss = loss1 + 0.4 * loss2
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=1
                                       )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        all_predictions.extend((torch.sigmoid(outputs.squeeze()) > 0.5)
                               .long()
                               .tolist()
                               )
        all_labels.extend(labels.tolist())
        y_logits.extend(outputs.squeeze()
                        .detach()
                        .cpu()
                        .numpy())
        total_loss += loss.item() * images.size(0)

    scheduler.step()
    metrics = calculate_metrics(all_labels,
                                all_predictions,
                                y_logits)
    _, _, train_patient_predictions, train_patient_labels, _, train_patient_probs = predict(model,
                                                                                            train_data_loader,
                                                                                            device
                                                                                            )
    train_patient_metrics = calculate_metrics(train_patient_labels,
                                              train_patient_predictions,
                                              train_patient_probs
                                              )
    print("[{time}][INFO]:\tTraining complete."
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    return total_loss / len(train_data_loader.dataset), metrics, train_patient_metrics


def validate(model, val_data_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_proba = []

    with torch.no_grad():
        for i in range(len(val_images)):
            images = torch.tensor(train_images[i],
                                  dtype=torch.float32
                                  ).to(device)
            labels = torch.tensor(train_images[i],
                                  dtype=torch.long
                                  ).to(device)
            with autocast():
                outputs = model(images)
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(),
                                                          labels.float()
                                                          )
            total_loss += loss.item() * images.size(0)
            all_predictions.extend((torch.sigmoid(outputs.squeeze()) > 0.5
                                    ).long().tolist())
            all_labels.extend(labels.tolist())
            y_proba.extend(torch.sigmoid(outputs.squeeze()
                                         .detach()
                                         .cpu()
                                         .float())
                           .numpy())
        metrics = calculate_metrics(all_labels,
                                    all_predictions,
                                    y_proba)
        print("[{time}][INFO]:\tValidation complete."
              .format(time=datetime.now().strftime("%H:%M:%S")
                      )
              )

    return total_loss / len(val_data_loader.dataset), metrics


def predict(model, data_loader, device):
    print("[{time}][INFO]:\tStarting prediction"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    model.eval()
    model = model.to(device)
    batch_predictions = []
    batch_labels = []
    study_summary = {}
    y_proba = []

    with torch.no_grad():
        for i in range(len(val_images)):
            inputs, labels = val_images[i].to(device), val_labels[i].to(device)
            outputs = model(inputs)
            output = outputs.view(-1)
            probs = torch.sigmoid(output).detach().cpu().numpy()
            y_proba.extend(probs)
            batch_predictions.extend((probs > 0.5)
                                     .astype(int)
                                     .tolist()
                                     )
            batch_labels.extend(labels
                                .tolist()
                                )
            for prediction, probability, label, study_id in zip(batch_predictions,
                                                                probs,
                                                                batch_labels,
                                                                study_ids):
                if study_id not in study_summary:
                    study_summary[study_id] = {'predictions': [],
                                               'probs': [],
                                               'labels': []
                                               }
                study_summary[study_id]['predictions'].append(prediction)
                study_summary[study_id]['probs'].append(probability)
                study_summary[study_id]['labels'].append(label)
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

    print("[{time}][INFO]:\tPrediction complete"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )

    return batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probabilities


def objective(trial):
    print("[{time}][INFO]:\tStarting trial: {trialnum}"
          .format(time=datetime.now().strftime("%H:%M:%S"),
                  trialnum=trial.number
                  )
          )
    run = wandb.init(project=parameters['wdb_name'], config={})
    wandb.save(parameters['wdb_save'])
    config = run.config
    trial_lr = trial.suggest_categorical('lr',
                                         parameters['tl_lr']
                                         )
    trial_batch_norm = trial.suggest_categorical('batch_norm',
                                                 parameters['tl_bn']
                                                 )
    trial_dropout_rate = trial.suggest_categorical('dropout_rate',
                                                   parameters['tl_dr']
                                                   )

    run.config.update({
        "lr": trial_lr,
        "batch_size": trial.suggest_categorical('batch_size',
                                                parameters['tl_bs']
                                                ),
        "num_epochs": parameters['num_e'],
        "weight_decay": trial.suggest_float('weight_decay',
                                            parameters['tl_wd_min'],
                                            parameters['tl_wd_max'],
                                            log=True
                                            ),
        "step_size": 5,
        "gamma": trial.suggest_float('gamma',
                                     parameters['tl_ga_min'],
                                     parameters['tl_ga_max'],
                                     step=parameters['tl_ga_tp']
                                     ),
        "batch_norm": trial_batch_norm,
        "dropout_rate": trial_dropout_rate
    }, allow_val_change=True)

    mp.set_start_method('spawn', force=True)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=parameters['dl_work'],
                                   pin_memory=True
                                   )
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=parameters['dl_work'],
                                 pin_memory=True
                                 )
    model, model_name = create_model(config["batch_norm"],
                                     config["dropout_rate"]
                                     )
    run.config.update({"model_name": model_name},
                      allow_val_change=True
                      )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_bal_acc = float('-inf')
    early_stop_counter = parameters['es_count']
    early_stop_limit = parameters['es_limit']

    for epoch in range(config["num_epochs"]):
        print("[{time}][INFO]:\tEpoch {eh1} - Start training"
              .format(time=datetime.now().strftime("%H:%M:%S"),
                      eh1=epoch + 1)
              )
        with profiler.profile(record_shapes=True) as proftrain:
            training_loss, training_metrics, train_patient_metrics = train(model, train_data_loader, optimizer,
                                                                           scheduler, device, scaler)
        print("[{time}][INFO TRAIN]:\n\n"
              .format(time=datetime.now().strftime("%H:%M:%S")
                      ),
              proftrain.key_averages().table(sort_by="self_cpu_time_total",
                                             row_limit=15
                                             )
              )
        print(f"[INFO]:\tEpoch {epoch + 1} - Training Loss: {training_loss:.4f}")
        log_metrics(training_metrics, 'train', 'img', training_loss)
        log_metrics(train_patient_metrics, 'train', 'ptnt', None)
        print("[{time}][INFO]:\tEpoch {eh1} - Start validation"
              .format(time=datetime.now().strftime("%H:%M:%S"),
                      eh1=epoch + 1
                      )
              )
        with profiler.profile(record_shapes=True) as profvald:
            validation_loss, validation_metrics = validate(model, val_data_loader, device)
        print("[INFO VALD]:\n\n", profvald.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        print(f"[INFO]:\tEpoch {epoch + 1} - Validation Loss: {validation_loss:.4f}")
        bal_acc = validation_metrics['bal_acc']
        is_best_loss = validation_loss < best_val_loss
        if is_best_loss:
            best_val_loss = validation_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), generate_filename('best_model', run.name, config, 'loss', epoch))
            with open('model_parameters.csv', 'a+') as model_csv:
                ml_params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
                model_csv.write(f"{run.name},loss,epoch={epoch + 1},{','.join(ml_params)}\n")
            model_csv.close()
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_limit:
                print('[{time}][INFO]:\tEarly stopping triggered\n\n\n'
                      .format(time=datetime.now().strftime("%H:%M:%S")
                              )
                      )
                break
        is_best_bal_acc = bal_acc > best_bal_acc
        if is_best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), generate_filename('best_model', run.name, config, 'bal_acc', epoch))
            with open('model_parameters.csv', 'a+') as model_csv:
                ml_params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
                model_csv.write(f"{run.name},bal_acc,epoch={epoch + 1},{','.join(ml_params)}\n")
            model_csv.close()

        log_metrics(validation_metrics, 'val', 'img', validation_loss)
        print(f"Epoch {epoch + 1} -",
              "Validation Metrics: Acc: {validation_metrics['acc']:.4f},",
              "F1: {validation_metrics['f1']:.4f},",
              "Bal Acc: {bal_acc:.4f}")
        print("[{time}][INFO]:\tEpoch {eh1} - Start prediction"
              .format(time=datetime.now().strftime("%H:%M:%S"),
                      eh1=epoch + 1
                      )
              )
        batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probs = predict(model,
                                                                                                               val_data_loader,
                                                                                                               device
                                                                                                               )
        print("[{time}][INFO]:\tEpoch {eh1} - Prediction done"
              .format(time=datetime.now().strftime("%H:%M:%S"),
                      eh1=epoch + 1
                      )
              )
        patient_metrics = calculate_metrics(patient_labels,
                                            patient_predictions,
                                            patient_probs
                                            )
        log_metrics(patient_metrics, 'val', 'ptnt', None)
        print(f"Epoch {epoch + 1} -",
              "Patient-Level Metrics: Acc: {patient_metrics['acc']:.4f},",
              "F1: {patient_metrics['f1']:.4f},",
              "Bal Acc: {patient_metrics['bal_acc']:.4f}"
              )

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    run.finish()
    torch.cuda.empty_cache()
    print("[{time}][INFO]:\tTrial {tnum} complete"
          .format(time=datetime.now().strftime("%H:%M:%S"),
                  tnum=trial.number)
          )

    return best_val_loss


def main():
    mp.set_start_method('spawn', force=True)
    wandb.init(project=parameters['wdb_name'])
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=parameters['num_t'])


if __name__ == '__main__':
    parameters = load_parameters(param_path)
    root_dir = parameters['root_dir']
    xlsx_dir = parameters['xlsx_dir']
    transform = transforms.Compose([custom_transform])
    train_dataset = CustomImageDataset(os.path.join(parameters['root_dir'], parameters['train_dir']),
                                       parameters['xlsx_dir'], transform=transform)
    val_dataset = CustomImageDataset(os.path.join(parameters['root_dir'], parameters['val_dir']),
                                     parameters['xlsx_dir'], transform=transform)
    train_class_counts = get_class_counts(train_dataset)
    val_class_counts = get_class_counts(val_dataset)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    print("[{time}][INFO]:\tLoading training images"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    train_images, train_labels, study_ids = load_img_label(train_dataset)
    print("[{time}][INFO]:\tloaded training images"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    print("[{time}][INFO]:\tLoading validation images"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    val_images, val_labels, study_ids = load_img_label(val_dataset)
    print("[{time}][INFO]:\tLoaded validation images"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    print("[{time}][INFO]:\tTraining set class counts:"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )

    for label, count in train_class_counts.items():
        print(f"Class {label}: {count} images")

    print("\n[{time}][INFO]:\tValidation set class counts:"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )

    for label, count in val_class_counts.items():
        print(f"Class {label}: {count} images")

    print("[{time}][INFO]:\tChecking for GPU availability"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[{time}][INFO]:\tUsing device:"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  ),
          device
          )

    if not os.path.exists('Best models'):
        os.makedirs('Best models')

    print("[{time}][INFO]:\tStarting main"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
    main()
    print("[{time}][INFO]:\tMain complete"
          .format(time=datetime.now().strftime("%H:%M:%S")
                  )
          )
