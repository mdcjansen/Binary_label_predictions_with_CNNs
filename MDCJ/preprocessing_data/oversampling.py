import os
import pandas as pd
import numpy as np
import cv2
import shutil
import random

train_save_dir = r"D:\path\to\save\directory"
excel_file = 'D:\path\to\BRS-label\xlsx'
output_dir = r"D:\path\to\output\directory"

df = pd.read_excel(excel_file)
study_id_to_label = dict(zip(df["study_id"], df["label"]))

images = []
labels = []
for root, _, files in os.walk(train_save_dir):
    for file in files:
        if file.endswith('.jpg'):
            study_id = int(file.split('_')[1])
            if study_id in study_id_to_label:
                labels.append(study_id_to_label[study_id])
                images.append(os.path.join(root, file))

class_counts = {label: labels.count(label) for label in set(labels)}
print(f"Class distribution before oversampling: {class_counts}")

majority_class = 0
minority_class = 1

num_oversample_majority = int(class_counts[majority_class] * 0.0)
num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])

flip_history = set()

def random_flip_save(image_list, output_dir, counter, class_type):
    while True:
        img_path = np.random.choice(image_list, 1)[0]
        
        img = cv2.imread(img_path)
    
        if (f"{img_path}_hflip" in flip_history) and (f"{img_path}_vflip" in flip_history):
            continue
        
        rand_flip_no = np.random.randint(-1, 1)
        
        if  rand_flip_no == -1:
            flip_type = 'hvflip'
            flip_key = f"{img_path}_{flip_type}"
        if  rand_flip_no == 0:
            flip_type = 'vflip'
            flip_key = f"{img_path}_{flip_type}"
        if rand_flip_no == 1:
            flip_type = 'hflip'
            flip_key = f"{img_path}_{flip_type}"

        flip_history.add(flip_key)

        if flip_type == 'hvflip':
            img = cv2.flip(img, -1)
        elif flip_type == 'vflip':
            img = cv2.flip(img, 0)
        elif flip_type == 'hflip':
            img = cv2.flip(img, 1)

        flipped_img_name = f"{os.path.basename(img_path).replace('.jpg', '')}_{flip_type}_{counter}.jpg"
        output_path = os.path.join(output_dir, flipped_img_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    
        return True

majority_images = [img for img, label in zip(images, labels) if label == majority_class]
minority_images = [img for img, label in zip(images, labels) if label == minority_class]

oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)

counter = 0
for _ in oversampled_majority:
    output_folder = os.path.join(output_dir, str(majority_class))
    random_flip_save(majority_images, output_folder, counter, majority_class)
    counter += 1

for _ in oversampled_minority:
    output_folder = os.path.join(output_dir, str(minority_class))
    random_flip_save(minority_images, output_folder, counter, minority_class)
    counter += 1

for img_path in images:
    output_path = img_path.replace(train_save_dir, output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(img_path, output_path)

class_counts[majority_class] += num_oversample_majority
class_counts[minority_class] += num_oversample_minority
print(f"Class distribution after oversampling: {class_counts}")
