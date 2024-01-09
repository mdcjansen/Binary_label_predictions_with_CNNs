#!/bin/usr/env python3

import multiprocessing
import os
import random
import torchvision.transforms as transforms

from datetime import datetime
from PIL import Image

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.0"
__date__ = "21/11/2023"


def augment_image(input_images, output_dir, ):
    brightness = random.uniform(0.0, 0.25)
    contrast = random.uniform(0.0, 0.2)
    saturation = random.uniform(0.0, 0.4)
    hue = random.uniform(0.0, 0.5)

    transform_param = transforms.Compose([
        transforms.ColorJitter(brightness=brightness,
                               contrast=contrast,
                               saturation=saturation,
                               hue=hue
                               ),
        transforms.ToTensor()
    ])

    image = Image.open(input_images).convert('RGB')
    aug_image = transform_param(image)
    tensor_to_pil = transforms.ToPILImage()
    aug_image_pil = tensor_to_pil(aug_image)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    in_img_name = os.path.basename(output_dir)
    name_no_suffix, suffix = os.path.splitext(in_img_name)
    transform_values = f"b{brightness:.2f}_c{contrast:.2f}_s{saturation:.2f}_h{hue:.2f}"

    output_img = f"{name_no_suffix}_{transform_values}{suffix}"
    output = os.path.join(os.path.dirname(output_dir), output_img)

    aug_image_pil.save(output)


def get_images(input_dir, output_dir):
    image_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                input_path = os.path.join(root, file)
                output_path = input_path.replace(input_dir, output_dir)
                image_list.append((input_path, output_path))
    return image_list


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}][INFO]:\tStarting colour augmentation")

    input_dir = r"\\path\to\input\directory"
    output_dir = r"\\path\to\ouput\directory"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}][INFO]:\tListing images")
    input_img = get_images(input_dir, output_dir)
    print(f"[{datetime.now().strftime('%H:%M:%S')}][INFO]:\tListed: {len(input_img)} to be augmented")

    print(f"[{datetime.now().strftime('%H:%M:%S')}][INFO]:\tStart augmentation")
    with multiprocessing.Pool() as pool:
        pool.starmap(augment_image, input_img)
        pool.close()
        pool.join()

    print(f"[{datetime.now().strftime('%H:%M:%S')}][INFO]:\tColour augmentation complete")


if __name__ == '__main__':
    main()
