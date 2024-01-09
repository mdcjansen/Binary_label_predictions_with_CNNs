#!/bin/usr/env python3

import os
import torch
import torchvision.transforms as transforms

from PIL import Image

# Credentials
__author__ = "M.D.C. Jansen"
__version__ = "1.1"
__date__ = "24/11/2023"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.images = os.listdir(input_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.input_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            aug_image = self.transform(image)
            aug_image_name = os.path.splittext(self.images[idx][0] + '_aug' + os.path.splitext(self.images[idx][1]))
            aug_image.save(os.path.join(output_dir, aug_image_name))
        
        return image


def main():
    Transform_param = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor()
        ])

    augment_image_dataset = CustomDataset(input_dir, transform=Transform_param)

    batch_size = 10
    data_loader = torch.utils.data.DataLoader(augment_image_dataset, batch_size=batch_size, shuffle=True)
    os.makedirs(output_dir, exist_ok=True)

    for batch_index, batch in enumerate(data_loader):
        print(f'Batch {batch_index+1}: Processed {len(batch)} images')


if __name__ == '__main__':
    input_dir = r"\\path\to\input\directory"
    output_dir = r"\\path\to\ouput\directory"
    os.makedirs(output_dir, exist_ok=True)

    main()
    
    print("Image Augmentation complete.")
