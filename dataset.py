import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

import transforms


class OrthoClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, transform=None, target_transform=None):
        self.classes = ['jaw-lower',
                        'jaw-upper',
                        'mouth-sagittal_fissure',
                        'mouth-vestibule-front-closed',
                        'mouth-vestibule-front-half_open',
                        'mouth-vestibule-half_profile-closed-left',
                        'mouth-vestibule-half_profile-closed-right',
                        'mouth-vestibule-profile-closed-left',
                        'mouth-vestibule-profile-closed-right',
                        'portrait']
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i,
                             cls_name in enumerate(self.classes)}
        self.images = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            self.images = [l.split()[0] for l in lines]
            self.targets = [l.split()[1] for l in lines]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], 0)
        label = self.class_to_idx[self.targets[idx]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    dataset = OrthoClassificationDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation.txt'),
                                         transform=transforms.ImageTransforms())
    while True:
        image, label = dataset[random.randint(0, len(dataset))]
        image = np.array(image)
        moved = np.moveaxis(image, 0, -1)
        plt.figure(figsize=(13, 13))
        plt.xlabel(dataset.idx_to_class[label])
        plt.imshow(moved, cmap='gray')
        plt.show()
