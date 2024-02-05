import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

import transforms


class OCDataset(torch.utils.data.Dataset):
    def __init__(self, dir_paths, transform=None, target_transform=None):

        self.classes = [os.path.basename(dir_path) for dir_path in dir_paths]

        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i,
                             cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        for dir_path in dir_paths:
            with os.scandir(dir_path) as image_paths:
                for image_path in image_paths:
                    self.image_paths.append(image_path.path)
                    self.targets.append(os.path.basename(dir_path))

        assert len(self.image_paths) == len(self.targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        label = self.class_to_idx[self.targets[idx]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    dataset = OCDataset([folder.path for folder in os.scandir(r'D:\repos\sorted_new_2023_10_24')],
                                         transform=transforms.ImageTransforms())
    while True:
        image, label = dataset[random.randint(0, len(dataset) - 1)]
        image = np.array(image)
        moved = np.moveaxis(image, 0, -1)
        cv2.imshow(dataset.idx_to_class[label], moved)
        cv2.waitKey(0)
