import random
import numpy as np
from PIL import Image
import imutils
import torchvision.transforms.functional as TF
from torchvision import transforms


class ImageTransforms():
    def __init__(self):
        pass

    def rotate(self, image, angle):
        angle = random.uniform(-angle, +angle)
        image = imutils.rotate(np.array(image), angle)
        return Image.fromarray(image)

    def resize(self, image, img_size):
        image = TF.resize(image, img_size)
        return image

    def color_jitter(self, image):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image

    def __call__(self, image):
        image = Image.fromarray(image)
        image = self.resize(image, (300, 300))
        image = self.color_jitter(image)
        image = self.rotate(image, angle=100)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image
