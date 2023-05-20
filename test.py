import cv2
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import ortho_classification.oc_model as oc_model
import random

weights_path = 'ortho_classification/face_classification_weights.pth'
path = 'ortho_classification/some_photo'

classes = ['jaw-lower',
           'jaw-upper',
           'mouth-sagittal_fissure',
           'mouth-vestibule-front-closed',
           'mouth-vestibule-front-half_open',
           'mouth-vestibule-half_profile-closed-left',
           'mouth-vestibule-half_profile-closed-right',
           'mouth-vestibule-profile-closed-left',
           'mouth-vestibule-profile-closed-right',
           'portrait']

class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

best_network = oc_model.Network(num_classes=len(classes)).cuda()
best_network.load_state_dict(torch.load(
    weights_path, map_location=torch.device('cuda')))
best_network.eval()

images_path = []
for root, dirs, files in os.walk(path):
    for filename in files:
        images_path.append(os.path.join(path, filename))

for img_path in images_path:
    original_image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    temp_image = grayscale_image
    temp_image = TF.resize(Image.fromarray(temp_image), size=(300, 300))
    temp_image = TF.to_tensor(temp_image)
    temp_image = TF.normalize(temp_image, [0.5], [0.5])
    with torch.no_grad():
        target = best_network(temp_image.unsqueeze(0).cuda())
    target = torch.argmax(torch.softmax(target, dim=1), dim=1)

    plt.figure(figsize=(13, 13))
    plt.imshow(display_image)
    print('predicted:\t', idx_to_class[target.item()], '\nreal:\t', img_path[img_path.rfind('/')+1:img_path.rfind('I')-1], '\n')
    # plt.xlabel(idx_to_class[target.item()])
    plt.show()
