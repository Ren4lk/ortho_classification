import cv2
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ortho_classification_model import OCModel


def test(model, img_path, classes):
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    display_img = cv2.imread(img_path)
    img = Image.fromarray(display_img)

    img = TF.resize(img, size=(300, 300))
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        predicted_target = model(img.unsqueeze(0).to(device))

        predicted_prob = F.softmax(predicted_target, dim=1)
        _, predicted_idx = torch.max(predicted_prob, dim=1)

    print(
        f'predicted: {idx_to_class[predicted_idx.item()]},  path: {img_path}')
    # cv2.imshow('test', display_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    classes = ['mouth-vestibule-front-half_open',
               'mouth-vestibule-half_profile-closed-left',
               'mouth-vestibule-half_profile-closed-right',
               'mouth-vestibule-profile-closed-left']

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = r'D:\repos\ortho_classification\resnet18_ortho_classification_weights_epoch_1.pth'

    model = OCModel(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device(device)))
    model.eval()

    test(model=model,
         img_path=r'D:\repos\sorted_new_2023_10_24_test\mouth-vestibule-half_profile-closed-left\IMG_0263.JPG',
         classes=classes)
