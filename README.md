# Orthodontic Images Classification
This project is about classifying orthodontic images into different categories based on the type photo. The goal was to build a machine learning model that can classify orthodontic images to help orthodontists categorize them.
## Data
The dataset consists more than 4700 orthodontic images, divided into ten categories and stored in separate folders: 
- jaw-lower
- jaw-upper
- mouth-sagittal_fissure
- mouth-vestibule-front-closed
- mouth-vestibule-front-half_open
- mouth-vestibule-half_profile-closed-left
- mouth-vestibule-half_profile-closed-right
- mouth-vestibule-profile-closed-left
- mouth-vestibule-profile-closed-right
- portrait
## Methodology
To solve the classification problem, we used a deep learning approach based on convolutional neural networks (CNNs), which can automatically extract features from images and classify them. We used a pre-trained CNN model called ResNet-50, which has been proven to perform well on various image recognition tasks. We fine-tuned the model on our orthodontic images dataset using transfer learning and data augmentation techniques. We also used a cross-validation strategy to evaluate the model’s performance and avoid overfitting.
