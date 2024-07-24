import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 100

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
    return images

def preprocess_data(data_dir):
    covid_images = load_images_from_folder(os.path.join(data_dir, 'COVID/images'))
    normal_images = load_images_from_folder(os.path.join(data_dir, 'Normal/images'))
    opacity_images = load_images_from_folder(os.path.join(data_dir, 'Lung_Opacity/images'))
    pneumonia_images = load_images_from_folder(os.path.join(data_dir, 'Viral_Pneumonia/images'))

    covid_labels = [0] * len(covid_images)
    normal_labels = [1] * len(normal_images)
    opacity_labels = [2] * len(opacity_images)
    pneumonia_labels = [3] * len(pneumonia_images)

    images = np.array(covid_images + normal_images + opacity_images + pneumonia_images)
    labels = np.array(covid_labels + normal_labels + opacity_labels + pneumonia_labels)

    images = images / 255.0
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
