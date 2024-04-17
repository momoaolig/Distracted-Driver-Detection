import os
import cv2
from tqdm import tqdm
import numpy as np

IMAGE_SIZE = (224,224)
# num: how many images in sequence do you want to load
def load_data(num):
    images = []
    labels = []
    dataset = './sfddd/imgs/train'
    for folder in os.listdir(dataset):
        label = folder
        subdir = f'{dataset}/{folder}'
        for file in tqdm(os.listdir(subdir)[:num]):
            img_path = os.path.join(subdir, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE) # to match ResNet
            images.append(image)
            labels.append(int(label[-1]))
    images = np.array(images)
    labels = np.array(labels)
    return [images, labels]