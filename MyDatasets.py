# MyDatasets.py
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        # Preprocess the image
        preprocessed_img = self.preprocess(image)
        return preprocessed_img

    def preprocess(self, image):
        # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold to separate black elements from other areas
        _, threshold_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
        # Convert to binary image
        binary_img = threshold_img / 255
        # Expand dimensions to match expected input shape
        return np.expand_dims(binary_img, axis=0)






