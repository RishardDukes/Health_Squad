import os
import cv2
import numpy as np
import tensorflow as tf

class MyDataset:
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
        return np.expand_dims(binary_img, axis=-1)

    def create_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(self.load_and_preprocess_image)
        dataset = dataset.batch(batch_size)
        return dataset

    def load_and_preprocess_image(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=1)  
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image







