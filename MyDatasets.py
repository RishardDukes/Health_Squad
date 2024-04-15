# MyDataset.py

import os
from PIL import Image
import numpy as np

def load_data(data_dir):
    images = []
    labels = []

    # Iterate through BMP files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".bmp"):
            # Load BMP image
            image = Image.open(os.path.join(data_dir, filename))

            # Perform segmentation to identify annotated cells
            annotated_cells = segment_cells(image)

            # Add annotated cells and their labels to the dataset
            for cell in annotated_cells:
                images.append(cell)  # Add segmented cell image
                labels.append(get_label(filename))  # Add label for the cell

    return np.array(images), np.array(labels)

# MyDataset.py

import os
from PIL import Image
import numpy as np

def load_data(data_dir):
    images = []
    labels = []

    # Iterate through BMP files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".bmp"):
            # Load BMP image
            image = Image.open(os.path.join(data_dir, filename))

            # Perform segmentation to identify annotated cells
            annotated_cells = segment_cells(image)

            # Add annotated cells and their labels to the dataset
            for cell in annotated_cells:
                images.append(cell)  # Add segmented cell image
                labels.append(get_label(filename))  # Add label for the cell

    return np.array(images), np.array(labels)

def segment_cells(image):
    # Convert image to grayscale
    gray_image = image.convert("L")

    # Threshold image to get binary mask
    _, binary_image = cv2.threshold(np.array(gray_image), 0, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes from contours
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Extract cells using bounding boxes
    cells = []
    for x, y, w, h in bounding_boxes:
        cell_image = image.crop((x, y, x + w, y + h))
        cells.append(cell_image)

    return cells

def get_label(filename):
    # Extract label from filename (assuming filename format: label_index.bmp)
    label = filename.split("_")[0]
    return label


