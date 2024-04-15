# MyTrainings.py

import tensorflow as tf
from MyModels import segmentation_model, classification_model
from MyDataset import load_data

# Load data
data_dir = "path/to/annotated_data"
images, labels = load_data(data_dir)

# Define model
input_shape = images.shape[1:]  # Shape of the input images
num_classes = len(set(labels))  # Number of classes (cell types)

# Choose segmentation or classification model
model = segmentation_model(input_shape, num_classes)
# model = classification_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
