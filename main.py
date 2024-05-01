from torch.utils.data import DataLoader
from MyDatasets import MyDataset
from MyModels import MyModel
from MyTrainings import train_model
import tensorflow as tf
import os

def main():
    # Define paths
    train_data_dir = "/workspaces/Health_Squad/sample_data/dataset_complete"
    test_data_dir = "/workspaces/Health_Squad/sample_data/unlabel_sample_data"
    output_dir = "/workspaces/Health_Squad/bmp_images"

    # Initialize datasets
    train_dataset = MyDataset(train_data_dir)
    test_dataset = MyDataset(test_data_dir)

    # Create TensorFlow datasets
    train_loader = train_dataset.create_dataset(batch_size=1)
    test_loader = test_dataset.create_dataset(batch_size=1)

    # Initialize model
    model = MyModel()

    # Train the model
    train_model(model, train_loader)

    # Save the model
    model_path = os.path.join(output_dir, "model.h5")
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
