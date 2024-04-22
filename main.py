# main.py
from torch.utils.data import DataLoader
from MyDatasets import MyDataset
from MyModels import MyModel
from MyTrainings import train_model
import torch
import os
import logging

def main():
    # Define paths
    train_data_dir = "/workspaces/Health_Squad/sample_data/dataset_complete"
    test_data_dir = "/workspaces/Health_Squad/sample_data/unlabel_sample_data"
    output_dir = "/workspaces/Health_Squad/bmp_images"

    # Debug prints for dataset initialization
    print("Initializing datasets...")
    train_dataset = MyDataset(train_data_dir)
    test_dataset = MyDataset(test_data_dir)
    print("Datasets initialized.")

    # Debug prints for dataloader initialization
    print("Initializing dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Dataloaders initialized.")

    # Debug prints for model initialization
    print("Initializing model...")
    model = MyModel()
    print("Model initialized.")

    # Debug prints for training
    print("Training model...")
    try:
        train_model(model, train_loader)
    except Exception as e:
        logging.exception("Exception occurred during training:")

    print("Model trained.")

    # Debug prints for model saving
    print("Saving model...")
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
