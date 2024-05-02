from torch.utils.data import DataLoader
from MyDatasets import MyDataset
from MyModels import MyModel
from MyTrainings import train_model
import tensorflow as tf
import os
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
>>>>>>> e63ff97 (Adding everything)

def main():
    # Define paths
    train_data_dir = "/workspaces/Health_Squad/sample_data/dataset_complete"
<<<<<<< HEAD
    test_data_dir = "/workspaces/Health_Squad/sample_data/unlabel_sample_data"
=======
    test_data_dir = "/workspaces/Health_Squad/sample_data/dataset_label/bmp_images"
>>>>>>> e63ff97 (Adding everything)
    output_dir = "/workspaces/Health_Squad/bmp_images"

    # Initialize datasets
    train_dataset = MyDataset(train_data_dir)
    test_dataset = MyDataset(test_data_dir)

    # Create TensorFlow datasets
    train_loader = train_dataset.create_dataset(batch_size=1)
    test_loader = test_dataset.create_dataset(batch_size=1)

    # Initialize model
    model = MyModel()

<<<<<<< HEAD
    # Train the model
    train_model(model, train_loader)
=======
    # Train the model and collect metrics
    loss, accuracy = train_model(model, train_loader)
>>>>>>> e63ff97 (Adding everything)

    # Save the model
    model_path = os.path.join(output_dir, "model.h5")
    model.save(model_path)
    print(f"Model saved at {model_path}")

<<<<<<< HEAD
=======
    # Plot loss and accuracy
    plot_metrics(loss, accuracy)

def plot_metrics(loss, accuracy):
    epochs = range(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss, '-o', color=color, markersize=3, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracy, '-o', color=color, markersize=3, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Loss and Accuracy vs Epoch')
    plt.show()

>>>>>>> e63ff97 (Adding everything)
if __name__ == "__main__":
    main()
