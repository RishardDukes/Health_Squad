import logging
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = outputs.float()  # Cast to float
            loss = criterion(outputs, images)  # Reconstruction loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Log loss after every `log_interval` batches
            log_interval = 100  # Adjust as needed
            if (batch_idx + 1) % log_interval == 0:
                current_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)
                logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {current_loss:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    logging.info("Training complete!")
