import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import debris_collate_fn

class Trainer:
    """Handles the training process for the CNN-LSTM model."""

    def __init__(self, model, train_dataset, val_dataset, batch_size, criterion, optimizer, learning_rate):
        """Initializes the Trainer with model, datasets, batch size, loss function, optimizer, and learning rate."""
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=debris_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=debris_collate_fn)
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)

    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()  # Set the model to training mode
        epoch_loss = 0.0
        for batch in self.train_loader:
            images = batch['images']
            sequence_lengths = batch['sequence_lengths']
            targets = ... # You need to define how targets are retrieved or computed from the batch

            self.optimizer.zero_grad()  # Zero the parameter gradients

            # Packing the padded sequences
            packed_images = pack_padded_sequence(images, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs = self.model(packed_images)  # Forward pass

            loss = self.criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Optimize the model

            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def validate(self):
        """Validates the model on the validation dataset."""
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for batch in self.val_loader:
                images = batch['images']
                sequence_lengths = batch['sequence_lengths']
                targets = ... # You need to define how targets are retrieved or computed from the batch

                # Packing the padded sequences
                packed_images = pack_padded_sequence(images, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
                outputs = self.model(packed_images)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def train(self, num_epochs):
        """Trains the model for a specified number of epochs."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()  # Train the model for one epoch
            val_loss = self.validate()  # Validate the model
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    def evaluate(self, test_dataset):
        """
        Evaluates the model on the test dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): The test dataset.

        Returns:
            float: The average loss on the test dataset.
            Any: The performance metric(s) for the model on the test dataset.
        """
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=debris_collate_fn)
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        # Define additional metrics here if needed

        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                images = batch['images']
                sequence_lengths = batch['sequence_lengths']
                targets = batch['targets']  # Assuming that you have targets in your batch dictionary

                # Packing the padded sequences
                packed_images = pack_padded_sequence(images, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
                outputs = self.model(packed_images)

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                # Calculate additional metrics here if needed

        avg_test_loss = test_loss / len(test_loader)
        # Calculate and return the final metrics here

        # For demonstration, let's return the average loss
        return avg_test_loss